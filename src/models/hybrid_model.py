from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .tokenizer import get_tokenizer

MAX_POSITION_EMBEDDINGS = 1024


class SelectiveContextAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        local_window: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.local_window = local_window
        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.selectivity = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

        self.local_weight = nn.Parameter(torch.tensor(0.5))
        self.global_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        selectivity_scores = torch.sigmoid(self.selectivity(x))
        selectivity_scores = selectivity_scores.transpose(1, 2).unsqueeze(2)
        attn_modulated = attn_scores * (0.2 + 0.8 * selectivity_scores)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            attn_modulated = attn_modulated + attention_mask

        i_idx = torch.arange(T, device=x.device).unsqueeze(1)
        j_idx = torch.arange(T, device=x.device).unsqueeze(0)
        outside_local = j_idx < (i_idx - self.local_window).clamp(min=0)
        attn_local = attn_scores.masked_fill(
            outside_local.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_global = attn_modulated

        w_local = torch.sigmoid(self.local_weight)
        w_global = torch.sigmoid(self.global_weight)
        w_sum = w_local + w_global
        w_local, w_global = w_local / w_sum, w_global / w_sum

        attn_local = F.softmax(attn_local, dim=-1)
        attn_global = F.softmax(attn_global, dim=-1)
        attn_local = torch.nan_to_num(attn_local, nan=0.0, posinf=0.0, neginf=0.0)
        attn_global = torch.nan_to_num(attn_global, nan=0.0, posinf=0.0, neginf=0.0)

        attn_combined = w_local * attn_local + w_global * attn_global
        attn_combined = self.attn_dropout(attn_combined)

        out = torch.matmul(attn_combined, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


def _make_mamba_block(hidden_size: int, layer_idx: int):
    from transformers import MambaConfig
    from transformers.models.mamba.modeling_mamba import MambaBlock
    cfg = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
    if cfg.hidden_size != hidden_size:
        cfg.hidden_size = hidden_size
        cfg.intermediate_size = hidden_size * 4
    return MambaBlock(cfg, layer_idx=layer_idx)


class MambaSelectiveContextAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        local_window: int = 256,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.local_window = local_window
        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.mamba_selector = _make_mamba_block(hidden_size, layer_idx)
        self.mamba_input_ln = nn.LayerNorm(hidden_size)
        self.selectivity_proj = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / (2 * 12)**0.5)
        nn.init.zeros_(self.c_proj.bias)

        nn.init.normal_(self.selectivity_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.selectivity_proj.bias)

        self.local_weight = nn.Parameter(torch.tensor(0.5))
        self.global_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        x_for_mamba = self.mamba_input_ln(x)
        mamba_features = self.mamba_selector(x_for_mamba)
        mamba_features = torch.clamp(mamba_features, -20.0, 20.0)

        selectivity_logits = self.selectivity_proj(mamba_features)
        selectivity_scores = torch.sigmoid(selectivity_logits)

        selectivity_scores = selectivity_scores.transpose(1, 2).unsqueeze(2)
        attn_modulated = attn_scores * (0.5 + 0.5 * selectivity_scores)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            attn_modulated = attn_modulated + attention_mask

        i_idx = torch.arange(T, device=x.device).unsqueeze(1)
        j_idx = torch.arange(T, device=x.device).unsqueeze(0)
        outside_local = j_idx < (i_idx - self.local_window).clamp(min=0)
        attn_local = attn_scores.masked_fill(
            outside_local.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_global = attn_modulated

        w_local = torch.sigmoid(self.local_weight)
        w_global = torch.sigmoid(self.global_weight)
        w_sum = w_local + w_global
        w_local, w_global = w_local / w_sum, w_global / w_sum

        attn_local = F.softmax(attn_local, dim=-1)
        attn_global = F.softmax(attn_global, dim=-1)
        attn_local = torch.nan_to_num(attn_local, nan=0.0, posinf=0.0, neginf=0.0)
        attn_global = torch.nan_to_num(attn_global, nan=0.0, posinf=0.0, neginf=0.0)

        attn_combined = w_local * attn_local + w_global * attn_global
        attn_combined = self.attn_dropout(attn_combined)

        out = torch.matmul(attn_combined, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class SelectiveContextBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        local_window: int = 256,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = SelectiveContextAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            local_window=local_window,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(hidden_size)
        intermediate = intermediate_size or 4 * hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MambaSelectiveContextBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        local_window: int = 256,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = MambaSelectiveContextAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            local_window=local_window,
            dropout=dropout,
            layer_idx=layer_idx,
        )
        self.ln_2 = nn.LayerNorm(hidden_size)
        intermediate = intermediate_size or 4 * hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_size),
            nn.Dropout(dropout),
        )

        nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.02 / (2 * 12)**0.5)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Gpt2MambaSelectiveBlock(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, dropout: float = 0.0, layer_idx: int = 0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.num_heads = num_heads

        self.mamba = _make_mamba_block(hidden_size, layer_idx)
        self.mamba_ln = nn.LayerNorm(hidden_size)
        self.selectivity_proj = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.selectivity_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.selectivity_proj.bias)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.ln_1(x)
        B, T, C = x_norm.shape
        qkv = self.c_attn(x_norm)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        mamba_out = self.mamba(self.mamba_ln(x))
        mamba_out = torch.clamp(mamba_out, -20.0, 20.0).detach()
        selectivity = torch.sigmoid(self.selectivity_proj(mamba_out))
        selectivity = selectivity.unsqueeze(1)
        v_mod = v * (0.3 + 0.7 * selectivity)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
        attn_probs = self.attn_dropout(attn_probs)
        out = torch.matmul(attn_probs, v_mod)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        x = residual + out

        x = x + self.mlp(self.ln_2(x))
        return x


class HybridMambaTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(MAX_POSITION_EMBEDDINGS, hidden_size)

        from transformers import MambaConfig, GPT2Config
        from transformers.models.mamba.modeling_mamba import MambaBlock
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        mamba_config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
        if mamba_config.hidden_size != hidden_size:
            mamba_config.hidden_size = hidden_size
            mamba_config.intermediate_size = hidden_size * 4
        gpt2_config = GPT2Config.from_pretrained("gpt2")
        gpt2_config.n_embd = hidden_size
        gpt2_config.n_inner = None
        gpt2_config.n_layer = 1
        gpt2_config._attn_implementation = "eager"

        self.gpt2_layers = nn.ModuleList([GPT2Block(gpt2_config, layer_idx=i) for i in range(12)])
        self.mamba_branch = nn.Sequential(
            MambaBlock(mamba_config, layer_idx=0),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.mamba_gate = nn.Parameter(torch.tensor([-4.0]))
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], device: torch.device
    ):
        batch, seq_len = input_ids.shape
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)
        causal = causal.masked_fill(causal.bool(), float("-inf"))
        if attention_mask is not None:
            padding = torch.zeros(batch, 1, 1, seq_len, device=device)
            padding = padding.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
            return causal[None, None, :, :] + padding
        return causal[None, None, :, :].expand(batch, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        seq_len = input_ids.size(1)
        device = input_ids.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(position_ids)

        attn_mask_4d = self._make_attention_mask(input_ids, attention_mask, device)
        h_attn = x
        for layer in self.gpt2_layers:
            h_attn = layer(h_attn, attention_mask=attn_mask_4d, use_cache=False)[0]

        h_mamba = self.mamba_branch[0](x)
        h_mamba = self.mamba_branch[1](h_mamba)
        h_mamba = self.mamba_branch[2](h_mamba)
        scale = 0.1 * torch.sigmoid(self.mamba_gate)
        h = h_attn + scale * h_mamba
        h = self.ln_f(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"loss": loss, "logits": logits})()


def _copy_pretrained_weights(model: HybridMambaTransformer) -> None:
    from transformers import GPT2LMHeadModel

    model_id = os.environ.get("GPT2_BASE_MODEL", "gpt2")
    gpt2_src = GPT2LMHeadModel.from_pretrained(model_id)
    gpt2_src.eval()

    model.embed.weight.data.copy_(gpt2_src.transformer.wte.weight.data)
    model.pos_embed.weight.data.copy_(gpt2_src.transformer.wpe.weight.data)
    for i, layer in enumerate(model.gpt2_layers):
        layer.load_state_dict(gpt2_src.transformer.h[i].state_dict())
    model.ln_f.weight.data.copy_(gpt2_src.transformer.ln_f.weight.data)
    model.ln_f.bias.data.copy_(gpt2_src.transformer.ln_f.bias.data)
    model.lm_head.weight.data.copy_(gpt2_src.lm_head.weight.data)

    nn.init.zeros_(model.mamba_branch[1].weight)
    nn.init.zeros_(model.mamba_branch[1].bias)

    del gpt2_src


def freeze_gpt2_components(model: HybridMambaTransformer) -> None:
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.pos_embed.parameters():
        param.requires_grad = False
    for param in model.gpt2_layers.parameters():
        param.requires_grad = False
    for param in model.ln_f.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False


def _layer_forward(layer: nn.Module, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    return layer(x, attention_mask=attn_mask)


class Gpt2MambaSelectiveTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_checkpointing = use_checkpointing
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(MAX_POSITION_EMBEDDINGS, hidden_size)
        self.layers = nn.ModuleList([
            Gpt2MambaSelectiveBlock(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, layer_idx=i)
            for i in range(12)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], device: torch.device
    ):
        batch, seq_len = input_ids.shape
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)
        causal = causal.masked_fill(causal.bool(), float("-inf"))
        if attention_mask is not None:
            padding = torch.zeros(batch, 1, 1, seq_len, device=device)
            padding = padding.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
            return causal[None, None, :, :] + padding
        return causal[None, None, :, :].expand(batch, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        seq_len = input_ids.size(1)
        device = input_ids.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(position_ids)
        attn_mask = self._make_attention_mask(input_ids, attention_mask, device)

        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    _layer_forward, layer, x, attn_mask, use_reentrant=False
                )
            else:
                x = layer(x, attention_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return type("Output", (), {"loss": loss, "logits": logits})()


def _copy_gpt2_to_mamba_selective(model: Gpt2MambaSelectiveTransformer) -> None:
    from transformers import GPT2LMHeadModel
    model_id = os.environ.get("GPT2_BASE_MODEL", "gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained(model_id)
    model.embed.weight.data.copy_(gpt2.transformer.wte.weight)
    model.pos_embed.weight.data.copy_(gpt2.transformer.wpe.weight)
    model.ln_f.weight.data.copy_(gpt2.transformer.ln_f.weight)
    model.ln_f.bias.data.copy_(gpt2.transformer.ln_f.bias)
    model.lm_head.weight.data.copy_(gpt2.lm_head.weight)
    for i, layer in enumerate(model.layers):
        src = gpt2.transformer.h[i]
        layer.ln_1.weight.data.copy_(src.ln_1.weight)
        layer.ln_1.bias.data.copy_(src.ln_1.bias)
        w = src.attn.c_attn.weight
        layer.c_attn.weight.data.copy_(w.T if w.shape != layer.c_attn.weight.shape else w)
        layer.c_attn.bias.data.copy_(src.attn.c_attn.bias)
        w = src.attn.c_proj.weight
        layer.c_proj.weight.data.copy_(w.T if w.shape != layer.c_proj.weight.shape else w)
        layer.c_proj.bias.data.copy_(src.attn.c_proj.bias)
        layer.ln_2.weight.data.copy_(src.ln_2.weight)
        layer.ln_2.bias.data.copy_(src.ln_2.bias)
        w = src.mlp.c_fc.weight
        layer.mlp[0].weight.data.copy_(w.T if w.shape != layer.mlp[0].weight.shape else w)
        layer.mlp[0].bias.data.copy_(src.mlp.c_fc.bias)
        w = src.mlp.c_proj.weight
        layer.mlp[2].weight.data.copy_(w.T if w.shape != layer.mlp[2].weight.shape else w)
        layer.mlp[2].bias.data.copy_(src.mlp.c_proj.bias)
    del gpt2


def load_hybrid(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    hidden_size = kwargs.pop("hidden_size", 768)
    use_pretrained = kwargs.pop("pretrained", True)
    kwargs.pop("copy_mamba_weights", None)
    do_freeze_gpt2 = kwargs.pop("freeze_gpt2", False)

    model = HybridMambaTransformer(vocab_size=vocab_size, hidden_size=hidden_size)
    if use_pretrained:
        _copy_pretrained_weights(model)
    if do_freeze_gpt2:
        freeze_gpt2_components(model)
    if device is not None:
        model = model.to(device)
    return model, tokenizer


class SelectiveContextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        local_window: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(MAX_POSITION_EMBEDDINGS, hidden_size)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SelectiveContextBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                local_window=local_window,
                dropout=dropout,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], device: torch.device
    ):
        batch, seq_len = input_ids.shape
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)
        causal = causal.masked_fill(causal.bool(), float("-inf"))
        if attention_mask is not None:
            padding = torch.zeros(batch, 1, 1, seq_len, device=device)
            padding = padding.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
            return causal[None, None, :, :] + padding
        return causal[None, None, :, :].expand(batch, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        seq_len = input_ids.size(1)
        device = input_ids.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos_embed(position_ids))

        attn_mask = self._make_attention_mask(input_ids, attention_mask, device)
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"loss": loss, "logits": logits})()


class MambaSelectiveContextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        local_window: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(MAX_POSITION_EMBEDDINGS, hidden_size)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MambaSelectiveContextBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                local_window=local_window,
                dropout=dropout,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], device: torch.device
    ):
        batch, seq_len = input_ids.shape
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)
        causal = causal.masked_fill(causal.bool(), float("-inf"))
        if attention_mask is not None:
            padding = torch.zeros(batch, 1, 1, seq_len, device=device)
            padding = padding.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
            return causal[None, None, :, :] + padding
        return causal[None, None, :, :].expand(batch, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        seq_len = input_ids.size(1)
        device = input_ids.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos_embed(position_ids))

        attn_mask = self._make_attention_mask(input_ids, attention_mask, device)
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"loss": loss, "logits": logits})()


def load_selective(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    hidden_size = kwargs.pop("hidden_size", 768)
    num_layers = kwargs.pop("num_layers", 12)
    num_heads = kwargs.pop("num_heads", 12)
    local_window = kwargs.pop("local_window", 256)
    dropout = kwargs.pop("dropout", 0.0)
    use_pretrained = kwargs.pop("pretrained", True)
    kwargs.pop("copy_mamba_weights", None)
    kwargs.pop("freeze_gpt2", None)

    model = SelectiveContextTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        local_window=local_window,
        dropout=dropout,
    )
    if use_pretrained:
        _init_selective_from_gpt2(model)
    if device is not None:
        model = model.to(device)
    return model, tokenizer


def _init_selective_from_gpt2(model: SelectiveContextTransformer) -> None:
    from transformers import GPT2LMHeadModel
    model_id = os.environ.get("GPT2_BASE_MODEL", "gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained(model_id)
    model.embed.weight.data.copy_(gpt2.transformer.wte.weight)
    model.pos_embed.weight.data.copy_(gpt2.transformer.wpe.weight)
    model.ln_f.weight.data.copy_(gpt2.transformer.ln_f.weight)
    model.ln_f.bias.data.copy_(gpt2.transformer.ln_f.bias)
    model.lm_head.weight.data.copy_(gpt2.lm_head.weight)
    del gpt2


def load_mamba_selective(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    hidden_size = kwargs.pop("hidden_size", 768)
    num_layers = kwargs.pop("num_layers", 12)
    num_heads = kwargs.pop("num_heads", 12)
    local_window = kwargs.pop("local_window", 256)
    dropout = kwargs.pop("dropout", 0.0)
    use_pretrained = kwargs.pop("pretrained", True)
    kwargs.pop("copy_mamba_weights", None)
    kwargs.pop("freeze_gpt2", None)

    model = MambaSelectiveContextTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        local_window=local_window,
        dropout=dropout,
    )
    if use_pretrained:
        _init_mamba_selective_from_gpt2(model)
    if device is not None:
        model = model.to(device)
    return model, tokenizer


def _init_mamba_selective_from_gpt2(model: MambaSelectiveContextTransformer) -> None:
    from transformers import GPT2LMHeadModel
    model_id = os.environ.get("GPT2_BASE_MODEL", "gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained(model_id)
    model.embed.weight.data.copy_(gpt2.transformer.wte.weight)
    model.pos_embed.weight.data.copy_(gpt2.transformer.wpe.weight)
    model.ln_f.weight.data.copy_(gpt2.transformer.ln_f.weight)
    model.ln_f.bias.data.copy_(gpt2.transformer.ln_f.bias)
    model.lm_head.weight.data.copy_(gpt2.lm_head.weight)
    del gpt2


def freeze_gpt2_in_mamba_selective(model: Gpt2MambaSelectiveTransformer) -> None:
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.pos_embed.parameters():
        param.requires_grad = False
    for param in model.ln_f.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
    for layer in model.layers:
        for param in layer.ln_1.parameters():
            param.requires_grad = False
        for param in layer.c_attn.parameters():
            param.requires_grad = False
        for param in layer.c_proj.parameters():
            param.requires_grad = False
        for param in layer.ln_2.parameters():
            param.requires_grad = False
        for param in layer.mlp.parameters():
            param.requires_grad = False


class Gpt2ResidualAdaptor(nn.Module):
    def __init__(self, base_model_id: str = "gpt2", adaptor_hidden: int = 256):
        super().__init__()
        from transformers import GPT2LMHeadModel

        self.base = GPT2LMHeadModel.from_pretrained(base_model_id)
        for p in self.base.parameters():
            p.requires_grad = False

        hidden_size = self.base.config.n_embd
        vocab_size = self.base.config.vocab_size
        self.vocab_size = vocab_size

        self.adaptor = nn.Sequential(
            nn.Linear(hidden_size, adaptor_hidden),
            nn.GELU(),
            nn.Linear(adaptor_hidden, vocab_size),
        )
        # Scalar gate; start near 0 so we initially match GPT-2.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        base_out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            labels=None,
        )
        logits_base = base_out.logits
        hidden = base_out.hidden_states[-1]

        delta_logits = self.adaptor(hidden)
        gate = torch.sigmoid(self.alpha)
        logits = logits_base + gate * delta_logits

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"loss": loss, "logits": logits})()


def load_gpt2_mamba_selective(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    hidden_size = kwargs.pop("hidden_size", 768)
    dropout = kwargs.pop("dropout", 0.0)
    use_checkpointing = kwargs.pop("use_checkpointing", False)
    use_pretrained = kwargs.pop("pretrained", True)
    do_freeze_gpt2 = kwargs.pop("freeze_gpt2", False)
    kwargs.pop("copy_mamba_weights", None)

    model = Gpt2MambaSelectiveTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        dropout=dropout,
        use_checkpointing=use_checkpointing,
    )
    if use_pretrained:
        _copy_gpt2_to_mamba_selective(model)
    if do_freeze_gpt2:
        freeze_gpt2_in_mamba_selective(model)
    if device is not None:
        model = model.to(device)
    return model, tokenizer


def load_hybrid2(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    tokenizer = get_tokenizer("gpt2")
    base_id = os.environ.get("GPT2_BASE_MODEL", "gpt2")
    adaptor_hidden = kwargs.pop("adaptor_hidden", 256)

    model = Gpt2ResidualAdaptor(base_model_id=base_id, adaptor_hidden=adaptor_hidden)
    if device is not None:
        model = model.to(device)
    return model, tokenizer
