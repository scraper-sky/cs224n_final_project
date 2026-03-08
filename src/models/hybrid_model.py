from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

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

        attn_combined = w_local * attn_local + w_global * attn_global
        attn_combined = self.attn_dropout(attn_combined)

        out = torch.matmul(attn_combined, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


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
        from transformers import MambaConfig
        from transformers.models.mamba.modeling_mamba import MambaBlock

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.local_window = local_window
        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        from transformers import MambaModel
        mamba_pretrained = MambaModel.from_pretrained("state-spaces/mamba-130m-hf")
        if layer_idx < len(mamba_pretrained.layers):
            self.mamba_selector = mamba_pretrained.layers[layer_idx]
        else:
            self.mamba_selector = mamba_pretrained.layers[0]

        for param in self.mamba_selector.parameters():
            param.requires_grad = False

        self.mamba_input_ln = nn.LayerNorm(hidden_size)
        self.mamba_output_ln = nn.LayerNorm(hidden_size)
        self.mamba_scale = nn.Parameter(torch.tensor(0.1))
        self.selectivity_proj = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_proj.bias)

        nn.init.normal_(self.selectivity_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.selectivity_proj.bias, 0.0)

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

        with torch.no_grad():
            x_for_mamba = self.mamba_input_ln(x)
            mamba_raw = self.mamba_selector(x_for_mamba)

        mamba_normed = self.mamba_output_ln(mamba_raw)
        mamba_scaled = self.mamba_scale * mamba_normed
        selectivity_logits = self.selectivity_proj(mamba_scaled)
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
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
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return type("Output", (), {"loss": loss, "logits": logits})()


def _copy_pretrained_weights(model: HybridMambaTransformer) -> None:
    from transformers import GPT2LMHeadModel

    gpt2_src = GPT2LMHeadModel.from_pretrained("gpt2")
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
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
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
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
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
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
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
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    model.embed.weight.data.copy_(gpt2.transformer.wte.weight)
    model.pos_embed.weight.data.copy_(gpt2.transformer.wpe.weight)
    model.ln_f.weight.data.copy_(gpt2.transformer.ln_f.weight)
    model.ln_f.bias.data.copy_(gpt2.transformer.ln_f.bias)
    model.lm_head.weight.data.copy_(gpt2.lm_head.weight)
    del gpt2
