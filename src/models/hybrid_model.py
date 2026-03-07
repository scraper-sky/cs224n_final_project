from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .tokenizer import get_tokenizer

MAX_POSITION_EMBEDDINGS = 1024


class MambaFormerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention,
        mamba,
        mamba_first: bool,
    ):
        super().__init__()
        self.mamba_first = mamba_first
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.attn = attention
        self.mamba = mamba
        self.gate_a = nn.Parameter(torch.ones(1))
        self.gate_b = nn.Parameter(torch.ones(1))

    @classmethod
    def with_mamba_gated(cls, hidden_size: int, attention, mamba, mamba_first: bool) -> "MambaFormerBlock":
        b = cls(hidden_size, attention, mamba, mamba_first)
        if mamba_first:
            nn.init.zeros_(b.gate_a)
        else:
            nn.init.zeros_(b.gate_b)
        return b

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mamba_first:
            x = x + self.gate_a * self.mamba(self.ln1(x))
            attn_out = self.attn(self.ln2(x), attention_mask=attn_mask, use_cache=False)[0]
            x = x + self.gate_b * attn_out
        else:
            attn_out = self.attn(self.ln1(x), attention_mask=attn_mask, use_cache=False)[0]
            x = x + self.gate_a * attn_out
            x = x + self.gate_b * self.mamba(self.ln2(x))
        return x


class HybridMambaTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768, n_layer: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(MAX_POSITION_EMBEDDINGS, hidden_size)

        from transformers import MambaConfig, GPT2Config
        from transformers.models.mamba.modeling_mamba import MambaBlock
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        mamba_config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
        if mamba_config.hidden_size != hidden_size:
            mamba_config.hidden_size = hidden_size
            mamba_config.intermediate_size = hidden_size * 4

        gpt2_config = GPT2Config.from_pretrained("gpt2")
        gpt2_config.n_embd = hidden_size
        gpt2_config.n_layer = 1
        gpt2_config._attn_implementation = "eager"

        self.initial_mamba = MambaBlock(mamba_config, layer_idx=0)
        self.initial_mamba_gate = nn.Parameter(torch.zeros(1))

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            attn = GPT2Attention(gpt2_config, layer_idx=i)
            mamba = MambaBlock(mamba_config, layer_idx=i + 1)
            mamba_first = i < n_layer // 2
            self.layers.append(
                MambaFormerBlock.with_mamba_gated(
                    hidden_size=hidden_size,
                    attention=attn,
                    mamba=mamba,
                    mamba_first=mamba_first,
                )
            )

        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], device: torch.device
    ) -> Optional[torch.Tensor]:
        batch, seq_len = input_ids.shape
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1
        )
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
        x = x + self.initial_mamba_gate * self.initial_mamba(x)

        attn_mask = self._make_attention_mask(input_ids, attention_mask, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

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


def _copy_pretrained_weights(model: HybridMambaTransformer) -> None:
    from transformers import GPT2LMHeadModel

    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2.eval()

    model.embed.weight.data.copy_(gpt2.transformer.wte.weight.data)
    model.pos_embed.weight.data.copy_(gpt2.transformer.wpe.weight.data)
    for i, layer in enumerate(model.layers):
        gpt2_block = gpt2.transformer.h[i]
        layer.attn.load_state_dict(gpt2_block.attn.state_dict(), strict=True)
    model.ln_f.weight.data.copy_(gpt2.transformer.ln_f.weight.data)
    model.ln_f.bias.data.copy_(gpt2.transformer.ln_f.bias.data)
    model.lm_head.weight.data.copy_(gpt2.lm_head.weight.data)

    del gpt2


def freeze_gpt2_components(model: HybridMambaTransformer) -> None:
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.pos_embed.parameters():
        param.requires_grad = False
    for param in model.initial_mamba.parameters():
        param.requires_grad = False
    for layer in model.layers:
        for param in layer.attn.parameters():
            param.requires_grad = False
    for param in model.ln_f.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False


def load_hybrid(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    hidden_size = kwargs.pop("hidden_size", 768)
    n_layer = kwargs.pop("n_layer", 12)
    use_pretrained = kwargs.pop("pretrained", True)
    kwargs.pop("copy_mamba_weights", None)
    do_freeze_gpt2 = kwargs.pop("freeze_gpt2", False)

    model = HybridMambaTransformer(vocab_size=vocab_size, hidden_size=hidden_size, n_layer=n_layer)
    if use_pretrained:
        _copy_pretrained_weights(model)
    if do_freeze_gpt2:
        freeze_gpt2_components(model)
    if device is not None:
        model = model.to(device)
    return model, tokenizer
