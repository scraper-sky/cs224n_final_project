# Hybrid Mamba + Transformer (HMT): 12 layers [Mamba,Mamba,Mamba,Attn] x3, same tokenizer as other models.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .tokenizer import get_tokenizer

# Layer pattern: 0,1,2 Mamba; 3 Attn; 4,5,6 Mamba; 7 Attn; 8,9,10 Mamba; 11 Attn
ATTN_LAYER_INDICES = (3, 7, 11)


class HybridMambaTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)

        # Load configs from pretrained for correct defaults
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

        self.layers = nn.ModuleList()
        mamba_idx = 0
        for i in range(12):
            if i in ATTN_LAYER_INDICES:
                self.layers.append(GPT2Block(gpt2_config, layer_idx=i))
            else:
                self.layers.append(MambaBlock(mamba_config, layer_idx=mamba_idx))
                mamba_idx += 1

        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_attention_mask(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], device: torch.device):
        batch, seq_len = input_ids.shape
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)
        causal = causal.masked_fill(causal.bool(), float("-inf"))
        if attention_mask is not None:
            padding = (1.0 - attention_mask[:, None, None, :]) * float("-inf")
            return causal[None, None, :, :] + padding
        return causal[None, None, :, :].expand(batch, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        x = self.embed(input_ids)
        seq_len = x.size(1)
        device = x.device
        attn_mask_4d = self._make_attention_mask(input_ids, attention_mask, device)

        for i, layer in enumerate(self.layers):
            if i in ATTN_LAYER_INDICES:
                out, _ = layer(x, attention_mask=attn_mask_4d, use_cache=False)
                x = out
            else:
                # MambaBlock expects attention_mask (batch, seq_len): 1 = keep, 0 = pad
                mamba_attn = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.float, device=device)
                x = x + layer(x, attention_mask=mamba_attn)

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


def load_hybrid(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    hidden_size = kwargs.pop("hidden_size", 768)
    model = HybridMambaTransformer(vocab_size=vocab_size, hidden_size=hidden_size)
    if device is not None:
        model = model.to(device)
    return model, tokenizer
