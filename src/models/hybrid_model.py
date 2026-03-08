from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .tokenizer import get_tokenizer

MAX_POSITION_EMBEDDINGS = 1024


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
        )
        self.mamba_gate = nn.Parameter(torch.zeros(1))
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
        gate = torch.clamp(self.mamba_gate, 0.0, 0.03)
        h = h_attn + gate * h_mamba
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
