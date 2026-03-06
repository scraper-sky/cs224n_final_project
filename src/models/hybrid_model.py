from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .tokenizer import get_tokenizer

ATTN_LAYER_INDICES = (0, 1, 2, 9, 10, 11)
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

        self.layers = nn.ModuleList()
        self.attn_to_mamba = nn.Linear(hidden_size, hidden_size)
        self.mamba_to_attn = nn.Linear(hidden_size, hidden_size)
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

        first_attn, mid_mamba, last_attn = (0, 1, 2), (3, 4, 5, 6, 7, 8), (9, 10, 11)
        for i, layer in enumerate(self.layers):
            if i in first_attn:
                x = layer(x, attention_mask=attn_mask_4d, use_cache=False)[0]
            elif i in mid_mamba:
                if i == mid_mamba[0]:
                    x = self.attn_to_mamba(x)
                mamba_attn = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.float, device=device)
                x = layer(x, attention_mask=mamba_attn)
            else:
                if i == last_attn[0]:
                    x = self.mamba_to_attn(x)
                x = layer(x, attention_mask=attn_mask_4d, use_cache=False)[0]

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


def _copy_pretrained_weights(model: HybridMambaTransformer, copy_mamba_weights: bool = False) -> None:
    from transformers import GPT2LMHeadModel

    if copy_mamba_weights:
        from transformers import MambaForCausalLM
        mamba_src = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        mamba_src.eval()
    else:
        mamba_src = None

    gpt2_src = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_src.eval()

    model.embed.weight.data.copy_(gpt2_src.transformer.wte.weight.data)
    model.pos_embed.weight.data.copy_(gpt2_src.transformer.wpe.weight.data)
    model.ln_f.weight.data.copy_(gpt2_src.transformer.ln_f.weight.data)
    model.ln_f.bias.data.copy_(gpt2_src.transformer.ln_f.bias.data)
    model.lm_head.weight.data.copy_(gpt2_src.lm_head.weight.data)

    attn_hybrid_indices = sorted(ATTN_LAYER_INDICES)
    n_attn = len(attn_hybrid_indices)
    n_gpt2_src = len(gpt2_src.transformer.h)
    src_gpt2_picks = [round(k * (n_gpt2_src - 1) / (n_attn - 1)) for k in range(n_attn)]
    for hybrid_i, src_i in zip(attn_hybrid_indices, src_gpt2_picks):
        model.layers[hybrid_i].load_state_dict(gpt2_src.transformer.h[src_i].state_dict())

    nn.init.eye_(model.attn_to_mamba.weight)
    if model.attn_to_mamba.bias is not None:
        nn.init.zeros_(model.attn_to_mamba.bias)
    nn.init.eye_(model.mamba_to_attn.weight)
    if model.mamba_to_attn.bias is not None:
        nn.init.zeros_(model.mamba_to_attn.bias)

    if copy_mamba_weights and mamba_src is not None:
        mamba_hybrid_indices = [i for i in range(12) if i not in ATTN_LAYER_INDICES]
        n_mamba = len(mamba_hybrid_indices)
        n_mamba_src = len(mamba_src.backbone.layers)
        src_mamba_picks = [round(k * (n_mamba_src - 1) / (n_mamba - 1)) for k in range(n_mamba)]
        for hybrid_i, src_i in zip(mamba_hybrid_indices, src_mamba_picks):
            model.layers[hybrid_i].load_state_dict(mamba_src.backbone.layers[src_i].state_dict())
        del mamba_src

    del gpt2_src


def freeze_gpt2_components(model: HybridMambaTransformer) -> None:
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.pos_embed.parameters():
        param.requires_grad = False
    for param in model.ln_f.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
    for i in ATTN_LAYER_INDICES:
        for param in model.layers[i].parameters():
            param.requires_grad = False
    for i, layer in enumerate(model.layers):
        if i not in ATTN_LAYER_INDICES:
            for param in layer.parameters():
                param.requires_grad = True


def load_hybrid(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    hidden_size = kwargs.pop("hidden_size", 768)
    use_pretrained = kwargs.pop("pretrained", True)
    copy_mamba_weights = kwargs.pop("copy_mamba_weights", False)
    do_freeze_gpt2 = kwargs.pop("freeze_gpt2", False)

    model = HybridMambaTransformer(vocab_size=vocab_size, hidden_size=hidden_size)
    if use_pretrained:
        _copy_pretrained_weights(model, copy_mamba_weights=copy_mamba_weights)
    if do_freeze_gpt2:
        freeze_gpt2_components(model)
    if device is not None:
        model = model.to(device)
    return model, tokenizer
