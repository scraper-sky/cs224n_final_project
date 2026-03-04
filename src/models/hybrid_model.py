# Hybrid Mamba + Transformer (HMT): 12 layers [Mamba,Mamba,Mamba,Attn] x3, same tokenizer as other models.
#
# Pretrained-weight strategy (load_hybrid, pretrained=True):
#   - Token embedding, positional embedding, final LayerNorm, LM head  <- GPT-2 (vocab_size 50257 matches our tokenizer)
#   - 3 Attention blocks sampled evenly from the 12 layers of GPT-2
#   - 9 Mamba blocks: randomly initialized by default (copy_mamba_weights=False).

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


MAX_POSITION_EMBEDDINGS = 1024  # Match GPT-2's hard positional-embedding limit

class HybridMambaTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        # ensure GPT-2 attention blocks receive position-aware hidden states as they were pretrained
        self.pos_embed = nn.Embedding(MAX_POSITION_EMBEDDINGS, hidden_size)

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
        gpt2_config._attn_implementation = "eager"

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
        # Causal mask: upper-triangle positions get -inf so attention cannot look ahead.
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1)
        causal = causal.masked_fill(causal.bool(), float("-inf"))
        if attention_mask is not None:
            # Build additive padding mask (0 for valid tokens, -inf for padding)
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

        # Token + positional embeddings (matches GPT-2 pretraining convention).
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(position_ids)

        attn_mask_4d = self._make_attention_mask(input_ids, attention_mask, device)

        for i, layer in enumerate(self.layers):
            if i in ATTN_LAYER_INDICES:
                x = layer(x, attention_mask=attn_mask_4d, use_cache=False)[0]
            else:
                mamba_attn = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.float, device=device)
                x = layer(x, attention_mask=mamba_attn)

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
    """Copy pretrained weights from GPT-2 (and optionally Mamba-130m) into the hybrid model.

    Always copies (from GPT-2):
      - embed, pos_embed       <- wte + wpe  (vocab_size=50257 matches the shared GPT-2 tokenizer)
      - ln_f, lm_head          <- final LayerNorm and output projection
      - 3 Attention blocks     <- evenly-spaced layers from the 12 GPT-2 layers

    Optionally copies (copy_mamba_weights=True):
      - 9 Mamba blocks         <- evenly-spaced layers from state-spaces/mamba-130m-hf (24 layers)

    WARNING: copy_mamba_weights=True causes instability.  Mamba-130m was trained in its own
    embedding space (EleutherAI gpt-neox-20b tokenizer).  Feeding GPT-2 embeddings through
    those copied SSM dynamics produces out-of-distribution recurrent states that diverge,
    yielding perplexity > 10^6.  Leave False (default) to randomly initialise the Mamba
    blocks and fine-tune them in-place against the stable GPT-2 scaffold.
    """
    from transformers import GPT2LMHeadModel

    if copy_mamba_weights:
        from transformers import MambaForCausalLM
        print("  loading pretrained weights: state-spaces/mamba-130m-hf + gpt2 ...")
        mamba_src = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        mamba_src.eval()
    else:
        print("  loading pretrained weights: gpt2 only (Mamba blocks randomly initialised) ...")
        mamba_src = None

    gpt2_src = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_src.eval()

    # From GPT-2: embedding, positional embedding, final LayerNorm, LM head
    # GPT-2 matches our tokenizer
    model.embed.weight.data.copy_(gpt2_src.transformer.wte.weight.data)
    model.pos_embed.weight.data.copy_(gpt2_src.transformer.wpe.weight.data)
    model.ln_f.weight.data.copy_(gpt2_src.transformer.ln_f.weight.data)
    model.ln_f.bias.data.copy_(gpt2_src.transformer.ln_f.bias.data)
    model.lm_head.weight.data.copy_(gpt2_src.lm_head.weight.data)

    # Sample evenly from the 12 pretrained GPT-2 layers
    attn_hybrid_indices = sorted(ATTN_LAYER_INDICES)
    n_attn = len(attn_hybrid_indices)
    n_gpt2_src = len(gpt2_src.transformer.h)
    src_gpt2_picks = [
        round(k * (n_gpt2_src - 1) / (n_attn - 1)) for k in range(n_attn)
    ]
    for hybrid_i, src_i in zip(attn_hybrid_indices, src_gpt2_picks):
        model.layers[hybrid_i].load_state_dict(
            gpt2_src.transformer.h[src_i].state_dict()
        )

    # Mamba blocks: optionally copy from Mamba-130m
    if copy_mamba_weights and mamba_src is not None:
        mamba_hybrid_indices = [i for i in range(12) if i not in ATTN_LAYER_INDICES]
        n_mamba = len(mamba_hybrid_indices)
        n_mamba_src = len(mamba_src.backbone.layers)
        src_mamba_picks = [
            round(k * (n_mamba_src - 1) / (n_mamba - 1)) for k in range(n_mamba)
        ]
        for hybrid_i, src_i in zip(mamba_hybrid_indices, src_mamba_picks):
            model.layers[hybrid_i].load_state_dict(
                mamba_src.backbone.layers[src_i].state_dict()
            )
        del mamba_src

    del gpt2_src
    print("  pretrained weights loaded successfully.")


def freeze_gpt2_components(model: HybridMambaTransformer) -> None:
    """Freeze all GPT-2-sourced parameters; only Mamba blocks remain trainable.

    This is the recommended first fine-tuning recipe: the GPT-2 embedding/attention
    scaffold is already well-trained, so freezing it lets the randomly-initialised Mamba
    blocks converge quickly (fewer effective parameters, stable gradient signal from the
    frozen attention layers).  Unfreeze everything for a second, lower-lr fine-tuning pass
    once the Mamba blocks have stabilised.

    Frozen components: embed, pos_embed, ln_f, lm_head, all attention (ATTN_LAYER_INDICES).
    Trainable components: all Mamba blocks.
    """
    # Freeze GPT-2 scaffold
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

    # Ensure Mamba blocks are trainable
    for i, layer in enumerate(model.layers):
        if i not in ATTN_LAYER_INDICES:
            for param in layer.parameters():
                param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  freeze_gpt2_components: {n_trainable:,} / {n_total:,} parameters trainable "
          f"({100 * n_trainable / n_total:.1f}%)")


def load_hybrid(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    """Load the HybridMambaTransformer with optional pretrained weight initialisation.

    Keyword args (all popped before forwarding to the model constructor):
      pretrained (bool, default True):
          If True, copy GPT-2 weights into embeddings, positional embeddings, final LayerNorm,
          LM head, and attention blocks.  Mamba blocks are randomly initialised unless
          copy_mamba_weights=True is also passed.
      copy_mamba_weights (bool, default False):
          Copy Mamba-130m weights into the Mamba blocks.  Disabled by default because those
          weights were trained in a different embedding space and cause numerical instability
          when combined with GPT-2 embeddings (see module docstring for full explanation).
      freeze_gpt2 (bool, default False):
          After weight loading, freeze all GPT-2-sourced parameters so that only the Mamba
          blocks are updated during fine-tuning.  Recommended for the first training phase.
    """
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
