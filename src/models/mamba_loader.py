# Load Hugging Face Mamba (causal LM).
#
# Tokenizer note: Mamba-130m was pretrained with the EleutherAI/gpt-neox-20b tokenizer
# (vocab size 50277, padded to 50280 in the checkpoint).  Previously this loader used the
# GPT-2 tokenizer (50257 tokens), creating a complete vocabulary distribution mismatch:
# token ID N in GPT-2's BPE maps to an entirely different word-piece than token ID N in
# Mamba's tokenizer, so the model's embedding matrix produced nonsensical representations
# for every token, directly explaining the catastrophic perplexity of ~6000.
#
# Fix: load the native tokenizer from the Mamba checkpoint.  Perplexity is now computed
# over the model's own tokenization of the same text, giving a fair comparison.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .tokenizer import get_tokenizer


def load_mamba(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    from transformers import MambaForCausalLM
    model_id = kwargs.pop("pretrained", None) or "state-spaces/mamba-130m-hf"
    # Use Mamba's native tokenizer (EleutherAI gpt-neox-20b, vocab 50280) so that token
    # IDs match the embedding matrix the model was actually trained with.
    tokenizer = get_tokenizer("mamba")
    model = MambaForCausalLM.from_pretrained(model_id, **kwargs)
    # the checkpoint omits lm_head.weight because it is tied to backbone.embeddings.weight
    # in the original implementation. the HF config has tie_word_embeddings=False so
    # tie_weights() is a no-op; we assign the tensor directly to enforce the tie.
    model.lm_head.weight = model.backbone.embeddings.weight
    if device is not None:
        model = model.to(device)
    return model, tokenizer
