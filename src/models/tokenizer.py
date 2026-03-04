# Tokenizer cache keyed by model name so different models can use their own native tokenizer.
# Previously this module forced every call to return the GPT-2 tokenizer, which caused a
# vocabulary distribution mismatch when Mamba (trained with its own 50280-token vocab) was
# evaluated using GPT-2-tokenized text.  Each model loader now requests its native tokenizer.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Per-name cache: keeps one loaded tokenizer per model identifier so repeated calls are cheap.
_tokenizer_cache: Dict[str, "PreTrainedTokenizerBase"] = {}


def get_tokenizer(name: str = "gpt2", **kwargs: Any) -> "PreTrainedTokenizerBase":
    """Return (and cache) the tokenizer for *name*.

    Supported shortcuts:
      "gpt2"  -> openai-community/gpt2  (50257 tokens, used by GPT-2 and HMT)
      "mamba" -> state-spaces/mamba-130m-hf native tokenizer (50280 tokens)
      Any other HuggingFace model ID is passed through directly.
    """
    # Normalise common aliases
    _aliases = {"mamba": "state-spaces/mamba-130m-hf"}
    resolved = _aliases.get(name, name)

    if resolved in _tokenizer_cache:
        return _tokenizer_cache[resolved]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolved, **kwargs)
    _tokenizer_cache[resolved] = tok
    return tok
