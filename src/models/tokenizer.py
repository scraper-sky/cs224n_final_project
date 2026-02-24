# the tokenizer (GPT-2) is shared for all models and is cached so training and eval use the same one 
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_tokenizer: Optional["PreTrainedTokenizerBase"] = None


def get_tokenizer(name: str = "gpt2", **kwargs: Any) -> "PreTrainedTokenizerBase":
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    if name != "gpt2":
        name = "gpt2"
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    return _tokenizer
