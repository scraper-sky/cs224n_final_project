from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_tokenizer_cache: Dict[str, "PreTrainedTokenizerBase"] = {}


def get_tokenizer(name: str = "gpt2", **kwargs: Any) -> "PreTrainedTokenizerBase":
    _aliases = {"mamba": "state-spaces/mamba-130m-hf"}
    resolved = _aliases.get(name, name)

    if resolved in _tokenizer_cache:
        return _tokenizer_cache[resolved]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolved, **kwargs)
    _tokenizer_cache[resolved] = tok
    return tok
