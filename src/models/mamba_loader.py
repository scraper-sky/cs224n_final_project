# Load Hugging Face Mamba (causal LM). Uses shared GPT-2 tokenizer.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .tokenizer import get_tokenizer


def load_mamba(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    from transformers import MambaForCausalLM
    model_id = kwargs.pop("pretrained", None) or "state-spaces/mamba-130m-hf"
    tokenizer = get_tokenizer("gpt2")
    model = MambaForCausalLM.from_pretrained(model_id, **kwargs)
    if device is not None:
        model = model.to(device)
    return model, tokenizer
