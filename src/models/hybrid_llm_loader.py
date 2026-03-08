from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def load_hybrid_llm(
    device: Optional[str] = None, **kwargs: Any
) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = kwargs.pop("pretrained", None) or "vukrosic/hybrid-llm"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device is not None:
        model = model.to(device)
    return model, tokenizer
