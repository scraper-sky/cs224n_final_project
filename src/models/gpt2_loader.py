# Load Hugging Face GPT-2 (causal LM). Same architecture as Karpathy minGPT.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_gpt2(device: Optional[str] = None, **kwargs: Any) -> Tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = kwargs.pop("pretrained", None) or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device is not None:
        model = model.to(device)
    return model, tokenizer
