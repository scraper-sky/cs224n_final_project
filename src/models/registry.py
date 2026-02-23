#we run models locally without making any API calls (which have limited inference and usage)
#we use the same tokenizer (GPT-2) across all models as a baseline; this ensures that prompts and metrics are comparable
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# these are imports that we need to local the models and tokenizers; do not require transformers until a model is requested
_models: dict[str, Any] = {}
_tokenizer: Optional["PreTrainedTokenizerBase"] = None


def get_tokenizer(name: str = "gpt2",**kwargs: Any,) -> "PreTrainedTokenizerBase":
    #here we load and cache the tokenizer, which is shared across all models 
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    if name != "gpt2":
        # for now we use GPT-2 tokenizer for all; later Mamba/Hybrid can share it
        name = "gpt2"
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    return _tokenizer


def get_model(name: str = "gpt2",*,tokenizer_name: str = "gpt2",
    device: Optional[str] = None,
    **model_kwargs: Any,
) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    # here we load the model and tokenizer given its string name
    # the string is either "gpt2", "mamba", or "hybrid"
    # we use the same tokenizer name (defaulted to gpt2 for all for comparability)
    # the device is either "cuda" or "cpu" and if None we do not move the model
    # based on this, we return the moddel and the tokenizer
    if name == "gpt2":
        return _get_gpt2(device=device, **model_kwargs)
    if name == "mamba":
        raise NotImplementedError("Mamba loader not yet implemented")
    if name == "hybrid":
        raise NotImplementedError("Hybrid (HMT) loader not yet implemented")
    raise ValueError(f"Invalid model name: {name}")


def _get_gpt2(
    device: Optional[str] = None,
    **kwargs: Any,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    # here we load the hugging face gpt2 model (same architecture as Karpathy's minGPT)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = kwargs.pop("pretrained", None) or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device is not None:
        model = model.to(device)
    return model, tokenizer


def list_models() -> list[str]:
    #here are the list of models that we will support for our experiment
    return ["gpt2", "mamba", "hybrid"]
