# We run models locally without making any API calls (which have limited inference and usage)
# We use the same tokenizer (GPT-2) across all models as a baseline; this ensures that prompts and metrics are comparable
from __future__ import annotations

from typing import Any, Optional, Tuple

from .tokenizer import get_tokenizer
from .gpt2_loader import load_gpt2
from .mamba_loader import load_mamba
from .hybrid_model import load_hybrid, load_selective
from .hybrid_llm_loader import load_hybrid_llm


def get_model(
    name: str = "gpt2",
    *,
    tokenizer_name: str = "gpt2",
    device: Optional[str] = None,
    **model_kwargs: Any,
) -> Tuple[Any, Any]:
    if name == "gpt2":
        return load_gpt2(device=device, **model_kwargs)
    if name == "mamba":
        return load_mamba(device=device, **model_kwargs)
    if name == "hybrid":
        return load_hybrid(device=device, **model_kwargs)
    if name == "selective":
        return load_selective(device=device, **model_kwargs)
    raise ValueError(f"Invalid model name: {name}")


def list_models() -> list[str]:
    # this lists the models that are available to use in the get_model function
    return ["gpt2", "mamba", "hybrid", "hybrid_llm", "selective"]
