from __future__ import annotations

from typing import Any, Optional, Tuple

from .tokenizer import get_tokenizer
from .gpt2_loader import load_gpt2
from .mamba_loader import load_mamba
from .hybrid_model import load_hybrid, load_selective, load_mamba_selective, load_gpt2_mamba_selective
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
    if name == "mamba_selective":
        return load_mamba_selective(device=device, **model_kwargs)
    if name == "gpt2_mamba_selective":
        return load_gpt2_mamba_selective(device=device, **model_kwargs)
    raise ValueError(f"Invalid model name: {name}")


def list_models() -> list[str]:
    return ["gpt2", "mamba", "hybrid", "hybrid_llm", "selective", "mamba_selective", "gpt2_mamba_selective"]
