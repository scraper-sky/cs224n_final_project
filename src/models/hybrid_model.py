# Hybrid (HMT): alternating Mamba + Transformer layers. Same tokenizer as other models.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from .tokenizer import get_tokenizer


def load_hybrid(device: Optional[str] = None, **kwargs: Any) -> Tuple[Any, "PreTrainedTokenizerBase"]:
    # TODO: implement HMT (12 layers: Mamba, Mamba, Mamba, Transformer, ...), RMSNorm, residuals, KV-cache at attn only.
    raise NotImplementedError("Hybrid (HMT) not yet implemented")
