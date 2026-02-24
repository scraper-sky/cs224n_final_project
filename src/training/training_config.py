# this is the training config that uses hyperparameters and paths
# this can be overridden via environment or pass dict.

import os
from typing import Any, Optional


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_config(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data_dir = os.environ.get("DATA_DIR", os.path.join(_project_root(), "data"))
    config = {
        "data_dir": data_dir,
        "math_preprocessed_jsonl": os.environ.get("MATH_PREPROCESSED_JSONL", os.path.join(data_dir, "olympiad_preprocessed.jsonl")),
        "model_name": os.environ.get("TRAIN_MODEL", "gpt2"),
        "device": os.environ.get("DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu"),
        "batch_size": int(os.environ.get("BATCH_SIZE", "2")),
        "max_length": int(os.environ.get("MAX_LENGTH", "512")),
        "lr": float(os.environ.get("LR", "5e-5")),
        "max_steps": int(os.environ.get("MAX_STEPS", "1000")),
        "checkpoint_dir": os.environ.get("CHECKPOINT_DIR", os.path.join(_project_root(), "checkpoints")),
        "save_every": int(os.environ.get("SAVE_EVERY", "500")),
        "seed": int(os.environ.get("SEED", "42")),
        "train_on_solution_only": os.environ.get("TRAIN_ON_SOLUTION_ONLY", "0").lower() in ("1", "true", "yes"),
    }
    if overrides:
        config.update(overrides)
    return config
