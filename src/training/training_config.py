# Training configuration.  All values can be overridden via environment variables or by
# passing a dict to get_config().
#
# HMT recommended Phase-1 recipe (freeze GPT-2, train only Mamba blocks):
#   TRAIN_MODEL=hybrid FREEZE_GPT2=1 LR=1e-4 WARMUP_STEPS=100 MAX_GRAD_NORM=1.0 \
#   BATCH_SIZE=2 MAX_STEPS=500 SAVE_EVERY=250 python -m src.training.run_training
#
# HMT Phase-2 (full fine-tune after Mamba blocks have stabilised):
#   TRAIN_MODEL=hybrid FREEZE_GPT2=0 LR=2e-5 WARMUP_STEPS=50 MAX_GRAD_NORM=1.0 \
#   MAX_STEPS=500 python -m src.training.run_training

import os
from typing import Any, Optional


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_config(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root(), "data")
    config = {
        "data_dir": data_dir,
        "math_preprocessed_jsonl": os.environ.get("MATH_PREPROCESSED_JSONL", os.path.join(data_dir, "olympiad_preprocessed.jsonl")),
        "model_name": os.environ.get("TRAIN_MODEL", "gpt2"),
        "device": os.environ.get("DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu"),
        "batch_size": int(os.environ.get("BATCH_SIZE", "2")),
        "max_length": int(os.environ.get("MAX_LENGTH", "512")),
        "lr": float(os.environ.get("LR", "5e-5")),
        "max_steps": int(os.environ.get("MAX_STEPS", "250")),
        "checkpoint_dir": os.environ.get("CHECKPOINT_DIR") or os.path.join(_project_root(), "checkpoints"),
        "save_every": int(os.environ.get("SAVE_EVERY", "500")),
        "seed": int(os.environ.get("SEED", "42")),
        "train_on_solution_only": os.environ.get("TRAIN_ON_SOLUTION_ONLY", "0").lower() in ("1", "true", "yes"),
        "append_answer": os.environ.get("APPEND_ANSWER", "1").lower() in ("1", "true", "yes"),
        "direct_qa": os.environ.get("TRAIN_DIRECT_QA", "0").lower() in ("1", "true", "yes"),

        # Additional parameters for stability and optimization
        # freeze_gpt2: in HMT, freeze embeddings, positional embeddings, ln_f, lm_head, attn blocks
        # so only Mamba blocks are updated.
        "freeze_gpt2": os.environ.get("FREEZE_GPT2", "0").lower() in ("1", "true", "yes"),
        "math_focused": os.environ.get("TRAIN_MATH_FOCUSED", "0").lower() in ("1", "true", "yes"),
        "balanced_training": os.environ.get("BALANCED_TRAINING", "0").lower() in ("1", "true", "yes"),
        "literature_jsonl": os.environ.get("GUTENBERG_CHUNKS_JSONL", os.path.join(data_dir, "gutenberg_7000_1192.jsonl")),
        "literature_ratio": float(os.environ.get("LITERATURE_RATIO", "0.5")),
        "gate_reg": float(os.environ.get("GATE_REG", "0.01")),
        # warmup_steps: number of steps over which LR linearly ramps from 0 to cfg["lr"].
        # (set to 10% of max_steps to prevent large gradient updates in early steps)
        "warmup_steps": int(os.environ.get("WARMUP_STEPS", "100")),
        # max_grad_norm: gradient clipping threshold. Set to 0 to disable.
        "max_grad_norm": float(os.environ.get("MAX_GRAD_NORM", "1.0")),
    }
    if overrides:
        config.update(overrides)
    if config.get("direct_qa"):
        config["math_focused"] = True
    if config.get("math_focused"):
        config["lr"] = 2e-5
        if os.environ.get("FREEZE_GPT2", "0").lower() not in ("1", "true", "yes"):
            config["freeze_gpt2"] = False
    return config
