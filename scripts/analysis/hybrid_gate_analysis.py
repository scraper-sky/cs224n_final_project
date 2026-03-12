"""this tracks hybrid gate values over training"""
from __future__ import annotations

import argparse
import math
import re

import torch

from scripts.analysis._common import ensure_dir, iter_checkpoint_files, write_csv


STEP_RE = re.compile(r"step(\d+)")
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect hybrid gate parameters across checkpoints.")
    parser.add_argument("--checkpoint-glob", default="checkpoints/**/*.pt")
    parser.add_argument("--output-dir", default="scripts/analysis/outputs/hybrid_gates")
    return parser.parse_args()


def extract_gate_rows(checkpoint_path: str) -> list[dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    step = checkpoint.get("step")
    if step is None:
        match = STEP_RE.search(checkpoint_path)
        step = int(match.group(1)) if match else -1

    rows = []
    for name, tensor in state_dict.items():
        if not (name.endswith("mamba_alpha") or name.endswith("mamba_gate")):
            continue
        value = float(tensor.detach().cpu().reshape(-1)[0])
        layer_match = LAYER_RE.search(name)
        layer_idx = int(layer_match.group(1)) if layer_match else -1
        rows.append(
            {
                "checkpoint_path": checkpoint_path,
                "checkpoint_name": checkpoint_path.split("/")[-1],
                "step": step,
                "parameter": name,
                "layer_idx": layer_idx,
                "raw_value": value,
                "sigmoid_value": 1.0 / (1.0 + math.exp(-value)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    rows = []
    for checkpoint_path in iter_checkpoint_files(args.checkpoint_glob):
        rows.extend(extract_gate_rows(str(checkpoint_path)))

    rows.sort(key=lambda row: (row["checkpoint_name"], row["layer_idx"], row["parameter"]))
    write_csv(out_dir / "hybrid_gate_values.csv", rows)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        params = sorted({row["parameter"] for row in rows})
        for parameter in params:
            param_rows = sorted((row for row in rows if row["parameter"] == parameter), key=lambda row: row["step"])
            label = parameter
            if param_rows and param_rows[0]["layer_idx"] >= 0:
                label = f"layer {param_rows[0]['layer_idx']}"
            ax.plot([row["step"] for row in param_rows], [row["sigmoid_value"] for row in param_rows], marker="o", label=label)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Sigmoid(gate)")
        ax.set_title("Hybrid gate trajectory")
        if params:
            ax.legend(fontsize=8, ncol=2)
        fig.savefig(out_dir / "hybrid_gate_trajectory.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Skipping gate plot: {exc}")

    print(f"Wrote hybrid gate outputs to {out_dir}")


if __name__ == "__main__":
    main()