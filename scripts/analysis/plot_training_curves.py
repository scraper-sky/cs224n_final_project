from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts.analysis._common import ensure_dir, iter_checkpoint_files, write_csv


STEP_RE = re.compile(r"step\s+(\d+)\s+loss\s+([0-9.eE+-]+)\s+lr\s+([0-9.eE+-]+)")
WARN_RE = re.compile(r"WARNING: NaN loss at step (\d+), skipping")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and plot training logs.")
    parser.add_argument("--log-glob", default="scripts/analysis/outputs/training_logs/*/*.log")
    parser.add_argument("--output-dir", default="scripts/analysis/outputs/training_curves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    point_rows = []
    warning_rows = []
    for log_path in iter_checkpoint_files(args.log_glob):
        model = log_path.stem
        run_tag = log_path.parent.name
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if match := STEP_RE.search(line):
                    point_rows.append(
                        {
                            "run_tag": run_tag,
                            "model": model,
                            "step": int(match.group(1)),
                            "loss": float(match.group(2)),
                            "lr": float(match.group(3)),
                            "log_path": str(log_path),
                        }
                    )
                elif match := WARN_RE.search(line):
                    warning_rows.append(
                        {
                            "run_tag": run_tag,
                            "model": model,
                            "step": int(match.group(1)),
                            "warning": "nan_loss",
                            "log_path": str(log_path),
                        }
                    )

    write_csv(out_dir / "training_points.csv", point_rows)
    write_csv(out_dir / "training_warnings.csv", warning_rows)

    try:
        import matplotlib.pyplot as plt

        for run_tag in sorted({row["run_tag"] for row in point_rows}):
            run_rows = [row for row in point_rows if row["run_tag"] == run_tag]
            if not run_rows:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for model in sorted({row["model"] for row in run_rows}):
                model_rows = sorted((row for row in run_rows if row["model"] == model), key=lambda row: row["step"])
                axes[0].plot([row["step"] for row in model_rows], [row["loss"] for row in model_rows], marker="o", label=model)
                axes[1].plot([row["step"] for row in model_rows], [row["lr"] for row in model_rows], marker="o", label=model)
            axes[0].set_title(f"{run_tag}: loss")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[1].set_title(f"{run_tag}: learning rate")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("LR")
            axes[0].legend()
            axes[1].legend()
            fig.savefig(out_dir / f"{run_tag}_training_curves.png", bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        print(f"Skipping plot generation: {exc}")

    print(f"Wrote training curve outputs to {out_dir}")


if __name__ == "__main__":
    main()