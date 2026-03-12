#!/usr/bin/env python3
"""runs all interpretability scripts for all models (win/loss, gates, saliency)""""
import glob
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_MAP = "hybrid=checkpoints/hybrid_step500.pt"


def run(cmd: list[str], cwd: Path | None = None) -> int:
    return subprocess.run(cmd, cwd=cwd or PROJECT_ROOT).returncode


def main() -> int:
    os.chdir(PROJECT_ROOT)
    ckpts = sorted(glob.glob(str(PROJECT_ROOT / "checkpoints" / "hybrid_step*.pt")), key=lambda p: int(p.split("step")[1].replace(".pt", "")) if "step" in p else 0)
    ckpt_map = CKPT_MAP
    if ckpts:
        ckpt_path = ckpts[-1]
        ckpt_map = f"hybrid={ckpt_path}"
        print(f"Using checkpoint: {ckpt_path}")

    steps = [
        (["python", "scripts/analysis/win_loss_analysis.py", "--models", "gpt2,mamba,hybrid", "--max-samples", "50", "--checkpoint-map", ckpt_map], "Win/loss"),
        (["python", "scripts/analysis/error_taxonomy.py"], "Error taxonomy"),
        (["python", "scripts/analysis/context_sensitivity.py", "--models", "gpt2,mamba,hybrid", "--mode", "both", "--lengths", "128,256,512,1024", "--checkpoint-map", ckpt_map], "Context sensitivity"),
        (["python", "scripts/analysis/hybrid_gate_analysis.py", "--checkpoint-glob", "checkpoints/*.pt"], "Hybrid gate"),
        (["python", "scripts/analysis/attention_analysis.py", "--model-name", "gpt2", "--max-samples", "20"], "Attention (GPT-2)"),
        (["python", "scripts/analysis/numeric_token_saliency.py", "--models", "gpt2,mamba,hybrid", "--max-samples", "15", "--checkpoint-map", ckpt_map], "Numeric saliency"),
        (["python", "scripts/analysis/mamba_saliency.py", "--model-name", "mamba", "--max-samples", "15"], "Mamba saliency"),
    ]
    failed = []
    for cmd, name in steps:
        print(f"\n=== {name} ===")
        if run(cmd) != 0:
            failed.append(name)
    if failed:
        print(f"\nFailed: {failed}")
        return 1
    print("\nAll analyses complete. Outputs in scripts/analysis/outputs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
