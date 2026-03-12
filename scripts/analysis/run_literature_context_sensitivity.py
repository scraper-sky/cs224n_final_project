#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Literature perplexity vs context length (one plot)")
    parser.add_argument("--models", default="gpt2,mamba,hybrid")
    parser.add_argument("--lengths", default="128,256,512,1024")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--checkpoint-map", default="hybrid=checkpoints/hybrid_step500.pt")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "context_sensitivity.py"),
        "--mode", "literature",
        "--models", args.models,
        "--lengths", args.lengths,
        "--max-samples", str(args.max_samples),
        "--checkpoint-map", args.checkpoint_map,
    ]
    code = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
    if code == 0:
        out = PROJECT_ROOT / "scripts/analysis/outputs/context_sensitivity/context_sensitivity_literature.png"
        print(f"\nPlot saved to: {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
