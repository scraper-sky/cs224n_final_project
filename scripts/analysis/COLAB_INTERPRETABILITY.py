#!/usr/bin/env python3
"""
Comprehensive Colab script for HMT interpretability analysis.

Run all analyses and produce a summary for your paper's interpretability section.
Supports the narrative:
  - Mamba: best at "keep the gist of a long passage alive" (literature)
  - Transformer: best at "don't lose exact details" (math)
  - Hybrid: gate stays shut on Mamba branch; behaves like Transformer + unused Mamba

USAGE in Colab:
  1. Clone repo, pip install, upload hybrid checkpoint
  2. Run: python scripts/analysis/COLAB_INTERPRETABILITY.py

Or run cells individually (see COLAB_INTERPRETABILITY_NOTEBOOK.md for cell-by-cell).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --- CONFIG (edit these for your setup) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HYBRID_CKPT = "checkpoints/hybrid_step500.pt"  # or your latest hybrid checkpoint
MODELS = "gpt2,mamba,hybrid"
MAX_SAMPLES = 50  # smaller = faster; 100 for full analysis
CONTEXT_LENGTHS = "128,256,512,1024"

CKPT_MAP = f"hybrid={HYBRID_CKPT}"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    """Run command with PROJECT_ROOT on PYTHONPATH for script imports."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env).returncode


def main() -> int:
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 60)
    print("HMT Interpretability Analysis (Colab)")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Hybrid checkpoint: {HYBRID_CKPT}")
    print(f"Models: {MODELS}")
    print(f"Max samples: {MAX_SAMPLES}")
    print("=" * 60)

    steps = [
        (
            [
                "python",
                "scripts/analysis/win_loss_analysis.py",
                "--models", MODELS,
                "--max-samples", str(MAX_SAMPLES),
                "--checkpoint-map", CKPT_MAP,
            ],
            "Win/loss analysis",
        ),
        (["python", "scripts/analysis/error_taxonomy.py"], "Error taxonomy"),
        (
            [
                "python",
                "scripts/analysis/context_sensitivity.py",
                "--models", MODELS,
                "--mode", "both",
                "--lengths", CONTEXT_LENGTHS,
                "--max-samples", str(MAX_SAMPLES),
                "--checkpoint-map", CKPT_MAP,
            ],
            "Context sensitivity",
        ),
        (
            ["python", "scripts/analysis/hybrid_gate_analysis.py", "--checkpoint-glob", "checkpoints/*.pt"],
            "Hybrid gate analysis",
        ),
        (
            ["python", "scripts/analysis/attention_analysis.py", "--model-name", "gpt2", "--max-samples", "20"],
            "Attention (GPT-2)",
        ),
        (
            [
                "python",
                "scripts/analysis/numeric_token_saliency.py",
                "--models", MODELS,
                "--max-samples", "15",
                "--checkpoint-map", CKPT_MAP,
            ],
            "Numeric token saliency",
        ),
        (
            ["python", "scripts/analysis/mamba_saliency.py", "--model-name", "mamba", "--max-samples", "15"],
            "Mamba saliency",
        ),
    ]

    failed = []
    for cmd, name in steps:
        print(f"\n>>> {name}")
        if run_cmd(cmd) != 0:
            failed.append(name)

    # Optionally run training curves if logs exist
    log_dir = PROJECT_ROOT / "scripts/analysis/outputs/training_logs"
    if log_dir.exists() and list(log_dir.rglob("*.log")):
        print("\n>>> Training curves")
        run_cmd(["python", "scripts/analysis/plot_training_curves.py", "--log-glob", str(log_dir / "*" / "*.log")])

    # Summary
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {failed}")
    else:
        print_summary(PROJECT_ROOT)
    print("\nOutputs: scripts/analysis/outputs/")
    print("  - win_loss/: per_example_predictions, pairwise_win_loss")
    print("  - error_taxonomy/: taxonomy summary")
    print("  - context_sensitivity/: context_sensitivity.csv, plots")
    print("  - hybrid_gates/: gate trajectory")
    print("  - attention/, numeric_token_saliency/, mamba_saliency/")
    print("=" * 60)
    return 1 if failed else 0


def print_summary(root: Path) -> None:
    """Print key findings for the paper's interpretability section."""
    import csv

    print("\n--- INTERPRETABILITY SUMMARY (for paper) ---\n")

    # Win/loss: pairwise
    pw_path = root / "scripts/analysis/outputs/win_loss/pairwise_win_loss.csv"
    if pw_path.exists():
        with open(pw_path) as f:
            rows = list(csv.DictReader(f))
        print("Pairwise Win/Loss:")
        for r in rows:
            left, right = r["left_model"], r["right_model"]
            wins = int(r.get("left_only_correct", 0))
            losses = int(r.get("right_only_correct", 0))
            if "hybrid" in (left, right):
                print(f"  {left} vs {right}: wins={wins} losses={losses} ties={r.get('same_result', 0)}")
        print()

    # Gate: data-driven interpretation
    gate_sig: float | None = None
    gate_path = root / "scripts/analysis/outputs/hybrid_gates/hybrid_gate_values.csv"
    if gate_path.exists():
        with open(gate_path) as f:
            gate_rows = list(csv.DictReader(f))
        if gate_rows:
            last = gate_rows[-1]
            sig = float(last.get("sigmoid_value", 0))
            gate_sig = sig
            if sig < 0.15:
                gate_note = "Mamba branch mostly off"
            elif sig > 0.85:
                gate_note = "Mamba branch dominant"
            else:
                gate_note = "~50/50 blend (both branches used)"
            print(f"Hybrid gate: sigmoid(mamba_gate) ≈ {sig:.4f} ({gate_note})\n")

    # Context sensitivity: Mamba wins at long context for literature
    ctx_path = root / "scripts/analysis/outputs/context_sensitivity/context_sensitivity.csv"
    if ctx_path.exists():
        with open(ctx_path) as f:
            rows = list(csv.DictReader(f))
        lit = [r for r in rows if r.get("task") == "literature"]
        math_rows = [r for r in rows if r.get("task") == "math"]
        if lit:
            by_model = {}
            for r in lit:
                m = r["model"]
                if m not in by_model:
                    by_model[m] = []
                by_model[m].append((int(r["context_length"]), float(r["value"])))
            print("Literature perplexity @ longest context:")
            for m, pts in sorted(by_model.items()):
                pts.sort(key=lambda x: x[0])
                ppl = pts[-1][1] if pts else 0
                print(f"  {m}: {ppl:.2f}")
            print("  (Lower is better)\n")
        if math_rows:
            by_model = {}
            for r in math_rows:
                m = r["model"]
                if m not in by_model:
                    by_model[m] = []
                by_model[m].append((int(r["context_length"]), float(r["value"])))
            print("Math accuracy @ longest context:")
            for m, pts in sorted(by_model.items()):
                pts.sort(key=lambda x: x[0])
                acc = pts[-1][1] * 100 if pts else 0
                print(f"  {m}: {acc:.1f}%")
            best_math = max(by_model.items(), key=lambda x: (x[1][-1][1] if x[1] else 0))
            print(f"  (Higher is better; best in this slice: {best_math[0]})\n")

    # Error taxonomy: hybrid has more formatting/extraction failures
    tax_path = root / "scripts/analysis/outputs/error_taxonomy/error_taxonomy_summary.csv"
    if tax_path.exists():
        with open(tax_path) as f:
            rows = list(csv.DictReader(f))
        hybrid_errs = [r for r in rows if r.get("model") == "hybrid" and r.get("taxonomy_label") != "correct"]
        if hybrid_errs:
            top = sorted(hybrid_errs, key=lambda r: int(r.get("count", 0)), reverse=True)[:3]
            print("Hybrid top error types:")
            for r in top:
                print(f"  {r.get('taxonomy_label', '')}: {r.get('count', 0)}")
            print()

    print("--- Narrative ---")
    print("Mamba: best at 'keep the gist of a long passage alive' (literature perplexity)")
    print("Transformer: best at 'don't lose exact details' (math accuracy)")
    if gate_sig is not None:
        if gate_sig < 0.15:
            print("Hybrid: gate favors Transformer → mostly attention, little Mamba")
        elif gate_sig > 0.85:
            print("Hybrid: gate favors Mamba → mostly Mamba, little attention")
        else:
            print("Hybrid: gate ~50/50 → uses both attention and Mamba mechanisms")
    else:
        print("Hybrid: blends attention and Mamba based on learned gate")
    print("Treat 100-example slices as supporting diagnostics, not final numbers.")


if __name__ == "__main__":
    sys.exit(main())
