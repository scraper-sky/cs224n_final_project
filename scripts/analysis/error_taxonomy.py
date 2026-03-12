from __future__ import annotations

import argparse
import re
from collections import Counter

from scripts.analysis._common import ensure_dir, read_jsonl, write_csv


NUM_RE = re.compile(r"-?\d+\.?\d*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic error taxonomy from win/loss predictions.")
    parser.add_argument(
        "--predictions-path",
        default="scripts/analysis/outputs/win_loss/per_example_predictions.jsonl",
    )
    parser.add_argument("--output-dir", default="scripts/analysis/outputs/error_taxonomy")
    return parser.parse_args()


def classify_error(gold: str, pred: str, is_correct: bool) -> str:
    if is_correct:
        return "correct"
    pred = (pred or "").strip().lower()
    gold = (gold or "").strip().lower()
    if not pred:
        return "empty_output"
    pred_nums = NUM_RE.findall(pred)
    gold_nums = NUM_RE.findall(gold)
    if gold_nums and not pred_nums:
        return "no_numeric_answer"
    if pred.endswith("..."):
        return "truncated_output"
    if "final answer" in pred and not pred_nums:
        return "formatting_without_answer"
    if gold_nums and pred_nums:
        gold_val = gold_nums[-1]
        pred_val = pred_nums[-1]
        try:
            if abs(float(gold_val)) == abs(float(pred_val)) and float(gold_val) != float(pred_val):
                return "sign_error"
            if abs(float(gold_val) - float(pred_val)) <= 1:
                return "off_by_one_or_small_arithmetic"
        except ValueError:
            pass
        if len(pred_nums) > 1:
            return "wrong_final_extraction"
        return "numeric_mismatch"
    if len(pred.split()) > 12:
        return "reasoning_trace_no_match"
    return "other_text_mismatch"


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    rows = read_jsonl(args.predictions_path)

    error_rows = []
    summary_counter = Counter()
    for row in rows:
        gold = row.get("gold", "")
        for key, value in row.items():
            if not key.endswith("_prediction"):
                continue
            model = key[: -len("_prediction")]
            pred = value
            is_correct = bool(row.get(f"{model}_correct", False))
            label = classify_error(gold, pred, is_correct)
            error_rows.append(
                {
                    "record_id": row.get("record_id", ""),
                    "model": model,
                    "gold": gold,
                    "prediction": pred,
                    "correct": is_correct,
                    "taxonomy_label": label,
                }
            )
            summary_counter[(model, label)] += 1

    summary_rows = [
        {"model": model, "taxonomy_label": label, "count": count}
        for (model, label), count in sorted(summary_counter.items())
    ]

    write_csv(out_dir / "error_taxonomy_per_example.csv", error_rows)
    write_csv(out_dir / "error_taxonomy_summary.csv", summary_rows)

    try:
        import matplotlib.pyplot as plt

        models = sorted({row["model"] for row in summary_rows})
        labels = sorted({row["taxonomy_label"] for row in summary_rows if row["taxonomy_label"] != "correct"})
        fig, ax = plt.subplots(figsize=(10, 4))
        width = 0.8 / max(1, len(models))
        for model_idx, model in enumerate(models):
            values = []
            for label in labels:
                count = next(
                    (row["count"] for row in summary_rows if row["model"] == model and row["taxonomy_label"] == label),
                    0,
                )
                values.append(count)
            xs = [idx + (model_idx - (len(models) - 1) / 2) * width for idx in range(len(labels))]
            ax.bar(xs, values, width=width, label=model)
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Heuristic error taxonomy")
        ax.legend()
        fig.savefig(out_dir / "error_taxonomy_summary.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Skipping error taxonomy plot: {exc}")

    print(f"Wrote error taxonomy outputs to {out_dir}")


if __name__ == "__main__":
    main()
