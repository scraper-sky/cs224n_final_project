from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from scripts.analysis._common import (
    build_math_prompt,
    default_data_dir,
    default_device,
    ensure_dir,
    generate_prediction,
    get_record_id,
    load_model_and_tokenizer,
    match_answer,
    parse_checkpoint_map,
    parse_csv_list,
    read_jsonl,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-example win/loss analysis across models.")
    parser.add_argument("--models", default="gpt2,mamba,hybrid_model_v2")
    parser.add_argument("--input-path", default=str(default_data_dir() / "olympiad_preprocessed.jsonl"))
    parser.add_argument("--output-dir", default="scripts/analysis/outputs/win_loss")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-map", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    models = parse_csv_list(args.models)
    checkpoint_map = parse_checkpoint_map(args.checkpoint_map)
    records = read_jsonl(args.input_path, limit=args.max_samples)

    loaded = {}
    for model_name in models:
        loaded[model_name] = load_model_and_tokenizer(
            model_name,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            checkpoint_map=checkpoint_map,
        )

    per_example_rows = []
    accuracy_rows = []
    pairwise_counts = Counter()
    model_correct = Counter()
    agreement_counts = Counter()

    for index, record in enumerate(tqdm(records, desc="win_loss")):
        prompt = build_math_prompt(record)
        gold = (record.get("final_answer") or "").strip()
        row = {
            "record_id": get_record_id(record, index),
            "dataset": record.get("dataset", "math"),
            "gold": gold,
            "question": (record.get("question") or "").strip(),
            "prompt": prompt,
            "predictions": {},
            "correct": {},
        }
        for model_name, (model, tokenizer, ckpt_path) in loaded.items():
            pred_text, prompt_ids, new_ids = generate_prediction(
                model,
                tokenizer,
                prompt,
                device=args.device,
                context_window=args.context_window,
                max_new_tokens=args.max_new_tokens,
            )
            is_correct = match_answer(gold, pred_text)
            row["predictions"][model_name] = pred_text
            row["correct"][model_name] = is_correct
            row[f"{model_name}_prediction"] = pred_text
            row[f"{model_name}_correct"] = is_correct
            row[f"{model_name}_checkpoint"] = ckpt_path
            row[f"{model_name}_prompt_tokens"] = len(prompt_ids)
            row[f"{model_name}_generated_tokens"] = len(new_ids)
            model_correct[model_name] += int(is_correct)

        correctness_signature = tuple(int(row["correct"][m]) for m in models)
        agreement_counts[correctness_signature] += 1
        for left, right in itertools.permutations(models, 2):
            left_correct = row["correct"][left]
            right_correct = row["correct"][right]
            if left_correct and not right_correct:
                pairwise_counts[(left, right, "wins")] += 1
            elif not left_correct and right_correct:
                pairwise_counts[(left, right, "losses")] += 1
            else:
                pairwise_counts[(left, right, "ties")] += 1
        per_example_rows.append(row)

    for model_name in models:
        accuracy_rows.append(
            {
                "model": model_name,
                "correct": model_correct[model_name],
                "total": len(per_example_rows),
                "accuracy": model_correct[model_name] / max(1, len(per_example_rows)),
            }
        )

    pairwise_rows = []
    for left, right in itertools.permutations(models, 2):
        pairwise_rows.append(
            {
                "left_model": left,
                "right_model": right,
                "left_only_correct": pairwise_counts[(left, right, "wins")],
                "right_only_correct": pairwise_counts[(left, right, "losses")],
                "same_result": pairwise_counts[(left, right, "ties")],
            }
        )

    agreement_rows = []
    for signature, count in sorted(agreement_counts.items()):
        agreement_rows.append(
            {
                "signature": json.dumps({model: bool(bit) for model, bit in zip(models, signature)}),
                "count": count,
            }
        )

    write_jsonl(out_dir / "per_example_predictions.jsonl", per_example_rows)
    write_csv(out_dir / "accuracy_summary.csv", accuracy_rows)
    write_csv(out_dir / "pairwise_win_loss.csv", pairwise_rows)
    write_csv(out_dir / "agreement_patterns.csv", agreement_rows)

    print(f"Wrote win/loss outputs to {out_dir}")


if __name__ == "__main__":
    main()