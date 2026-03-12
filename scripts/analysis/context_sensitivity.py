"""
Re-run evaluation at several context lengths.
Measures how math exact match and/or literature perplexity change as visible context shrinks.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
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
    parse_int_list,
    read_jsonl,
    truncate_prompt_by_tokens,
    write_csv,
)


def compute_literature_perplexity(
    model,
    tokenizer,
    chunks_path: str,
    device: str,
    max_context_tokens: int,
    max_target_tokens: int,
    max_samples: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with open(chunks_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    lines = lines[:max_samples] if max_samples else lines

    with torch.inference_mode():
        for line in tqdm(lines, desc=f"perplexity_ctx{max_context_tokens}"):
            obj = json.loads(line)
            ctx_ids = tokenizer.encode(obj.get("context_text", ""), add_special_tokens=False)
            tgt_ids = tokenizer.encode(obj.get("target_text", ""), add_special_tokens=False)[:max_target_tokens]
            ctx_budget = max(0, max_context_tokens - len(tgt_ids))
            ctx_ids = ctx_ids[-ctx_budget:] if ctx_budget > 0 else []
            all_ids = ctx_ids + tgt_ids
            if not tgt_ids:
                continue
            input_ids = torch.tensor([all_ids], dtype=torch.long).to(device)
            labels = torch.full_like(input_ids, -100)
            labels[0, len(ctx_ids) :] = input_ids[0, len(ctx_ids) :]
            out = model(input_ids=input_ids, labels=labels)
            n_tgt = len(tgt_ids)
            total_loss += out.loss.item() * n_tgt
            total_tokens += n_tgt
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def compute_math_accuracy(
    model,
    tokenizer,
    records: list,
    device: str,
    max_context_tokens: int,
    context_window: int,
    max_new_tokens: int,
) -> float:
    correct = 0
    for index, record in enumerate(tqdm(records, desc=f"math_ctx{max_context_tokens}")):
        prompt = build_math_prompt(record)
        prompt = truncate_prompt_by_tokens(tokenizer, prompt, max_context_tokens)
        gold = (record.get("final_answer") or "").strip()
        pred_text, _, _ = generate_prediction(
            model,
            tokenizer,
            prompt,
            device=device,
            context_window=min(context_window, max_context_tokens + max_new_tokens),
            max_new_tokens=max_new_tokens,
        )
        if match_answer(gold, pred_text):
            correct += 1
    return correct / len(records) if records else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context-length sensitivity: math accuracy and/or literature perplexity.")
    parser.add_argument("--models", default="gpt2,mamba,hybrid")
    parser.add_argument("--input-path", default=str(default_data_dir() / "olympiad_preprocessed.jsonl"))
    parser.add_argument(
        "--literature-path",
        default=str(default_data_dir() / "gutenberg_7000_1192.jsonl"),
    )
    parser.add_argument("--output-dir", default="scripts/analysis/outputs/context_sensitivity")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--context-window", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-target-tokens", type=int, default=64)
    parser.add_argument("--lengths", default="128,256,512,1024", help="Comma-separated context lengths in tokens")
    parser.add_argument("--mode", choices=["math", "literature", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples for math and literature")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-map", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    models = parse_csv_list(args.models)
    lengths = parse_int_list(args.lengths)
    checkpoint_map = parse_checkpoint_map(args.checkpoint_map)

    rows = []

    for model_name in models:
        model, tokenizer, ckpt_path = load_model_and_tokenizer(
            model_name,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            checkpoint_map=checkpoint_map,
        )

        for length in lengths:
            if args.mode in ("math", "both"):
                records = read_jsonl(args.input_path, limit=args.max_samples)
                acc = compute_math_accuracy(
                    model,
                    tokenizer,
                    records,
                    args.device,
                    max_context_tokens=length,
                    context_window=args.context_window,
                    max_new_tokens=args.max_new_tokens,
                )
                rows.append(
                    {
                        "model": model_name,
                        "checkpoint": ckpt_path,
                        "context_length": length,
                        "task": "math",
                        "metric": "exact_match_acc",
                        "value": acc,
                    }
                )

            if args.mode in ("literature", "both"):
                try:
                    ppl = compute_literature_perplexity(
                        model,
                        tokenizer,
                        args.literature_path,
                        args.device,
                        max_context_tokens=length,
                        max_target_tokens=args.max_target_tokens,
                        max_samples=args.max_samples,
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "checkpoint": ckpt_path,
                            "context_length": length,
                            "task": "literature",
                            "metric": "perplexity",
                            "value": ppl,
                        }
                    )
                except FileNotFoundError:
                    print(f"Skipping literature for {model_name} @ {length}: {args.literature_path} not found")

    write_csv(out_dir / "context_sensitivity.csv", rows)

    try:
        import matplotlib.pyplot as plt

        math_rows = [r for r in rows if r["task"] == "math"]
        lit_rows = [r for r in rows if r["task"] == "literature"]

        if math_rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            for m in models:
                mrows = [r for r in math_rows if r["model"] == m]
                if mrows:
                    ax.plot(
                        [r["context_length"] for r in mrows],
                        [r["value"] for r in mrows],
                        marker="o",
                        label=m,
                    )
            ax.set_xlabel("Context length (tokens)")
            ax.set_ylabel("Exact match accuracy")
            ax.set_title("Math: accuracy vs context length")
            ax.legend()
            fig.savefig(out_dir / "context_sensitivity_math.png", bbox_inches="tight")
            plt.close(fig)

        if lit_rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            for m in models:
                mrows = [r for r in lit_rows if r["model"] == m]
                if mrows:
                    ax.plot(
                        [r["context_length"] for r in mrows],
                        [r["value"] for r in mrows],
                        marker="o",
                        label=m,
                    )
            ax.set_xlabel("Context length (tokens)")
            ax.set_ylabel("Perplexity")
            ax.set_title("Literature: perplexity vs context length")
            ax.legend()
            fig.savefig(out_dir / "context_sensitivity_literature.png", bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        print(f"Skipping context sensitivity plots: {exc}")

    print(f"Wrote context sensitivity outputs to {out_dir}")


if __name__ == "__main__":
    main()
