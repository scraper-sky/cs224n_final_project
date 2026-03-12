"""this is a numeric token saliency for multiple models"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from scripts.analysis._common import (
    build_math_prompt,
    compute_teacher_forced_loss,
    decode_token_pieces,
    default_data_dir,
    default_device,
    ensure_dir,
    get_record_id,
    is_numeric_token_piece,
    load_model_and_tokenizer,
    parse_checkpoint_map,
    parse_csv_list,
    read_jsonl,
    seed_everything,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saliency of numeric vs non-numeric prompt tokens.")
    parser.add_argument("--models", default="gpt2,mamba,hybrid_model_v2")
    parser.add_argument("--input-path", default=str(default_data_dir() / "olympiad_preprocessed.jsonl"))
    parser.add_argument("--output-dir", default="scripts/analysis/outputs/numeric_token_saliency")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-tokens-per-type", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-map", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    rng = seed_everything(args.seed)
    models = parse_csv_list(args.models)
    checkpoint_map = parse_checkpoint_map(args.checkpoint_map)
    records = read_jsonl(args.input_path, limit=args.max_samples)

    per_token_rows = []
    summary_rows = []

    for model_name in models:
        model, tokenizer, ckpt_path = load_model_and_tokenizer(
            model_name,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            checkpoint_map=checkpoint_map,
        )
        replacement_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        deltas_by_type = {"numeric": [], "control": []}

        for index, record in enumerate(tqdm(records, desc=f"{model_name}_numeric_saliency")):
            prompt = build_math_prompt(record)
            gold = (record.get("final_answer") or "").strip()
            if not gold:
                continue
            base_loss, input_ids, attention_mask, labels, prompt_len = compute_teacher_forced_loss(
                model,
                tokenizer,
                prompt,
                gold,
                device=args.device,
                context_window=args.context_window,
                max_target_tokens=args.max_target_tokens,
            )
            pieces = decode_token_pieces(tokenizer, input_ids[0, :prompt_len].detach().cpu().tolist())
            numeric_positions = [idx for idx, piece in enumerate(pieces) if is_numeric_token_piece(piece)]
            control_positions = [idx for idx, piece in enumerate(pieces) if piece.strip() and not is_numeric_token_piece(piece)]
            rng.shuffle(numeric_positions)
            rng.shuffle(control_positions)
            chosen = {
                "numeric": numeric_positions[: args.max_tokens_per_type],
                "control": control_positions[: args.max_tokens_per_type],
            }

            for token_type, positions in chosen.items():
                for position in positions:
                    ablated_ids = input_ids.clone()
                    ablated_ids[:, position] = replacement_id
                    with torch.inference_mode():
                        out = model(input_ids=ablated_ids, attention_mask=attention_mask, labels=labels)
                    delta_loss = float(out.loss.detach().cpu()) - base_loss
                    deltas_by_type[token_type].append(delta_loss)
                    per_token_rows.append(
                        {
                            "model": model_name,
                            "checkpoint": ckpt_path,
                            "record_id": get_record_id(record, index),
                            "token_type": token_type,
                            "token_position": position,
                            "token_piece": pieces[position],
                            "delta_loss": delta_loss,
                        }
                    )

        for token_type, deltas in deltas_by_type.items():
            summary_rows.append(
                {
                    "model": model_name,
                    "checkpoint": ckpt_path,
                    "token_type": token_type,
                    "mean_delta_loss": sum(deltas) / len(deltas) if deltas else 0.0,
                    "num_tokens": len(deltas),
                }
            )

    write_csv(out_dir / "numeric_token_saliency_per_token.csv", per_token_rows)
    write_csv(out_dir / "numeric_token_saliency_summary.csv", summary_rows)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        models = [row["model"] for row in summary_rows if row["token_type"] == "numeric"]
        numeric_vals = [row["mean_delta_loss"] for row in summary_rows if row["token_type"] == "numeric"]
        control_vals = [row["mean_delta_loss"] for row in summary_rows if row["token_type"] == "control"]
        xs = range(len(models))
        ax.bar([x - 0.2 for x in xs], numeric_vals, width=0.4, label="numeric")
        ax.bar([x + 0.2 for x in xs], control_vals, width=0.4, label="control")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("Mean delta loss")
        ax.set_title("Numeric token saliency")
        ax.legend()
        fig.savefig(out_dir / "numeric_token_saliency.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Skipping numeric saliency plot: {exc}")

    print(f"Wrote numeric token saliency outputs to {out_dir}")


if __name__ == "__main__":
    main()