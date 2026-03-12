"""this is a Mamba saliency analysis on math examples"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from scripts.analysis._common import (
    build_math_prompt,
    build_teacher_forced_batch,
    chunk_token_spans,
    compute_teacher_forced_loss,
    default_data_dir,
    default_device,
    ensure_dir,
    get_record_id,
    load_model_and_tokenizer,
    parse_checkpoint_map,
    read_jsonl,
    replace_token_span,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perturbation saliency and hidden-state sensitivity for Mamba.")
    parser.add_argument("--model-name", default="mamba")
    parser.add_argument("--input-path", default=str(default_data_dir() / "olympiad_preprocessed.jsonl"))
    parser.add_argument("--output-dir", default="scripts/analysis/outputs/mamba_saliency")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--num-spans", type=int, default=8)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-map", default="")
    return parser.parse_args()


def forward_with_hidden_states(model, input_ids, attention_mask, labels):
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    checkpoint_map = parse_checkpoint_map(args.checkpoint_map)
    model, tokenizer, ckpt_path = load_model_and_tokenizer(
        args.model_name,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        checkpoint_map=checkpoint_map,
    )
    if not hasattr(model, "forward"):
        raise ValueError("Model does not support forward pass.")

    records = read_jsonl(args.input_path, limit=args.max_samples)
    example_rows = []
    layer_rows = []
    replacement_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for index, record in enumerate(tqdm(records, desc="mamba_saliency")):
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
        base_out = forward_with_hidden_states(model, input_ids, attention_mask, labels)
        base_hidden = [hidden.detach().cpu() for hidden in base_out.hidden_states]

        spans = chunk_token_spans(prompt_len, args.num_spans)
        span_results = []
        for span in spans:
            ablated_ids = replace_token_span(input_ids, span, replacement_id)
            with torch.inference_mode():
                out = model(input_ids=ablated_ids, attention_mask=attention_mask, labels=labels)
            span_results.append(
                {
                    "span_start": span[0],
                    "span_end": span[1],
                    "loss": float(out.loss.detach().cpu()),
                    "delta_loss": float(out.loss.detach().cpu()) - base_loss,
                }
            )

        if not span_results:
            continue
        best_span = max(span_results, key=lambda row: row["delta_loss"])
        ablated_ids = replace_token_span(input_ids, (best_span["span_start"], best_span["span_end"]), replacement_id)
        ablated_out = forward_with_hidden_states(model, ablated_ids, attention_mask, labels)
        ablated_hidden = [hidden.detach().cpu() for hidden in ablated_out.hidden_states]

        for layer_idx, (base_layer, ablated_layer) in enumerate(zip(base_hidden, ablated_hidden)):
            hidden_delta = (ablated_layer - base_layer).abs().mean().item()
            layer_rows.append(
                {
                    "record_id": get_record_id(record, index),
                    "layer_idx": layer_idx,
                    "hidden_delta_mean_abs": hidden_delta,
                    "best_span_start": best_span["span_start"],
                    "best_span_end": best_span["span_end"],
                }
            )

        example_rows.append(
            {
                "record_id": get_record_id(record, index),
                "checkpoint": ckpt_path,
                "prompt": prompt,
                "gold": gold,
                "base_loss": base_loss,
                "best_span_start": best_span["span_start"],
                "best_span_end": best_span["span_end"],
                "best_delta_loss": best_span["delta_loss"],
                "span_results": span_results,
            }
        )

    write_jsonl(out_dir / f"{args.model_name}_mamba_saliency_examples.jsonl", example_rows)
    write_csv(out_dir / f"{args.model_name}_mamba_hidden_deltas.csv", layer_rows)

    try:
        import matplotlib.pyplot as plt

        summary = {}
        for row in layer_rows:
            summary.setdefault(row["layer_idx"], []).append(row["hidden_delta_mean_abs"])
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = sorted(summary)
        ys = [sum(summary[x]) / len(summary[x]) for x in xs]
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean |hidden delta|")
        ax.set_title(f"{args.model_name}: hidden-state sensitivity")
        fig.savefig(out_dir / f"{args.model_name}_mamba_hidden_deltas.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Skipping Mamba saliency plot: {exc}")

    print(f"Wrote Mamba saliency outputs to {out_dir}")


if __name__ == "__main__":
    main()