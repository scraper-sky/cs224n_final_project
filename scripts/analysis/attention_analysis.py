from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from tqdm import tqdm

from scripts.analysis._common import (
    build_math_prompt,
    decode_token_pieces,
    default_data_dir,
    default_device,
    ensure_dir,
    get_record_id,
    is_numeric_token_piece,
    load_model_and_tokenizer,
    parse_checkpoint_map,
    read_jsonl,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention summary analysis for GPT-2 and hybrid attention layers.")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--input-path", default=str(default_data_dir() / "olympiad_preprocessed.jsonl"))
    parser.add_argument("--output-dir", default="scripts/analysis/outputs/attention")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-map", default="")
    return parser.parse_args()


def _build_bool_attention_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    _, seq_len = input_ids.shape
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0)
    if attention_mask is None:
        return causal
    padding = attention_mask[:, None, None, :].eq(0)
    return causal | padding


def collect_gpt2_attentions(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    if hasattr(model, "config"):
        setattr(model.config, "_attn_implementation", "eager")
    if hasattr(model, "transformer") and hasattr(model.transformer, "config"):
        setattr(model.transformer.config, "_attn_implementation", "eager")

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    attentions = getattr(out, "attentions", None)
    if attentions is None or any(attn is None for attn in attentions):
        raise RuntimeError(
            "GPT-2 attentions are unavailable (returned None). "
            "Try transformers eager attention mode or use a model/backend that exposes attentions."
        )
    return [attn.detach().cpu() for attn in attentions]


def collect_custom_attentions(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    captured = []
    hooks = []
    for layer in model.layers:
        hooks.append(
            layer.ln_1.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output.detach().clone())
            )
        )
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for hook in hooks:
            hook.remove()

    mask = _build_bool_attention_mask(input_ids, attention_mask)
    probs = []
    with torch.no_grad():
        for hidden, layer in zip(captured, model.layers):
            bsz, seq_len, hidden_size = hidden.shape
            qkv = layer.c_attn(hidden)
            q, k, _v = qkv.split(hidden_size, dim=-1)
            q = q.view(bsz, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, layer.num_heads, layer.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * layer.scale
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            attn_probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
            attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs.append(attn_probs.detach().cpu())
    return probs


def summarize_last_token(attentions: list[torch.Tensor], token_pieces: list[str], *, top_k: int) -> tuple[list[dict], list[dict]]:
    numeric_positions = [idx for idx, piece in enumerate(token_pieces) if is_numeric_token_piece(piece)]
    per_layer_rows = []
    aggregate_rows = []
    for layer_idx, layer_attn in enumerate(attentions):
        avg_attn = layer_attn[0].mean(dim=0)
        last_idx = avg_attn.shape[0] - 1
        last_vec = avg_attn[last_idx]
        denom = float(last_vec.sum().item()) or 1.0
        weighted_distance = 0.0
        for pos, weight in enumerate(last_vec.tolist()):
            weighted_distance += (last_idx - pos) * weight
        numeric_mass = sum(last_vec[pos].item() for pos in numeric_positions) if numeric_positions else 0.0
        top_indices = torch.topk(last_vec, k=min(top_k, last_vec.numel())).indices.tolist()
        top_tokens = [{"position": idx, "piece": token_pieces[idx], "weight": float(last_vec[idx].item())} for idx in top_indices]
        per_layer_rows.append(
            {
                "layer_idx": layer_idx,
                "mean_distance": weighted_distance / denom,
                "numeric_mass": numeric_mass,
                "top_tokens": top_tokens,
            }
        )
        aggregate_rows.append(
            {
                "layer_idx": layer_idx,
                "mean_distance": weighted_distance / denom,
                "numeric_mass": numeric_mass,
            }
        )
    return per_layer_rows, aggregate_rows


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
    records = read_jsonl(args.input_path, limit=args.max_samples)

    if hasattr(model, "transformer"):
        collector = collect_gpt2_attentions
    elif hasattr(model, "layers") and hasattr(model.layers[0], "c_attn") and hasattr(model.layers[0], "ln_1"):
        collector = collect_custom_attentions
    else:
        raise ValueError(f"Unsupported model for attention analysis: {args.model_name}")

    per_example_rows = []
    aggregate = defaultdict(list)

    for index, record in enumerate(tqdm(records, desc="attention")):
        prompt = build_math_prompt(record)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context_window)
        input_ids = encoded["input_ids"].to(args.device)
        attention_mask = encoded["attention_mask"].to(args.device)
        attentions = collector(model, input_ids, attention_mask)
        token_pieces = decode_token_pieces(tokenizer, input_ids[0].detach().cpu().tolist())
        layer_rows, aggregate_rows = summarize_last_token(attentions, token_pieces, top_k=args.top_k)
        for row in aggregate_rows:
            aggregate[row["layer_idx"]].append(row)
        per_example_rows.append(
            {
                "record_id": get_record_id(record, index),
                "checkpoint": ckpt_path,
                "prompt": prompt,
                "token_pieces": token_pieces,
                "layer_summaries": layer_rows,
            }
        )

    summary_rows = []
    for layer_idx, rows in sorted(aggregate.items()):
        summary_rows.append(
            {
                "model": args.model_name,
                "checkpoint": ckpt_path,
                "layer_idx": layer_idx,
                "mean_distance": sum(row["mean_distance"] for row in rows) / len(rows),
                "numeric_mass": sum(row["numeric_mass"] for row in rows) / len(rows),
                "num_examples": len(rows),
            }
        )

    write_jsonl(out_dir / f"{args.model_name}_attention_examples.jsonl", per_example_rows)
    write_csv(out_dir / f"{args.model_name}_attention_summary.csv", summary_rows)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([row["layer_idx"] for row in summary_rows], [row["mean_distance"] for row in summary_rows], marker="o")
        axes[0].set_title(f"{args.model_name}: mean attention distance")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Distance from last token")
        axes[1].plot([row["layer_idx"] for row in summary_rows], [row["numeric_mass"] for row in summary_rows], marker="o")
        axes[1].set_title(f"{args.model_name}: attention mass on numeric tokens")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Attention mass")
        fig.savefig(out_dir / f"{args.model_name}_attention_summary.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Skipping attention plots: {exc}")

    print(f"Wrote attention analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()