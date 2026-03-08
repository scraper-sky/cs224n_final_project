"""
Run in Colab after cloning to diagnose perplexity. Run from project root:
  python scripts/diagnose_perplexity.py
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
CHUNKS = os.environ.get("GUTENBERG_CHUNKS_JSONL", os.path.join(DATA_DIR, "gutenberg_7000_1192.jsonl"))


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== 1. Verify hybrid_model has LayerNorm ===")
    with open(os.path.join(PROJECT_ROOT, "src/models/hybrid_model.py")) as f:
        content = f.read()
    if "LayerNorm" in content and "mamba_branch[2]" in content:
        print("OK: LayerNorm in mamba_branch found")
    else:
        print("MISSING: LayerNorm not found - you may have old code")

    print("\n=== 2. GPT-2 baseline perplexity (no hybrid) ===")
    from src.models import get_model

    model, tokenizer = get_model("gpt2", device=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import json
    import math
    from tqdm import tqdm

    context_window = 1024
    max_target_tokens = 512
    max_samples = 50

    with open(CHUNKS, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()][:max_samples]

    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for line in tqdm(lines, desc="gpt2 perplexity"):
            obj = json.loads(line)
            ctx_ids = tokenizer.encode(obj["context_text"], add_special_tokens=False)
            tgt_ids = tokenizer.encode(obj["target_text"], add_special_tokens=False)[:max_target_tokens]
            ctx_budget = context_window - len(tgt_ids)
            ctx_ids = ctx_ids[-ctx_budget:] if ctx_budget > 0 else []
            all_ids = ctx_ids + tgt_ids
            input_ids = torch.tensor([all_ids], dtype=torch.long).to(device)
            labels = torch.full_like(input_ids, -100)
            labels[0, len(ctx_ids) :] = input_ids[0, len(ctx_ids) :]
            out = model(input_ids=input_ids, labels=labels)
            total_loss += out.loss.item() * len(tgt_ids)
            total_tokens += len(tgt_ids)

    ppl = math.exp(total_loss / total_tokens)
    print(f"GPT-2 perplexity: {ppl:.2f}")
    if ppl > 1000:
        print("WARNING: GPT-2 also has bad perplexity - check eval setup or data")
    else:
        print("GPT-2 baseline is good - hybrid should aim for similar")

    print("\n=== 3. Hybrid (no checkpoint) perplexity ===")
    model, _ = get_model("hybrid", device=device, freeze_gpt2=True)
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for line in tqdm(lines, desc="hybrid (init) perplexity"):
            obj = json.loads(line)
            ctx_ids = tokenizer.encode(obj["context_text"], add_special_tokens=False)
            tgt_ids = tokenizer.encode(obj["target_text"], add_special_tokens=False)[:max_target_tokens]
            ctx_budget = context_window - len(tgt_ids)
            ctx_ids = ctx_ids[-ctx_budget:] if ctx_budget > 0 else []
            all_ids = ctx_ids + tgt_ids
            input_ids = torch.tensor([all_ids], dtype=torch.long).to(device)
            labels = torch.full_like(input_ids, -100)
            labels[0, len(ctx_ids) :] = input_ids[0, len(ctx_ids) :]
            out = model(input_ids=input_ids, labels=labels)
            total_loss += out.loss.item() * len(tgt_ids)
            total_tokens += len(tgt_ids)

    ppl_hybrid_init = math.exp(total_loss / total_tokens)
    print(f"Hybrid (untrained) perplexity: {ppl_hybrid_init:.2f}")
    print("\nDone. If GPT-2 is good but hybrid-init is bad, the architecture may need tuning.")
    print("If both are bad, the eval or data may be wrong.")


if __name__ == "__main__":
    main()
