import json
import math
import os
import re
import sys

import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_perplexity(model, tokenizer, chunks_path, device, context_window, max_target_tokens, max_samples):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with open(chunks_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_samples:
        lines = lines[:max_samples]

    with torch.inference_mode():
        for line in tqdm(lines, desc="perplexity"):
                obj = json.loads(line)
                ctx_ids = tokenizer.encode(obj["context_text"], add_special_tokens=False)
                tgt_ids = tokenizer.encode(obj["target_text"], add_special_tokens=False)
                tgt_ids = tgt_ids[:max_target_tokens]
                ctx_budget = context_window - len(tgt_ids)
                ctx_ids = ctx_ids[-ctx_budget:] if ctx_budget > 0 else []
                all_ids = ctx_ids + tgt_ids
                input_ids = torch.tensor([all_ids], dtype=torch.long).to(device)
                labels = torch.full_like(input_ids, -100)
                labels[0, len(ctx_ids):] = input_ids[0, len(ctx_ids):]
                out = model(input_ids=input_ids, labels=labels)
                n_tgt = len(tgt_ids)
                total_loss += out.loss.item() * n_tgt
                total_tokens += n_tgt
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def _normalize(s: str) -> str:
    return s.strip().lower()


def _match_answer(gold: str, predicted: str) -> bool:
    if gold in predicted:
        return True
    pred_clean = "".join(c for c in predicted if c.isalnum() or c in ".,/-")
    gold_clean = "".join(c for c in gold if c.isalnum() or c in ".,/-")
    if gold_clean and gold_clean in pred_clean:
        return True
    nums_pred = re.findall(r"-?\d+\.?\d*", pred_clean)
    if gold_clean.replace(".", "").replace("-", "").replace("/", "").isdigit():
        for n in nums_pred:
            try:
                if float(n) == float(gold_clean) or n == gold_clean:
                    return True
            except ValueError:
                pass
    return False


def _greedy_decode(model, input_ids, max_new_tokens, eos_token_id):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        out = model(input_ids=generated)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == eos_token_id:
            break
    return generated


def _maybe_empty_cuda_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def compute_exact_match(model, tokenizer, math_path, device, context_window, max_new_tokens, max_samples):
    model.eval()
    correct = 0
    total = 0

    with open(math_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_samples:
        lines = lines[:max_samples]

    debug = os.environ.get("EVAL_DEBUG", "0").lower() in ("1", "true", "yes")
    debug_samples = []
    with torch.inference_mode():
        for line in tqdm(lines, desc="exact match"):
            _maybe_empty_cuda_cache(device)
            obj = json.loads(line)
            question = (obj.get("question") or "").strip()
            gold = _normalize(obj.get("final_answer") or "")
            if not question or not gold:
                continue
            prompt = f"Question: {question}\n\nFinal answer:"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_window)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            if hasattr(model, "generate"):
                gen_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                gen_ids = _greedy_decode(model, input_ids, max_new_tokens, tokenizer.eos_token_id)

            new_ids = gen_ids[0, input_ids.shape[1]:].cpu()
            predicted = _normalize(tokenizer.decode(new_ids, skip_special_tokens=True))
            if _match_answer(gold, predicted):
                correct += 1
            if debug and len(debug_samples) < 5:
                debug_samples.append((gold[:50], predicted[:100]))
            total += 1
    if debug and debug_samples:
        print("  [EVAL_DEBUG] sample (gold, predicted):")
        for g, p in debug_samples:
            print(f"    gold={g!r} pred={p!r}")
    return correct / total if total > 0 else 0.0


def main():
    from src.models import get_model

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root, "data")
    chunks_path = os.environ.get("GUTENBERG_CHUNKS_JSONL", os.path.join(data_dir, "gutenberg_7000_1192.jsonl"))
    math_path = os.environ.get("MATH_PREPROCESSED_JSONL", os.path.join(data_dir, "olympiad_preprocessed.jsonl"))
    model_names = os.environ.get("EVAL_MODELS", "gpt2").split(",")
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    results_path = os.environ.get("RESULTS_JSON") or os.path.join(_project_root, "eval_results.json")

    context_window = int(os.environ.get("CONTEXT_WINDOW", "1024"))
    max_target_tokens = int(os.environ.get("MAX_TARGET_TOKENS", "512"))

    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "256"))
    context_window_math = int(os.environ.get("CONTEXT_WINDOW_MATH", "512"))
    max_new_tokens_math = int(os.environ.get("MAX_NEW_TOKENS_MATH", "128"))
    max_lit_samples = int(os.environ.get("MAX_LIT_SAMPLES", "50"))
    max_math_samples = int(os.environ.get("MAX_MATH_SAMPLES", "100"))

    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    checkpoint_path = os.environ.get("EVAL_CHECKPOINT", "")
    for name in [n.strip() for n in model_names if n.strip()]:
        print(f"\n=== {name} ===")
        print(f"    context_window={context_window}  max_target_tokens={max_target_tokens}")
        print(f"    math: context={context_window_math}  max_new_tokens={max_new_tokens_math}")
        model, tokenizer = get_model(name, device=device)
        ckpt_path = checkpoint_path if name in ("hybrid", "selective", "mamba_selective", "gpt2_mamba_selective") else ""
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            print(f"    loaded checkpoint: {ckpt_path} (step {ckpt.get('step', '?')})")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        perplexity = compute_perplexity(
            model, tokenizer, chunks_path, device,
            context_window, max_target_tokens, max_lit_samples,
        )
        print(f"  perplexity  (literature): {perplexity:.2f}")

        _maybe_empty_cuda_cache(device)
        em = compute_exact_match(
            model, tokenizer, math_path, device,
            context_window_math, max_new_tokens_math, max_math_samples,
        )
        print(f"  exact match (math):       {em:.4f}")

        all_results[name] = {
            "perplexity_literature": round(perplexity, 4),
            "exact_match_math": round(em, 4),
            "context_window": context_window,
            "max_target_tokens": max_target_tokens,
            "context_window_math": context_window_math,
            "max_new_tokens_math": max_new_tokens_math,
            "lit_samples": max_lit_samples,
            "math_samples": max_math_samples,
        }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nresults saved to {results_path}")


if __name__ == "__main__":
    main()
