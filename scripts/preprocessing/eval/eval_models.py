import json
import math
import os
import re
import sys
import random

import torch
import torch.nn.functional as F
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
    if max_samples is not None:
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


def _token_logprob_sum(model, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Returns sum of log p(label_t | prefix) over non-masked labels.
    Expects input_ids shape [1, T], labels shape [1, T] with -100 mask.
    """
    out = model(input_ids=input_ids)
    logits = out.logits  # [1, T, V]
    # Shift so logits[t] predicts labels[t+1]
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels.ne(-100)
    if mask.sum().item() == 0:
        return float("-inf")
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return float((token_lp * mask).sum().item())


def compute_next_sentence_preference(
    model,
    tokenizer,
    chunks_path: str,
    device,
    context_window: int,
    max_target_tokens: int,
    max_samples: int | None,
    seed: int = 42,
) -> float:
    """
    For each sample, compares the log-likelihood of the true continuation vs
    a randomly-chosen continuation from a different sample.

    Returns preference accuracy in [0, 1].
    """
    model.eval()
    with open(chunks_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_samples is not None:
        lines = lines[:max_samples]
    if len(lines) < 2:
        return 0.0

    objs = [json.loads(l) for l in lines]
    targets = [(o.get("target_text") or "") for o in objs]

    rng = random.Random(seed)
    correct = 0
    total = 0

    with torch.inference_mode():
        for i, obj in enumerate(tqdm(objs, desc="next-sentence pref")):
            ctx = (obj.get("context_text") or "").strip()
            true_tgt = (obj.get("target_text") or "").strip()
            if not ctx or not true_tgt:
                continue

            # pick a negative from a different sample
            j = i
            while j == i:
                j = rng.randrange(0, len(targets))
            neg_tgt = (targets[j] or "").strip()
            if not neg_tgt:
                continue

            def score(tgt: str) -> float:
                ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
                tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)[:max_target_tokens]
                ctx_budget = context_window - len(tgt_ids)
                ctx_ids2 = ctx_ids[-ctx_budget:] if ctx_budget > 0 else []
                all_ids = ctx_ids2 + tgt_ids
                input_ids = torch.tensor([all_ids], dtype=torch.long).to(device)
                labels = torch.full_like(input_ids, -100)
                labels[0, len(ctx_ids2):] = input_ids[0, len(ctx_ids2):]
                return _token_logprob_sum(model, input_ids, labels)

            lp_true = score(true_tgt)
            lp_neg = score(neg_tgt)
            if lp_true > lp_neg:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


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
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def compute_exact_match(model, tokenizer, math_path, device, context_window, max_new_tokens, max_samples):
    model.eval()
    correct = 0
    total = 0

    with open(math_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_samples is not None:
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
    secondary_math_path = os.environ.get("SECONDARY_MATH_PREPROCESSED_JSONL", os.path.join(data_dir, "secondary_math_preprocessed.jsonl"))
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
    max_secondary_samples = int(os.environ.get("MAX_SECONDARY_SAMPLES", "100"))

    next_sent_enabled = os.environ.get("NEXT_SENT_TEST", "0").lower() in ("1", "true", "yes")
    next_sent_only = os.environ.get("NEXT_SENT_ONLY", "0").lower() in ("1", "true", "yes")
    next_sent_samples = os.environ.get("NEXT_SENT_SAMPLES")
    next_sent_samples_i = int(next_sent_samples) if next_sent_samples is not None else max_lit_samples
    next_sent_seed = int(os.environ.get("NEXT_SENT_SEED", "42"))
    next_sent_max_tgt = int(os.environ.get("NEXT_SENT_MAX_TARGET_TOKENS", str(max_target_tokens)))

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
        ckpt_path = checkpoint_path if name in ("hybrid", "selective", "mamba_selective", "gpt2_mamba_selective", "gpt2", "mamba") else ""
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            print(f"    loaded checkpoint: {ckpt_path} (step {ckpt.get('step', '?')})")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        next_sent_acc = None
        if next_sent_enabled:
            next_sent_acc = compute_next_sentence_preference(
                model,
                tokenizer,
                chunks_path,
                device,
                context_window=context_window,
                max_target_tokens=next_sent_max_tgt,
                max_samples=next_sent_samples_i,
                seed=next_sent_seed,
            )
            print(f"  next-sentence pref (literature): {next_sent_acc:.4f}")
            if next_sent_only:
                all_results[name] = {
                    "next_sentence_pref_literature": round(next_sent_acc, 4),
                    "next_sent_samples": next_sent_samples_i,
                    "next_sent_seed": next_sent_seed,
                    "context_window": context_window,
                    "max_target_tokens": next_sent_max_tgt,
                }
                continue

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
        em_secondary = None
        if os.path.isfile(secondary_math_path):
            em_secondary = compute_exact_match(
                model, tokenizer, secondary_math_path, device,
                context_window_math, max_new_tokens_math, max_secondary_samples,
            )
            print(f"  exact match (secondary): {em_secondary:.4f}")

        all_results[name] = {
            "perplexity_literature": round(perplexity, 4),
            "next_sentence_pref_literature": round(next_sent_acc, 4) if next_sent_acc is not None else None,
            "exact_match_math": round(em, 4),
            "exact_match_secondary": round(em_secondary, 4) if em_secondary is not None else None,
            "context_window": context_window,
            "max_target_tokens": max_target_tokens,
            "context_window_math": context_window_math,
            "max_new_tokens_math": max_new_tokens_math,
            "lit_samples": max_lit_samples,
            "math_samples": max_math_samples,
            "secondary_samples": max_secondary_samples,
        }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nresults saved to {results_path}")


if __name__ == "__main__":
    main()
