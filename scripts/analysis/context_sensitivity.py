import json
import math
import os
import sys

import torch
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _data_dir() -> str:
    from scripts.preprocessing.training_config import _project_root  # type: ignore
    return os.environ.get("DATA_DIR") or os.path.join(_project_root(), "data")


def _maybe_cuda() -> str:
    return os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def analyze_attention_gpt2() -> None:
    from src.models import get_model

    data_dir = _data_dir()
    chunks_path = os.path.join(data_dir, "gutenberg_7000_1192.jsonl")
    math_path = os.path.join(data_dir, "olympiad_preprocessed.jsonl")
    device = _maybe_cuda()

    model, tokenizer = get_model("gpt2", device=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    out_path = os.path.join(data_dir, "context_attention_gpt2.jsonl")
    max_lit = int(os.environ.get("ATTN_MAX_LIT", "10"))
    max_math = int(os.environ.get("ATTN_MAX_MATH", "20"))
    context_window = int(os.environ.get("CONTEXT_WINDOW", "1024"))

    with open(out_path, "w", encoding="utf-8") as fout:
        with open(chunks_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, line in enumerate(tqdm(lines[:max_lit], desc="attn_lit")):
            obj = json.loads(line)
            ctx = obj["context_text"]
            tgt = obj["target_text"]
            full = ctx + " " + tgt
            enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=context_window)
            input_ids = enc["input_ids"].to(device)
            with torch.inference_mode():
                out = model(input_ids=input_ids, output_attentions=True)
            attn = out.attentions[-1]  # type: ignore
            attn_mean = attn.mean(dim=1)[0]  # heads avg, shape (T,T)
            t = attn_mean.size(0) - 1
            vec = attn_mean[t].detach().cpu().tolist()
            rec = {
                "type": "literature",
                "index": i,
                "text": full,
                "attn_last_token": vec,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with open(math_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, line in enumerate(tqdm(lines[:max_math], desc="attn_math")):
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            prompt = f"Question: {q}\n\nFinal answer:"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_window)
            input_ids = enc["input_ids"].to(device)
            with torch.inference_mode():
                out = model(input_ids=input_ids, output_attentions=True)
            attn = out.attentions[-1]  # type: ignore
            attn_mean = attn.mean(dim=1)[0]
            t = attn_mean.size(0) - 1
            vec = attn_mean[t].detach().cpu().tolist()
            rec = {
                "type": "math",
                "index": i,
                "text": prompt,
                "attn_last_token": vec,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _segment_indices(length: int):
    if length <= 3:
        return [(0, length)]
    a = length // 3
    b = 2 * length // 3
    return [(0, a), (a, b), (b, length)]


def _mask_span(input_ids: torch.Tensor, span, pad_id: int) -> torch.Tensor:
    out = input_ids.clone()
    out[:, span[0]:span[1]] = pad_id
    return out


def analyze_mask_sensitivity(model_name: str) -> None:
    from src.models import get_model

    data_dir = _data_dir()
    math_path = os.path.join(data_dir, "olympiad_preprocessed.jsonl")
    device = _maybe_cuda()

    model, tokenizer = get_model(model_name, device=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    model.eval()

    out_path = os.path.join(data_dir, f"mask_sensitivity_{model_name}.jsonl")
    max_math = int(os.environ.get("MASK_MAX_MATH", "50"))
    context_window = int(os.environ.get("CONTEXT_WINDOW_MATH", "512"))

    with open(math_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    with open(out_path, "w", encoding="utf-8") as fout, torch.inference_mode():
        for i, line in enumerate(tqdm(lines[:max_math], desc=f"mask_{model_name}")):
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            gold = (obj.get("final_answer") or "").strip()
            if not q or not gold:
                continue
            prompt = f"Question: {q}\n\nFinal answer:{gold}"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_window)
            input_ids = enc["input_ids"].to(device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            base_out = model(input_ids=input_ids, labels=labels)
            base_loss = float(base_out.loss.detach().cpu())

            spans = _segment_indices(input_ids.size(1))
            losses = []
            for span in spans:
                masked_ids = _mask_span(input_ids, span, pad_id)
                out = model(input_ids=masked_ids, labels=labels)
                losses.append(float(out.loss.detach().cpu()))

            rec = {
                "index": i,
                "prompt": prompt,
                "base_loss": base_loss,
                "span_losses": losses,
                "seq_len": int(input_ids.size(1)),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if os.environ.get("RUN_ATTN_GPT2", "1").lower() in ("1", "true", "yes"):
        analyze_attention_gpt2()
    for name in ["gpt2", "mamba"]:
        if os.environ.get(f"RUN_MASK_{name.upper()}", "1").lower() in ("1", "true", "yes"):
            analyze_mask_sensitivity(name)


if __name__ == "__main__":
    main()

