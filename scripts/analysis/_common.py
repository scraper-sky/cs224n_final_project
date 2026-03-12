"""this is a shared helpers for loading models, building prompts, and I/O for our analysis scripts"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model


CHECKPOINTED_MODEL_NAMES = {
    "hybrid",
    "hybrid_llm",
    "selective",
    "mamba_selective",
    "gpt2_mamba_selective",
    "hybrid_model_v2",
    "gpt2_mamba_selective_v2",
    "hybrid2",
}


def default_device() -> str:
    return os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def default_data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_checkpoint_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in parse_csv_list(raw):
        if "=" not in item:
            continue
        model_name, path = item.split("=", 1)
        if model_name.strip() and path.strip():
            mapping[model_name.strip()] = path.strip()
    return mapping


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_empty_cuda_cache(device: str | torch.device) -> None:
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def normalize_answer(text: str) -> str:
    return (text or "").strip().lower()


def match_answer(gold: str, predicted: str) -> bool:
    gold = normalize_answer(gold)
    predicted = normalize_answer(predicted)
    if not gold or not predicted:
        return False
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


def build_math_prompt(record: dict[str, Any]) -> str:
    question = (record.get("question") or "").strip()
    return f"Question: {question}\n\nFinal answer:"


def get_record_id(record: dict[str, Any], fallback_index: int) -> str:
    dataset = (record.get("dataset") or "math").strip()
    rec_id = str(record.get("id", "")).strip() or str(fallback_index)
    return f"{dataset}:{rec_id}"


def load_model_and_tokenizer(
    model_name: str,
    *,
    device: str | None = None,
    checkpoint_path: str = "",
    checkpoint_map: dict[str, str] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any, str]:
    device = device or default_device()
    model_kwargs = dict(model_kwargs or {})
    model, tokenizer = get_model(model_name, device=device, **model_kwargs)

    ckpt_path = ""
    if checkpoint_map:
        ckpt_path = checkpoint_map.get(model_name, "")
    if not ckpt_path and model_name in CHECKPOINTED_MODEL_NAMES:
        ckpt_path = checkpoint_path
    if ckpt_path and os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, ckpt_path


def greedy_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        out = model(input_ids=generated)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return generated


def generate_prediction(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
    context_window: int,
    max_new_tokens: int,
) -> tuple[str, list[int], list[int]]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_window)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        if hasattr(model, "generate"):
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            generated = greedy_generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
    new_ids = generated[0, input_ids.shape[1]:].detach().cpu().tolist()
    prompt_ids = input_ids[0].detach().cpu().tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, prompt_ids, new_ids


def truncate_prompt_by_tokens(tokenizer: Any, prompt: str, max_prompt_tokens: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_prompt_tokens:
        return prompt
    truncated_ids = token_ids[-max_prompt_tokens:]
    return tokenizer.decode(truncated_ids, skip_special_tokens=False)


def build_teacher_forced_batch(
    tokenizer: Any,
    prompt: str,
    target_text: str,
    *,
    device: str,
    context_window: int,
    max_target_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    separator = "" if prompt.endswith((" ", "\n")) else " "
    target_ids = tokenizer.encode(separator + target_text, add_special_tokens=False)[:max_target_tokens]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    available_prompt = max(0, context_window - len(target_ids))
    prompt_ids = prompt_ids[-available_prompt:] if available_prompt else []
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prompt_ids):] = input_ids[0, len(prompt_ids):]
    return input_ids, attention_mask, labels, len(prompt_ids)


def compute_teacher_forced_loss(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_text: str,
    *,
    device: str,
    context_window: int,
    max_target_tokens: int = 64,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    batch = build_teacher_forced_batch(
        tokenizer,
        prompt,
        target_text,
        device=device,
        context_window=context_window,
        max_target_tokens=max_target_tokens,
    )
    input_ids, attention_mask, labels, prompt_len = batch
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss.detach().cpu()), input_ids, attention_mask, labels, prompt_len


def chunk_token_spans(length: int, num_spans: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    num_spans = max(1, min(num_spans, length))
    spans: list[tuple[int, int]] = []
    for i in range(num_spans):
        start = math.floor(i * length / num_spans)
        end = math.floor((i + 1) * length / num_spans)
        if start < end:
            spans.append((start, end))
    return spans


def replace_token_span(
    input_ids: torch.Tensor,
    span: tuple[int, int],
    replacement_id: int,
) -> torch.Tensor:
    out = input_ids.clone()
    out[:, span[0]:span[1]] = replacement_id
    return out


def decode_token_pieces(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in token_ids]


def is_numeric_token_piece(piece: str) -> bool:
    return bool(re.search(r"\d", piece))


def seed_everything(seed: int) -> random.Random:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def maybe_save_plot(fig: Any, out_path: str | Path) -> None:
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass


def iter_checkpoint_files(pattern: str) -> Iterator[Path]:
    parent = Path(pattern).parent
    glob_pat = Path(pattern).name
    for path in sorted(parent.glob(glob_pat)):
        if path.is_file():
            yield path
