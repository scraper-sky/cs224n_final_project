# load preprocessed math JSONL from scripts/preprocessing/preprocess_math.py, tokenize, batch

import json
import torch


def _format_record(record: dict, solution_only: bool, append_answer: bool = True, direct_qa: bool = False) -> str:
    question = (record.get("question") or "").strip()
    solution = (record.get("solution") or "").strip()
    answer = (record.get("final_answer") or "").strip()
    if direct_qa and question and answer:
        return f"Question: {question}\n\nFinal answer: {answer}"
    if solution_only:
        base = solution or question
    else:
        base = f"{question}\n\n{solution}".strip() or ""
    if append_answer and base and answer:
        base = f"{base}\n\nFinal answer: {answer}"
    return base or ""


class MathDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512, solution_only: bool = False, append_answer: bool = True, direct_qa: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.solution_only = solution_only
        self.append_answer = append_answer
        self.direct_qa = direct_qa
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = _format_record(record, solution_only, append_answer, direct_qa)
                if not text:
                    continue
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                self.examples.append({
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_math_dataloader(jsonl_path: str, tokenizer, batch_size: int = 2, max_length: int = 512, solution_only: bool = False, append_answer: bool = True, direct_qa: bool = False, seed: int = 42):
    dataset = MathDataset(jsonl_path, tokenizer, max_length=max_length, solution_only=solution_only, append_answer=append_answer, direct_qa=direct_qa)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        },
    )
