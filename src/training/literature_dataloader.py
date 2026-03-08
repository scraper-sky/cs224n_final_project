import json
import random

import torch


class LiteratureDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = (row.get("context_text", "") + row.get("target_text", "")).strip()
        if not text:
            ids = self.tokenizer.encode(" ", add_special_tokens=False)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(ids) <= self.max_length:
            segment = ids
        else:
            rng = random.Random(self.seed + idx)
            start = rng.randint(0, len(ids) - self.max_length)
            segment = ids[start : start + self.max_length]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        orig_len = len(segment)
        if orig_len < self.max_length:
            segment = segment + [pad_id] * (self.max_length - orig_len)
        attention_mask = [1] * orig_len + [0] * (self.max_length - orig_len)

        return {
            "input_ids": torch.tensor(segment, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def get_literature_dataloader(
    jsonl_path: str,
    tokenizer,
    batch_size: int = 2,
    max_length: int = 512,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    dataset = LiteratureDataset(jsonl_path, tokenizer, max_length=max_length, seed=seed)
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
