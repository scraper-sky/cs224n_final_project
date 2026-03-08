import json
import os
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _mock_lit_jsonl(path: str):
    base = "The quick brown fox jumps. " * 100
    with open(path, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"context_text": base[:2000], "target_text": base[2000:4000]}, ensure_ascii=False) + "\n")


def _mock_math_jsonl(path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"question": f"Q{i}", "solution": "", "final_answer": "2"}, ensure_ascii=False) + "\n")


def test_literature_dataset():
    from src.models import get_tokenizer
    from src.training.literature_dataloader import LiteratureDataset

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        _mock_lit_jsonl(f.name)
        path = f.name
    try:
        tok = get_tokenizer("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        ds = LiteratureDataset(path, tok, max_length=128, seed=42)
        assert len(ds) == 3
        item = ds[0]
        assert item["input_ids"].shape == (128,)
        assert item["attention_mask"].shape == (128,)
        assert item["input_ids"].dtype == torch.long
    finally:
        os.unlink(path)


def test_literature_dataloader_batch():
    from src.models import get_tokenizer
    from src.training.literature_dataloader import get_literature_dataloader

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        _mock_lit_jsonl(f.name)
        path = f.name
    try:
        tok = get_tokenizer("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        dl = get_literature_dataloader(path, tok, batch_size=2, max_length=128, seed=42)
        batch = next(iter(dl))
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] == 128
    finally:
        os.unlink(path)


def test_mixed_batch_iterator():
    from src.models import get_tokenizer
    from src.training.math_dataloader import get_math_dataloader
    from src.training.literature_dataloader import get_literature_dataloader
    from src.training.train_loop import _mixed_batch_iterator

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fm:
        _mock_math_jsonl(fm.name)
        math_path = fm.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fl:
        _mock_lit_jsonl(fl.name)
        lit_path = fl.name
    try:
        tok = get_tokenizer("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        math_dl = get_math_dataloader(math_path, tok, batch_size=2, max_length=128, direct_qa=True, seed=42)
        lit_dl = get_literature_dataloader(lit_path, tok, batch_size=2, max_length=128, seed=42)

        it = _mixed_batch_iterator(math_dl, lit_dl, literature_ratio=0.5, seed=42)
        batches = [next(it) for _ in range(10)]
        assert all(b["input_ids"].dim() == 2 for b in batches)
        assert all(b["input_ids"].shape[1] == 128 for b in batches)
    finally:
        os.unlink(math_path)
        os.unlink(lit_path)


if __name__ == "__main__":
    test_literature_dataset()
    test_literature_dataloader_batch()
    test_mixed_batch_iterator()
    print("All tests passed.")
