"""this tests balanced math/literature training"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
os.makedirs(DATA_DIR, exist_ok=True)


def create_mock_math_jsonl(path: str, num_rows: int = 10):
    text = "What is 2 + 2? Final answer: 4"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_rows):
            f.write(json.dumps({"question": f"Q{i}: What is 1+1?", "solution": "", "final_answer": "2"}, ensure_ascii=False) + "\n")


def create_mock_literature_jsonl(path: str, num_rows: int = 5):
    base = "The quick brown fox jumps over the lazy dog. " * 200
    ctx = base[: 3000]
    tgt = base[3000: 6000]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_rows):
            f.write(json.dumps({"context_text": ctx, "target_text": tgt}, ensure_ascii=False) + "\n")


def main():
    math_path = os.path.join(DATA_DIR, "olympiad_preprocessed.jsonl")
    lit_path = os.path.join(DATA_DIR, "gutenberg_7000_1192.jsonl")
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "test")
    os.makedirs(ckpt_dir, exist_ok=True)

    create_mock_math_jsonl(math_path)
    create_mock_literature_jsonl(lit_path)

    from src.training.run_training import run_train

    run_train(config_overrides={
        "math_preprocessed_jsonl": math_path,
        "literature_jsonl": lit_path,
        "checkpoint_dir": ckpt_dir,
        "model_name": "hybrid",
        "freeze_gpt2": True,
        "balanced_training": True,
        "literature_ratio": 0.5,
        "direct_qa": True,
        "max_steps": 5,
        "save_every": 10,
        "batch_size": 2,
        "warmup_steps": 2,
        "device": "cpu",
    })
    print("\n=== test_balanced_training PASSED ===")


if __name__ == "__main__":
    main()
