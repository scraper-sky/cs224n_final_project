import json
import os
from typing import Any

from datasets import load_dataset


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def main() -> None:
    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)

    hf_id: str = _get_env("AMC23_DATASET", "math-ai/amc23")  # type: ignore[assignment]
    split: str = _get_env("AMC23_SPLIT", "test") or "test"  # type: ignore[assignment]
    question_key: str = _get_env("AMC23_QUESTION_KEY", "problem") or "problem"  # type: ignore[assignment]
    solution_key: str | None = _get_env("AMC23_SOLUTION_KEY", None)
    answer_key: str = _get_env("AMC23_ANSWER_KEY", "answer") or "answer"  # type: ignore[assignment]

    ds = load_dataset(hf_id, split=split)

    records: list[dict[str, Any]] = []
    for row in ds:
        raw_q = row.get(question_key)
        q = (raw_q or "").strip()
        if not q:
            q = str(row).strip()
        sol = (row.get(solution_key) or "").strip() if solution_key else ""
        ans = (row.get(answer_key) or "").strip()
        rec = {
            "question": q,
            "solution": sol,
            "final_answer": ans,
        }
        records.append(rec)

    import random

    random.seed(42)
    random.shuffle(records)
    n_train = int(0.75 * len(records))
    train, test = records[:n_train], records[n_train:]

    train_path = os.path.join(data_dir, "amc23_train_preprocessed.jsonl")
    test_path = os.path.join(data_dir, "amc23_test_preprocessed.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for rec in train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(train)} train records to {train_path}")
    print(f"wrote {len(test)} test records to {test_path}")


if __name__ == "__main__":
    main()

