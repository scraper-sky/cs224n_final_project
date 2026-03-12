"""this script downloads GSM8K and AMC23 datasets"""
import json
import os
from typing import Any, Dict, List


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_hf_dataset(hf_id: str, split: str, config: str | None = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    if config:
        ds = load_dataset(hf_id, config, split=split)
    else:
        ds = load_dataset(hf_id, split=split)
    return list(ds)


def main() -> None:
    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "secondary_math_raw.jsonl")

    DATASETS: List[Dict[str, Any]] = [
        {
            "name": "gsm8k",
            "hf_id": os.environ.get("GSM8K_DATASET", "openai/gsm8k"),
            "config": os.environ.get("GSM8K_CONFIG", "main"),
            "split": os.environ.get("GSM8K_SPLIT", "test"),
            "question_key": "question",
            "solution_key": "answer",
            "final_answer_key": None,
            "id_key": "id",
        },
        {
            "name": "math500",
            "hf_id": os.environ.get("MATH500_DATASET", ""),
            "config": os.environ.get("MATH500_CONFIG", ""),
            "split": os.environ.get("MATH500_SPLIT", "test"),
            "question_key": os.environ.get("MATH500_QUESTION_KEY", "problem"),
            "solution_key": os.environ.get("MATH500_SOLUTION_KEY", "solution"),
            "final_answer_key": os.environ.get("MATH500_ANSWER_KEY", "answer"),
            "id_key": os.environ.get("MATH500_ID_KEY", "id"),
        },
        {
            "name": "aime24",
            "hf_id": os.environ.get("AIME24_DATASET", ""),
            "config": os.environ.get("AIME24_CONFIG", ""),
            "split": os.environ.get("AIME24_SPLIT", "test"),
            "question_key": os.environ.get("AIME24_QUESTION_KEY", "problem"),
            "solution_key": os.environ.get("AIME24_SOLUTION_KEY", "solution"),
            "final_answer_key": os.environ.get("AIME24_ANSWER_KEY", "answer"),
            "id_key": os.environ.get("AIME24_ID_KEY", "id"),
        },
        {
            "name": "aime25",
            "hf_id": os.environ.get("AIME25_DATASET", ""),
            "config": os.environ.get("AIME25_CONFIG", ""),
            "split": os.environ.get("AIME25_SPLIT", "test"),
            "question_key": os.environ.get("AIME25_QUESTION_KEY", "problem"),
            "solution_key": os.environ.get("AIME25_SOLUTION_KEY", "solution"),
            "final_answer_key": os.environ.get("AIME25_ANSWER_KEY", "answer"),
            "id_key": os.environ.get("AIME25_ID_KEY", "id"),
        },
        {
            "name": "aime26",
            "hf_id": os.environ.get("AIME26_DATASET", ""),
            "config": os.environ.get("AIME26_CONFIG", ""),
            "split": os.environ.get("AIME26_SPLIT", "test"),
            "question_key": os.environ.get("AIME26_QUESTION_KEY", "problem"),
            "solution_key": os.environ.get("AIME26_SOLUTION_KEY", "solution"),
            "final_answer_key": os.environ.get("AIME26_ANSWER_KEY", "answer"),
            "id_key": os.environ.get("AIME26_ID_KEY", "id"),
        },
        {
            "name": "amc23",
            "hf_id": os.environ.get("AMC23_DATASET", ""),
            "config": os.environ.get("AMC23_CONFIG", ""),
            "split": os.environ.get("AMC23_SPLIT", "test"),
            "question_key": os.environ.get("AMC23_QUESTION_KEY", "problem"),
            "solution_key": os.environ.get("AMC23_SOLUTION_KEY", "solution"),
            "final_answer_key": os.environ.get("AMC23_ANSWER_KEY", "answer"),
            "id_key": os.environ.get("AMC23_ID_KEY", "id"),
        },
    ]

    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for spec in DATASETS:
            hf_id = spec["hf_id"]
            if not hf_id:
                continue
            rows = _load_hf_dataset(hf_id, spec["split"], spec.get("config"))
            for row in rows:
                q = (row.get(spec["question_key"]) or "").strip()
                sol = (row.get(spec["solution_key"]) or "").strip()
                ans = None
                if spec["final_answer_key"] is not None:
                    ans = (row.get(spec["final_answer_key"]) or "").strip()
                rec = {
                    "dataset": spec["name"],
                    "id": str(row.get(spec["id_key"], "")),
                    "question": q,
                    "solution": sol,
                    "final_answer": ans,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} records to {out_path}")


if __name__ == "__main__":
    main()

