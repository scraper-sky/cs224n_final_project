import json
import os
import re


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _normalize_answer(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_final_from_gsm8k(solution: str) -> str:
    if not solution:
        return ""
    nums = re.findall(r"-?\d+\.?\d*", solution)
    return nums[-1] if nums else ""


def main() -> None:
    from datasets import load_dataset

    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "gsm8k_train_preprocessed.jsonl")

    ds = load_dataset("openai/gsm8k", "main", split="train")
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = (row.get("question") or "").strip()
            solution = (row.get("answer") or "").strip()
            if not question:
                continue
            final_answer = _normalize_answer(_extract_final_from_gsm8k(solution))
            rec = {
                "question": question,
                "solution": solution,
                "final_answer": final_answer,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} records to {out_path}")


if __name__ == "__main__":
    main()
