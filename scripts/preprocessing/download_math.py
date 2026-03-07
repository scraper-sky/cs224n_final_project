import json
import os


def _load_olympiad_via_parquet(configs):
    """Load OlympiadBench from parquet URLs (bypasses broken dataset card)."""
    from datasets import load_dataset

    base = "https://huggingface.co/datasets/Hothan/OlympiadBench/resolve/main/OlympiadBench"
    all_rows = []
    for cfg in configs:
        url = f"{base}/{cfg}/{cfg}.parquet"
        ds = load_dataset("parquet", data_files={"train": url}, split="train")
        all_rows.extend(list(ds))
    return all_rows


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root, "data")
    out_path = os.environ.get("MATH_OLYMPIAD_JSON", os.path.join(data_dir, "math_olympiad_questions.json"))
    os.makedirs(data_dir, exist_ok=True)

    configs = ["OE_TO_maths_en_COMP", "TP_TO_maths_en_COMP"]

    try:
        all_rows = _load_olympiad_via_parquet(configs)
    except Exception as e:
        raise RuntimeError(f"Failed to load OlympiadBench: {e}") from e

    items = []
    for row in all_rows:
        item = {
            "id": row.get("id"),
            "question": row.get("question") or "",
            "solution": row.get("solution"),
            "final_answer": row.get("final_answer"),
            "subfield": row.get("subfield") or "",
        }
        items.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()
