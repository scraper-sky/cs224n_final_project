import json
import os


def main():
    from datasets import load_dataset

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.environ.get("DATA_DIR", os.path.abspath(os.path.join(script_dir, "..", "..", "data")))
    out_path = os.environ.get("MATH_OLYMPIAD_JSON", os.path.join(data_dir, "math_olympiad_questions.json"))
    os.makedirs(data_dir, exist_ok=True)

    configs = ["OE_TO_maths_en_COMP", "TP_TO_maths_en_COMP"]
    all_rows = []
    for cfg in configs:
        ds = load_dataset("Hothan/OlympiadBench", cfg, split="train", trust_remote_code=True)
        all_rows.extend(list(ds))
    dataset = all_rows
    items = []
    for row in dataset:
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
