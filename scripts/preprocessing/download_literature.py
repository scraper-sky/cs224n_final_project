"""this script downloads and saves high-quality samples from Gutenberg dataset"""
import json
import os
import random

def main():
    from datasets import load_dataset

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root, "data")
    out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset("BEE-spoke-data/gutenberg-en-v1-clean", split="train")
    filtered = dataset.filter(lambda x: x["score"] > 0.95)
    n_sample = min(500, len(filtered))
    if len(filtered) < 500:
        print(f"{len(filtered)} items have score > 0.95.")
    indices = list(range(len(filtered)))
    random.seed(42)
    chosen = random.sample(indices, n_sample)
    out_path = os.path.join(out_dir, "gutenberg_literature.jsonl")

    with open(out_path, "w") as f:
        for i in chosen:
            row = filtered[int(i)]
            obj = {
                "text": row["text"],
                "score": float(row["score"]),
                "word_count": int(row["word_count"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {n_sample} items to {out_path}")


if __name__ == "__main__":
    main()
