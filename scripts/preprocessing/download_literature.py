
# Download and prepare the Gutenberg literature dataset for the synthetic (perplexity) task.

# Uses Hugging Face datasets; no manual download. Run from project root:
#   python scripts/download_gutenberg.py

# Requires: pip install datasets


import json
import os
import random

def main():
    from datasets import load_dataset

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.environ.get("DATA_DIR", os.path.abspath(os.path.join(script_dir, "..", "..", "data")))
    out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load BEE-spoke-data/gutenberg-en-v1-clean (train split)
    dataset = load_dataset("BEE-spoke-data/gutenberg-en-v1-clean", split="train")
    # Filter by score > 0.95
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

    # Verify that the data is written correctly
    print(f"Wrote {n_sample} items to {out_path}")


if __name__ == "__main__":
    main()
