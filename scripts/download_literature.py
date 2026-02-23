"""
Download and prepare the Gutenberg literature dataset for the synthetic (perplexity) task.

Uses Hugging Face datasets; no manual download. Run from project root:
    python scripts/download_gutenberg.py

Requires: pip install datasets
"""

import json
import os
import random

def main():
    from datasets import load_dataset

    # Load dataset (use "train" to avoid loading all ~11k rows at once; we only need 500)
    print("Loading BEE-spoke-data/gutenberg-en-v1-clean (train split)...")
    dataset = load_dataset("BEE-spoke-data/gutenberg-en-v1-clean", split="train")

    # Filter: score > 0.95 as per project spec
    print("Filtering by score > 0.95...")
    filtered = dataset.filter(lambda x: x["score"] > 0.95)

    # Sample 500 with fixed seed for reproducibility
    n_sample = 500
    seed = 42
    if len(filtered) < n_sample:
        n_sample = len(filtered)
        print(f"Only {len(filtered)} items have score > 0.95; using all.")
    indices = list(range(len(filtered)))
    random.seed(seed)
    chosen = random.sample(indices, n_sample)

    # Save to data/gutenberg_literature.jsonl (one JSON object per line)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gutenberg_literature.jsonl")

    with open(out_path, "w") as f:
        for i in chosen:
            row = filtered[int(i)]
            # Store text and metadata; preprocessing (strip headers, fix line breaks) can be done later
            obj = {
                "text": row["text"],
                "score": float(row["score"]),
                "word_count": int(row["word_count"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {n_sample} items to {out_path}")
    print("Next: run preprocessing to strip Gutenberg headers/footers and normalize line breaks.")


if __name__ == "__main__":
    main()
