# here we preprocess the literature data for the perplexity evaluation
# we split each book into context (7000 tokens) and target (1192 tokens)
# we read the gutenberg_literature.jsonl file and write the gutenberg_7000_1192.jsonl file
# we run this from the project root with 'python scripts/preprocess_literature.py'
# this requires running the command 'pip install datasets'
# this requires gutenberg_literature.jsonl from scripts/downloa_literature.py 
import json
import os
import sys

# here we set the paths for the input and output files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# here we set the context and target lengths
# the context length is what we feed to the model as context
# the target length is what the length of the token sequence that the model predicts
CONTEXT_LEN = 7000
TARGET_LEN = 1192
TOTAL_LEN = CONTEXT_LEN + TARGET_LEN


def main():
    from src.models import get_tokenizer

    data_dir = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
    in_path = os.environ.get("GUTENBERG_JSONL", os.path.join(data_dir, "gutenberg_literature.jsonl"))
    out_path = os.environ.get("GUTENBERG_CHUNKS_JSONL", os.path.join(data_dir, "gutenberg_7000_1192.jsonl"))

    tokenizer = get_tokenizer("gpt2")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    written = 0
    skipped = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                skipped += 1
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < TOTAL_LEN:
                skipped += 1
                continue
            context_ids = ids[:CONTEXT_LEN]
            target_ids = ids[CONTEXT_LEN:TOTAL_LEN]
            context_text = tokenizer.decode(context_ids, skip_special_tokens=True)
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            row = {
                "index": i,
                "context_text": context_text,
                "target_text": target_text,
                "context_len": CONTEXT_LEN,
                "target_len": TARGET_LEN,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} chunks to {out_path} (skipped {skipped} items with < {TOTAL_LEN} tokens)")


if __name__ == "__main__":
    main()
