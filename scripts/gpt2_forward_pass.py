# load GPT-2 locally and run one forward pass with it
# we run this from the project root with 'python scripts/gpt2_forward_pass.py'
# this requires running the command 'pip install transformers torch'

import sys
import os

# we run from project root so src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import get_model

def main():
    model, tokenizer = get_model("gpt2", device="cpu")
    # tokenize the prompt to test it
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model(**inputs, labels=inputs["input_ids"])
    # check that the model is now running locally
    print(f"  Loss: {out.loss.item():.4f}")
    print("GPT-2 is now running locally")


if __name__ == "__main__":
    main()
