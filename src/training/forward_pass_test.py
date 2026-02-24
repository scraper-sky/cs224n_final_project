#sanity check where we load GPT-2 and run one forward pass
# here we use: python -m src.training.forward_pass_test
from src.models import get_model


def main():
    model, tokenizer = get_model("gpt2", device="cpu")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model(**inputs, labels=inputs["input_ids"])
    print(f"  Loss: {out.loss.item():.4f}")
    print("GPT-2 is now running locally")


if __name__ == "__main__":
    main()
