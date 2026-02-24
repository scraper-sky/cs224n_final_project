# Train loop: load model from registry, next-token prediction on preprocessed math JSONL, save checkpoints.

import os
from typing import Any, Optional

import torch

from src.models import get_model


def run_train(config: Optional[dict[str, Any]] = None, config_overrides: Optional[dict[str, Any]] = None):
    from .training_config import get_config
    from .math_dataloader import get_math_dataloader

    cfg = get_config(config_overrides or config)
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["seed"])

    model, tokenizer = get_model(cfg["model_name"], device=cfg["device"])
    dataloader = get_math_dataloader(
        cfg["math_preprocessed_jsonl"],
        tokenizer,
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
        solution_only=cfg["train_on_solution_only"],
        seed=cfg["seed"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    step = 0
    model.train()

    while step < cfg["max_steps"]:
        for batch in dataloader:
            if step >= cfg["max_steps"]:
                break
            input_ids = batch["input_ids"].to(cfg["device"])
            attention_mask = batch["attention_mask"].to(cfg["device"])
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            step += 1

            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f}")

            if step % cfg["save_every"] == 0 and step > 0:
                ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{cfg['model_name']}_step{step}.pt")
                torch.save({"step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
                print(f"saved {ckpt_path}")

    print("training finished")

