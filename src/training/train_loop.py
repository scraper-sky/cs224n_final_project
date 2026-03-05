# Train loop: load model from registry, next-token prediction on preprocessed math JSONL, save checkpoints.
#
# Key additions over the original loop:
#   - Gradient clipping (max_grad_norm): prevents exploding gradients during the early
#     steps of HMT fine-tuning when the randomly-initialised Mamba blocks produce large
#     losses (perplexity > 10^6 before any gradient updates).
#   - Linear LR warmup (warmup_steps): ramps the learning rate from 0 to cfg["lr"] over
#     the first N steps.  Combined with gradient clipping this keeps the initial updates
#     small enough that the Mamba blocks adapt smoothly without destabilising the frozen
#     GPT-2 attention blocks.
#   - freeze_gpt2 flag (via model kwargs): forwarded to load_hybrid so only Mamba block
#     parameters are updated during Phase 1 training.

import os
from typing import Any, Optional

import torch

from src.models import get_model


def _get_lr_scale(step: int, warmup_steps: int) -> float:
    """Linear warmup: returns a scale in [0, 1] that reaches 1.0 at warmup_steps."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def run_train(config: Optional[dict[str, Any]] = None, config_overrides: Optional[dict[str, Any]] = None):
    from .training_config import get_config
    from .math_dataloader import get_math_dataloader

    cfg = get_config(config_overrides or config)
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["seed"])

    # Pass freeze_gpt2 into load_hybrid (ignored by gpt2/mamba loaders via **kwargs pop).
    model_kwargs: dict[str, Any] = {}
    if cfg.get("freeze_gpt2"):
        model_kwargs["freeze_gpt2"] = True

    model, tokenizer = get_model(cfg["model_name"], device=cfg["device"], **model_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataloader = get_math_dataloader(
        cfg["math_preprocessed_jsonl"],
        tokenizer,
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
        solution_only=cfg["train_on_solution_only"],
        seed=cfg["seed"],
    )

    # Only optimize parameters that require gradients (respects freeze_gpt2)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"])

    warmup_steps = cfg["warmup_steps"]
    max_grad_norm = cfg["max_grad_norm"]

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

            # Apply linear LR warmup (scale base learning rate each step)
            lr_scale = _get_lr_scale(step, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg["lr"] * lr_scale

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()

            # Gradient clipping for starting from a high-loss initialisation
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            optimizer.step()
            step += 1

            if step % 50 == 0:
                print(f"step {step:>5}  loss {loss.item():.4f}  lr {cfg['lr'] * lr_scale:.2e}")

            if step % cfg["save_every"] == 0 and step > 0:
                ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{cfg['model_name']}_step{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                print(f"saved {ckpt_path}")

    print("training finished")

