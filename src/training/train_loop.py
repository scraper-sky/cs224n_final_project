import itertools
import os
import random
from typing import Any, Iterator, Optional

import torch

from src.models import get_model


def _get_lr_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def _mixed_batch_iterator(
    math_dl: torch.utils.data.DataLoader,
    lit_dl: torch.utils.data.DataLoader,
    literature_ratio: float,
    seed: int,
) -> Iterator[dict]:
    rng = random.Random(seed)
    math_iter: Optional[Iterator] = None
    lit_iter: Optional[Iterator] = None

    def _next_math() -> dict:
        nonlocal math_iter
        if math_iter is None:
            math_iter = iter(math_dl)
        try:
            return next(math_iter)
        except StopIteration:
            math_iter = iter(math_dl)
            return next(math_iter)

    def _next_lit() -> dict:
        nonlocal lit_iter
        if lit_iter is None:
            lit_iter = iter(lit_dl)
        try:
            return next(lit_iter)
        except StopIteration:
            lit_iter = iter(lit_dl)
            return next(lit_iter)

    while True:
        if rng.random() < literature_ratio:
            yield _next_lit()
        else:
            yield _next_math()


def run_train(config: Optional[dict[str, Any]] = None, config_overrides: Optional[dict[str, Any]] = None):
    from .training_config import get_config
    from .math_dataloader import get_math_dataloader
    from .literature_dataloader import get_literature_dataloader

    cfg = get_config(config_overrides or config)
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg["seed"])

    model_kwargs: dict[str, Any] = {}
    if cfg.get("freeze_gpt2") and cfg["model_name"] in ("hybrid", "gpt2_mamba_selective"):
        model_kwargs["freeze_gpt2"] = True

    model, tokenizer = get_model(cfg["model_name"], device=cfg["device"], **model_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    math_dataloader = get_math_dataloader(
        cfg["math_preprocessed_jsonl"],
        tokenizer,
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
        solution_only=cfg["train_on_solution_only"],
        append_answer=cfg.get("append_answer", True),
        direct_qa=cfg.get("direct_qa", False),
        seed=cfg["seed"],
    )

    if cfg.get("balanced_training"):
        lit_path = cfg.get("literature_jsonl")
        if not os.path.isfile(lit_path):
            raise FileNotFoundError(
                f"Balanced training requires literature data at {lit_path}. "
                "Run scripts/preprocessing/preprocess_literature.py and download_literature.py first."
            )
        lit_dataloader = get_literature_dataloader(
            lit_path,
            tokenizer,
            batch_size=cfg["batch_size"],
            max_length=cfg["max_length"],
            seed=cfg["seed"],
        )
        batch_iter = _mixed_batch_iterator(
            math_dataloader,
            lit_dataloader,
            literature_ratio=cfg.get("literature_ratio", 0.5),
            seed=cfg["seed"],
        )
    else:
        batch_iter = itertools.cycle(math_dataloader)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"])

    warmup_steps = cfg["warmup_steps"]
    max_grad_norm = cfg["max_grad_norm"]

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    step = 0
    consecutive_nan = 0
    max_consecutive_nan = 20
    model.train()

    while step < cfg["max_steps"]:
        batch = next(batch_iter)
        input_ids = batch["input_ids"].to(cfg["device"])
        attention_mask = batch["attention_mask"].to(cfg["device"])
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        lr_scale = _get_lr_scale(step, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg["lr"] * lr_scale

        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss

        if torch.isnan(loss) or torch.isinf(loss):
            consecutive_nan += 1
            step += 1
            if consecutive_nan <= 5 or consecutive_nan % 10 == 0:
                print(f"WARNING: NaN loss at step {step}, skipping")
            if consecutive_nan >= max_consecutive_nan:
                raise RuntimeError(
                    f"NaN loss {consecutive_nan} steps in a row. Check model init and data."
                )
            continue

        consecutive_nan = 0

        gate_reg = cfg.get("gate_reg", 0.0)
        if gate_reg > 0 and hasattr(model, "mamba_gate"):
            g = torch.sigmoid(model.mamba_gate)
            loss = loss + gate_reg * (g**2).sum()

        loss.backward()

        for param in trainable_params:
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                param.grad.zero_()

        if max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            if step % 50 == 0 and grad_norm > max_grad_norm * 0.5:
                print(f"  (grad_norm: {grad_norm:.2f})")

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

