#!/bin/bash
# Frozen GPT-2 + Mamba selective attention, tuned for lower literature perplexity.
# Uses gradient checkpointing to avoid OOM on ~15GB GPUs.
export TRAIN_MODEL=gpt2_mamba_selective
export FREEZE_GPT2=1
export BALANCED_TRAINING=1
export LITERATURE_FOCUSED=1
export BATCH_SIZE=1
export MAX_LENGTH=384
export MAX_STEPS=500
python -m src.training.run_training
