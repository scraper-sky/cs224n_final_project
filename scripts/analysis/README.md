# Analysis Scripts

This directory contains lightweight analysis scripts for the paper's interpretability section. They are designed to work with the existing model loaders, checkpoints, math prompts, and evaluation style already used in the repo.

## Files

- `_common.py`
  Shared helpers for loading models, loading checkpoints, generation, answer matching, teacher-forced loss, JSONL/CSV writing, and token-level utilities.

- `win_loss_analysis.py`
  Runs multiple models on the same math dataset and writes per-example predictions, correctness flags, model accuracy, pairwise win/loss counts, and agreement patterns.

- `context_sensitivity.py`
  Re-runs evaluation at several context lengths and measures how math exact match and/or literature perplexity change as the visible context shrinks.

- `run_training_logging.sh`
  Bash loop for re-running training with logs and checkpoints saved in a structured way.

- `plot_training_curves.py`
  Parses training logs from `run_training_logging.sh` and writes CSVs plus simple loss/LR plots.

- `hybrid_gate_analysis.py`
  Scans hybrid checkpoints and extracts gate parameters like `mamba_alpha` or `mamba_gate`, then plots their sigmoid values over training steps.

- `attention_analysis.py`
  Summarizes attention behavior for GPT-2 and the custom hybrid attention layers by measuring mean attention distance, attention mass on numeric tokens, and top-attended tokens for the last position.

- `mamba_saliency.py`
  Runs perturbation-based saliency for Mamba by ablating spans in the prompt, measuring delta loss on the gold answer, and comparing hidden states before and after the most important ablation.

- `numeric_token_saliency.py`
  Compares how much loss increases when numeric prompt tokens are masked versus non-numeric control tokens.

- `error_taxonomy.py`
  Consumes per-example predictions from `win_loss_analysis.py` and assigns a heuristic error label such as `sign_error`, `wrong_final_extraction`, or `no_numeric_answer`.

## Typical setup

From the repo root:

```bash
python -m pip install matplotlib tqdm
```

If you are on Colab, you can paste the same commands into notebook cells with `!`.

## How to run

### 1. Win/loss analysis

```bash
python scripts/analysis/win_loss_analysis.py \
  --models gpt2,mamba,hybrid_model_v2 \
  --input-path data/olympiad_preprocessed.jsonl \
  --max-samples 100 \
  --checkpoint-map "hybrid_model_v2=checkpoints/hybrid_model_v2_step500.pt"
```

Outputs:

- `per_example_predictions.jsonl`
- `accuracy_summary.csv`
- `pairwise_win_loss.csv`
- `agreement_patterns.csv`

### 2. Context-length sensitivity

```bash
python scripts/analysis/context_sensitivity.py \
  --models gpt2,mamba,hybrid_model_v2 \
  --mode both \
  --lengths 64,128,256,512,1024 \
  --checkpoint-map "hybrid_model_v2=checkpoints/hybrid_model_v2_step500.pt"
```

Outputs:

- `context_sensitivity.csv`
- optional PNG plots for math and literature

### 3. Re-run training with logs

```bash
chmod +x scripts/analysis/run_training_logging.sh
MODELS="gpt2,mamba,hybrid_model_v2" \
MAX_STEPS=500 \
SAVE_EVERY=100 \
BALANCED_TRAINING=1 \
LITERATURE_RATIO=0.5 \
scripts/analysis/run_training_logging.sh
```

This writes:

- training logs under `scripts/analysis/outputs/training_logs/<run_tag>/`
- checkpoints under `checkpoints/<run_tag>/`

Then parse the logs:

```bash
python scripts/analysis/plot_training_curves.py \
  --log-glob "scripts/analysis/outputs/training_logs/*/*.log"
```

### 4. Hybrid gate analysis

```bash
python scripts/analysis/hybrid_gate_analysis.py \
  --checkpoint-glob "checkpoints/**/*.pt"
```

Outputs:

- `hybrid_gate_values.csv`
- `hybrid_gate_trajectory.png`

### 5. Attention analysis

For GPT-2:

```bash
python scripts/analysis/attention_analysis.py \
  --model-name gpt2 \
  --input-path data/olympiad_preprocessed.jsonl \
  --max-samples 25
```

For the hybrid:

```bash
python scripts/analysis/attention_analysis.py \
  --model-name hybrid_model_v2 \
  --input-path data/olympiad_preprocessed.jsonl \
  --max-samples 25 \
  --checkpoint-map "hybrid_model_v2=checkpoints/hybrid_model_v2_step500.pt"
```

Outputs:

- per-example JSONL with token-level attention summaries
- per-layer CSV and optional plot

### 6. Mamba saliency and hidden-state sensitivity

```bash
python scripts/analysis/mamba_saliency.py \
  --model-name mamba \
  --input-path data/olympiad_preprocessed.jsonl \
  --max-samples 25
```

Outputs:

- per-example span-importance JSONL
- per-layer hidden delta CSV
- optional plot

### 7. Numeric token saliency

```bash
python scripts/analysis/numeric_token_saliency.py \
  --models gpt2,mamba,hybrid_model_v2 \
  --max-samples 20 \
  --checkpoint-map "hybrid_model_v2=checkpoints/hybrid_model_v2_step500.pt"
```

Outputs:

- per-token CSV
- summary CSV
- optional bar chart

### 8. Error taxonomy

Run this after `win_loss_analysis.py`:

```bash
python scripts/analysis/error_taxonomy.py \
  --predictions-path scripts/analysis/outputs/win_loss/per_example_predictions.jsonl
```

Outputs:

- `error_taxonomy_per_example.csv`
- `error_taxonomy_summary.csv`
- optional bar chart

## Suggested workflow

1. Run `win_loss_analysis.py` to get aligned predictions across models.
2. Run `error_taxonomy.py` on those predictions.
3. Run `context_sensitivity.py` for the main architecture story.
4. If you re-train, use `run_training_logging.sh`, then `plot_training_curves.py` and `hybrid_gate_analysis.py`.
5. Add `attention_analysis.py`, `mamba_saliency.py`, and `numeric_token_saliency.py` for the interpretability section.

## Notes

- `attention_analysis.py` supports plain GPT-2 and the custom hybrid models with explicit `layers`, `ln_1`, and `c_attn` modules.
- `mamba_saliency.py` is intended for the pure Mamba model and relies on `output_hidden_states=True`.
- `error_taxonomy.py` uses heuristics, so it is best treated as a first-pass labeling tool before manual inspection.
- For hybrid checkpoints, prefer `--checkpoint-map` when comparing multiple custom models at once.