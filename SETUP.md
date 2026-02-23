# Project setup: steps (no code)

Use this as a checklist to set up folders and a minimal template. Implement the code yourself in each step.

---

## 1. Repo layout

Create the following directories (and keep them even if empty at first):

- **`data/`** — Raw and preprocessed datasets, and eval result files (e.g. JSONL).
- **`src/`** (or **`models/`**) — Model code: Mamba (S6), Transformer blocks, hybrid stack, RMSNorm, etc.
- **`scripts/`** — Runners: download data, preprocess, train, evaluate (analytical + synthetic).
- **`configs/`** (optional) — YAML/JSON for hyperparameters and paths so you don’t hardcode.
- **`notebooks/`** (optional) — Exploratory analysis or small experiments.

Decide where to put “entry point” scripts: e.g. `scripts/download_data.py`, `scripts/run_analytical_eval.py`, `scripts/run_synthetic_eval.py`, and either a single `train.py` at root or `scripts/train.py`.

---

## 2. Environment

- Create a **virtual environment** (e.g. `python -m venv .venv`) and activate it.
- Add a **`requirements.txt`** (or use `pyproject.toml`). Include at least: `torch`, `datasets`, `transformers` (for tokenizer/GPT-2), and any Mamba reference impl you use; add `tqdm`, `pyyaml`, `python-dotenv` if you use them. Pin major versions so runs are reproducible.
- Use a **`.env`** (and `.env.example`) for paths or API keys if needed; add `.env` to `.gitignore`.

---

## 3. Data pipeline

- **Download:** In `scripts/` or `data/`, add a script that:
  - Loads **OlympiadBench** from HuggingFace (see [OpenBMB/OlympiadBench](https://github.com/OpenBMB/OlympiadBench); full benchmark has 8,476 problems). Filter to **text-only English** (e.g. `*_TO_*_en_*.json`), then take **100 examples** as the training subset so training stays tractable. Write canonical JSONL with question, solution, final_answer, subject, subfield. Optionally keep a separate small held-out set (e.g. 50–100) for evaluation.
  - Loads **gutenberg-en-v1-clean**, filters by label score > 0.95, samples 500 (or your target size), and writes one JSONL or one file per book with raw text.
- **Preprocess:**
  - OlympiadBench (100-example subset): normalize math strings and apply LatexStandardizer to solution/final_answer; decide how you represent “question” and “final answer” for prompts and scoring.
  - Gutenberg: strip headers/footers, replace hard line breaks; then in the eval script (or a separate step), split each book into “first 7000 tokens” and “next 1192” (e.g. with your chosen tokenizer).
- Keep **paths configurable** (env or config file) so you can point to `data/olympiad_*.jsonl` and `data/gutenberg_*.jsonl` without hardcoding.

---

## 4. Model code layout

- Under **`src/`** (or `models/`):
  - **Attention:** Implement causal self-attention with KV-cache (and optionally multi-head); use it in “Transformer block” layers (e.g. layers 4, 8, 12).
  - **Mamba / S6:** Implement or wrap a Selective SSM (S6) block; use it in the Mamba layers (1–3, 5–7, 9–11). If you rely on an external Mamba impl, put a thin wrapper here so the rest of the code only depends on your interface.
  - **Hybrid:** A module that composes 12 layers in the alternating pattern (Mamba, Mamba, Mamba, Transformer, …), with RMSNorm at the start of each block and residual connections; manage KV-cache only at Transformer layers.
- **Baselines:** Decide how you’ll run “pure Mamba” (e.g. 24-layer Mamba-only stack) and “GPT-2” (e.g. HuggingFace `transformers`). You can have a small wrapper or config that selects model type (HMT / pure Mamba / GPT-2) so the same eval scripts can loop over them.

---

## 5. Training (if you train from scratch)

- **Config:** Max sequence length, batch size, learning rate, number of steps/epochs, checkpoint dir, data paths. Put these in `configs/` or env.
- **Dataloader:** For math you might do next-token prediction on (question + solution) or only solution; for literature, next-token on book chunks. Expose a single iterator that returns input_ids and labels.
- **Train loop:** One script that loads HMT (or baseline), runs forward/backward, logs loss, saves checkpoints. Optionally log throughput (tokens/sec) so you can compare later.

---

## 6. Evaluation scripts

- **Analytical (OlympiadBench):**
  - **Prompt template:** One function that takes a question (and maybe subject) and returns the string you send to the model (e.g. “Solve step by step and put the final answer in \boxed{...}”).
  - **Output extractor:** One function that takes the model’s raw generation and returns the extracted answer string or a sentinel like `[invalid]` (see EVALUATION_GOALS.md and A4’s `standard_output_extractor`).
  - **Runner:** Load test JSONL; for each model (HMT, Mamba, GPT-2), for each problem: call model, extract answer, compare to ground truth (with normalizations), record correct/incorrect and optionally wall-clock time and token counts. Write one row per (problem, model) to something like `data/olympiad_eval_results.jsonl`.
  - **Summary:** Print accuracy per model and throughput (e.g. time per problem or tokens/sec). Optionally plot (e.g. bar chart of accuracies).
- **Synthetic (literature):**
  - **Data:** For each book, have 7000-token context and 1192-token target sequence (8192 total). Use the same tokenizer as the model.
  - **Scoring:** For each model, run forward on (context + target) and compute mean log-probability of the target tokens given context (or use the loss). Perplexity = exp(mean negative log-prob).
  - **Runner:** Loop over books and models; write per-book, per-model perplexity to a file (e.g. `data/literature_eval_results.jsonl` or CSV). Print mean (and optionally std) perplexity per model. Optionally plot distributions.

---

## 7. Reproducibility

- **Seeds:** Set `torch.manual_seed`, `numpy.random.seed`, and any dataloader `worker_init_fn` so runs are reproducible.
- **Artifacts:** Save config (or script args) next to checkpoints; record data version (e.g. dataset name and split) in eval result files.
- **README:** Document how to install deps, download/preprocess data, train, and run both evals so a TA or teammate can rerun everything.

---

## 8. Order of operations (suggested)

1. Create folders and venv; add `requirements.txt` and `.gitignore`.
2. Implement data download and preprocessing; verify JSONL format and tokenization (e.g. 7000 + 1192 for Gutenberg).
3. Implement or wire up the three model variants (HMT, pure Mamba, GPT-2) so you can run forward and get logits (and optionally loss).
4. Implement analytical eval (prompt + extractor + loop + results JSONL + accuracy/throughput).
5. Implement synthetic eval (perplexity on 1192 tokens + results file + summary).
6. If training: add train script and config; train HMT (and optionally baselines) then re-run evals.

You can start with a tiny subset of data (e.g. 10 math problems, 5 books) to validate the pipeline before full runs.
