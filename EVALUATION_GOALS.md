# Model Evaluation Goals (aligned with A4 patterns)

This document maps evaluation patterns from **cs224n-A4-student** to the final project so we reuse the same mental model and avoid reinventing the wheel.

---

## What A4 gives us

From `/users/jerryxiao/Downloads/cs224n-A4-student`:

| A4 component | Purpose | Our use |
|--------------|---------|--------|
| **Prompt template + output extractor** | Turn task into a fixed input format; parse model output with regex (e.g. `#### <answer>`) and handle unparseable as `[invalid]`. | **Analytical task:** One (or more) prompt templates for OlympiadBench; extractor for **boxed** final answer (e.g. `\boxed{...}` or `#### ...`); treat unparseable as wrong or track separately. |
| **Score tracker (correct / total)** | Per-model dict: `correct_answers`, `total_asked`; compute accuracy = correct/total; optionally catch exceptions per problem. | Same: per-model (HMT, pure Mamba, GPT-2) track correct vs total on OlympiadBench; report accuracy and optionally invalid count. |
| **Throughput / cost tracking** | A4’s `QueryResponse` has `input_tokens`, `output_tokens`, `cost`. | We care about **throughput** (tokens/sec or time per problem). Goal: HMT ≥ 80% of Mamba’s throughput. So log time and token counts per run. |
| **Results to JSONL** | Save each example with input, model outputs, and derived label (e.g. `alpaca_eval_results.jsonl`: instruction, response_E, response_F, judge_reasoning, winner). | Save **analytical** results to JSONL: question_id, question (or ref), model_id, raw_generation, extracted_answer, ground_truth, correct, (optional) time/tokens). Enables later analysis and plots. |
| **Structured judge output** | Tags like `<MODEL_E_BETTER>` so judge output is parseable; extract winner E/F/TIE/INVALID. | We don’t use an LLM judge for math; we use **exact match**. For literature we use **perplexity**, not pairwise comparison. So no tag-based judge, but the habit of “structured output → parse → aggregate” still applies. |
| **Plots** | Bar chart (E vs F wins, ties); histograms (e.g. preferred vs not-preferred response length). | Optional: bar chart (accuracy per model); histogram of response lengths or of perplexity per book; throughput comparison. |
| **Data loaders** | Read JSONL line-by-line; one dict per line; same format for all scripts. | Same: load OlympiadBench (and Gutenberg) from JSONL or HuggingFace; one canonical format after preprocessing. |

---

## Evaluation goals for this project

### Analytical (OlympiadBench)

- **Metric:** Accuracy = (# correct final answers) / (# questions). Target: match GPT-2 accuracy while achieving ≥ 80% of Mamba’s throughput.
- **Implementation pattern (from A4):**
  1. **Prompt template(s):** Map each problem (question field) to a model input (e.g. “Solve step by step and put the final answer in `\boxed{...}`” or “#### &lt;answer&gt;”).
  2. **Output extractor:** Regex (or small parser) for boxed/#### answer; return a canonical string or `[invalid]`; compare to ground-truth (solution/final_answer field) with normalizations (float, strip, etc.).
  3. **Loop:** For each model (HMT, Mamba, GPT-2): run on each problem, extract answer, compare to ground truth; track correct/total and optionally time and token counts.
  4. **Persistence:** Write results to JSONL (e.g. `data/olympiad_eval_results.jsonl`) for reproducibility and plotting.
  5. **Throughput:** Measure time (and tokens if available) per run; report tokens/sec or time per problem so we can check “≥ 80% of Mamba’s throughput.”

### Synthetic (literature continuation)

- **Metric:** Perplexity on the next 1192 tokens given the first 7000. Lower = better synthesis. No A4 direct equivalent; we add a small pipeline.
- **Implementation pattern:**
  1. **Data:** For each book (or subset): first 7000 tokens = context; next 1192 = target sequence (8192 total).
  2. **Scoring:** For each model, compute mean log-probability (or loss) on the target 1192 tokens conditioned on the 7000-token context; then perplexity = exp(mean negative log-prob).
  3. **Persistence:** Save per-book, per-model perplexity (and optionally mean log-prob) to JSONL or CSV for tables and plots.
  4. **Optional:** Plot perplexity distribution per model (e.g. histogram or box plot).

---

## Summary

- Reuse from A4: **prompt template + output extractor**, **score tracker (correct/total)**, **save results to JSONL**, **throughput/timing**, and **optional plots**.
- Add for this project: **boxed-answer extraction** for OlympiadBench, **throughput comparison** (HMT vs Mamba vs GPT-2), and **perplexity pipeline** for the literature task.
- Keep one canonical **data layout** (e.g. `data/` with train/val/test or splits) and one **eval script per task** that writes to a known results path so everything stays reproducible.
