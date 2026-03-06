# =============================================================================
# CS224N Hybrid Model: Full Colab Run
# Paste this entire block into a single Colab cell and run.
# =============================================================================

import os
import subprocess
import sys

# --- 1. Clone and setup (use /content to avoid nested-path issues) ---
REPO = "https://github.com/scraper-sky/cs224n_final_project"
BASE = os.environ.get("COLAB_DRIVE", "/content")  # Colab default; set if using Drive
PROJ = os.path.join(BASE, "cs224n_final_project")
os.makedirs(BASE, exist_ok=True)
os.chdir(BASE)
subprocess.run(["rm", "-rf", "cs224n_final_project"], check=False)
subprocess.run(["git", "clone", REPO], check=True)
os.chdir(PROJ)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

# --- 2. Environment ---
os.environ["DATA_DIR"] = os.path.join(PROJ, "data")
os.environ["CHECKPOINT_DIR"] = os.path.join(PROJ, "checkpoints")
os.environ["MATH_PREPROCESSED_JSONL"] = os.path.join(os.environ["DATA_DIR"], "olympiad_preprocessed.jsonl")
os.environ["GUTENBERG_CHUNKS_JSONL"] = os.path.join(os.environ["DATA_DIR"], "gutenberg_7000_1192.jsonl")
os.environ["RESULTS_JSON"] = os.path.join(PROJ, "eval_results.json")

# --- 3. Data preparation ---
print("Downloading math...")
subprocess.run([sys.executable, "scripts/preprocessing/download_math.py"], check=True, env=os.environ)

print("Preprocessing math...")
subprocess.run([sys.executable, "scripts/preprocessing/preprocess_math.py"], check=True, env=os.environ)

print("Downloading literature...")
subprocess.run([sys.executable, "scripts/preprocessing/download_literature.py"], check=True, env=os.environ)

print("Preprocessing literature...")
subprocess.run([sys.executable, "scripts/preprocessing/preprocess_literature.py"], check=True, env=os.environ)

# --- 4. Training (hybrid, direct_qa, full fine-tune) ---
train_env = os.environ.copy()
train_env["TRAIN_MODEL"] = "hybrid"
train_env["TRAIN_DIRECT_QA"] = "1"
train_env["APPEND_ANSWER"] = "1"
train_env["MAX_STEPS"] = "250"
train_env["SAVE_EVERY"] = "125"
train_env["BATCH_SIZE"] = "2"
train_env["MAX_LENGTH"] = "512"
train_env["WARMUP_STEPS"] = "25"
print("Training hybrid model...")
subprocess.run(
    [sys.executable, "-m", "src.training.run_training"],
    check=True,
    env=train_env,
)

# --- 5. Evaluation ---
ckpt = os.path.join(PROJ, "checkpoints", "hybrid_step250.pt")
eval_env = os.environ.copy()
eval_env["EVAL_MODELS"] = "hybrid"
eval_env["EVAL_CHECKPOINT"] = ckpt
eval_env["CONTEXT_WINDOW"] = "1024"
eval_env["MAX_NEW_TOKENS"] = "256"
eval_env["MAX_MATH_SAMPLES"] = "100"
eval_env["MAX_LIT_SAMPLES"] = "50"
print("Evaluating...")
subprocess.run(
    [sys.executable, "scripts/preprocessing/eval/eval_models.py"],
    check=True,
    env=eval_env,
)

print("\nDone. Results:", os.environ["RESULTS_JSON"])
