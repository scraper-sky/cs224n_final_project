# entry point to run training
# here we use: python -m src.training.run_training

from .train_loop import run_train

if __name__ == "__main__":
    run_train()
