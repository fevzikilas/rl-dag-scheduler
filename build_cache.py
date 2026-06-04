"""Download dataset from HuggingFace and cache computation graphs locally."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rl_scheduler"))

from data_loader import _download_and_cache

if __name__ == "__main__":
    _download_and_cache()
    print("\nCache built successfully!")
    print("Run: python -m rl_scheduler.pretrain   # (optional: behavioral cloning)")
    print("Then: python -m rl_scheduler.train     # (PPO training)")
