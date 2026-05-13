"""
One-time graph cache builder.
Run this ONCE before training:
    python build_cache.py

It downloads the HuggingFace dataset, builds nx.DiGraph objects, and saves
them to graphs_cache.pkl.  Training then loads from the pickle (no TF/datasets
import → no CUDA conflict with PyTorch).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Force TF to CPU before it can grab the GPU
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    print("[build_cache] TensorFlow set to CPU-only.")
except Exception:
    pass

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rl_scheduler"))

from data_loader import _download_and_cache
_download_and_cache()
print("Cache built successfully.  You can now run: python -m rl_scheduler.train")
