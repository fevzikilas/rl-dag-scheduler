"""RL-based DAG scheduler for single-GPU inference.

A MaskablePPO agent (with a GNN encoder) learns to assign DAG operators to
parallel CUDA streams while minimising makespan and respecting VRAM limits.

See README.md for an overview and `rl_scheduler/config.py` for hyperparameters.
"""
