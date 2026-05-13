"""
Diagnose silent segfaults by testing each import in its own subprocess.
Run: python diag.py
"""
import subprocess, sys, os

BASE = os.path.dirname(__file__)
RL   = os.path.join(BASE, "rl_scheduler")

TESTS = [
    ("torch",          "import torch; print(torch.__version__)"),
    ("gymnasium",      "import gymnasium"),
    ("sb3",            "from stable_baselines3 import PPO"),
    ("sb3-contrib",    "from sb3_contrib import MaskablePPO"),
    ("datasets",       "from datasets import load_dataset"),
    ("config",         f"import sys; sys.path.insert(0,r'{RL}'); import config"),
    ("data_loader",    f"import sys; sys.path.insert(0,r'{RL}'); from data_loader import load_all_graphs"),
    ("gnn_policy",     f"import sys; sys.path.insert(0,r'{RL}'); from gnn_policy import NodeEncoder"),
    ("hpc_env",        f"import sys; sys.path.insert(0,r'{RL}'); from hpc_env import HPCClusterEnv"),
    ("baselines",      f"import sys; sys.path.insert(0,r'{RL}'); from baselines import heft"),
]

for name, code in TESTS:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30
    )
    ok  = result.returncode == 0
    err = (result.stderr.strip().split("\n")[-1] if result.stderr.strip() else "").replace(
        "2026-05", "").strip()[:120]
    print(f"{'  OK' if ok else 'FAIL'}  {name:<15}  rc={result.returncode}"
          + (f"  → {err}" if not ok else ""))
