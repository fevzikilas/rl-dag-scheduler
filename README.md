# Task Scheduling in High-Performance Computing Environments with Reinforcement Learning

> ** Hacettepe University **  
> Fevzi Kılaş . Ahmet Deniz Güner . Yunus Can Bilge

---

## Overview

This repository contains the full implementation of a **Reinforcement Learning-based DAG scheduler** for single-GPU inference. The agent (MaskablePPO + GNN encoder) learns to assign computation graph operators to parallel CUDA streams while minimising execution makespan and enforcing shared VRAM limits via action masking.

Key design goal: **zero-shot hardware generalisation** - both GPU memory capacity (4–100 GB) and stream count (2–8) are randomised per episode during training, so a single trained model adapts at inference time to any GPU in those ranges without retraining.

- **Dataset:** 172 real-world DAGs extracted from CNNs, Vision Transformers, and BERT-family models - published as [`nieche/DL-Architectural-DAGs-2026`](https://huggingface.co/datasets/nieche/DL-Architectural-DAGs-2026) on HuggingFace Hub.
- **Baselines:** HEFT (Heterogeneous Earliest Finish Time) and Round Robin.
- **Paper:** ICLR-format report in [`main.tex`](main.tex).

---

## Repository Structure

```
RL project/
├── rl_scheduler/               # Main Python package
│   ├── config.py               # Hardware specs & training hyperparameters
│   ├── data_loader.py          # HuggingFace dataset → NetworkX DAGs + features
│   ├── gnn_policy.py           # GNN encoder (mean-agg MP) + SB3 feature extractor
│   ├── hpc_env.py              # Gymnasium environment (list-scheduling MDP)
│   ├── baselines.py            # HEFT and Round Robin schedulers
│   ├── train.py                # MaskablePPO training loop + eval callback + CSV log
│   └── evaluate.py             # Per-graph evaluation → CSV + plots
├── dataset/
│   ├── extract_hpc_dataset.py  # Extracts DAGs from timm/torchvision/HF models
│   ├── models.ipynb            # Exploration notebook
│   └── upload_to_hub.py        # Pushes dataset to HuggingFace Hub
├── checkpoints/                # Saved model weights (generated at runtime)
├── build_cache.py              # One-time: downloads dataset → graphs_cache.pkl
├── main.tex                    # ICLR-format paper (LaTeX source)
├── evaluation_results.csv      # Per-graph results (35 test graphs)
├── evaluation_plot.png         # 3-panel evaluation figure
├── requirements_rl.txt         # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Create and activate a conda environment

```bash
conda create -n rl_sch python=3.10 -y
conda activate rl_sch
```

### 2. Install dependencies

```bash
pip install -r requirements_rl.txt
```

> **Windows note:** `torch_geometric` may cause a DLL crash on Windows. The code detects this automatically and falls back to the built-in mean-aggregation GNN - no action needed.

### 3. Build the dataset cache (run once)

Downloads the 172 DAGs from HuggingFace and saves them to `graphs_cache.pkl`:

```bash
python build_cache.py
```

This only needs to be run once. All subsequent runs load from the pickle file.

### 4. Train the agent

```bash
python -m rl_scheduler.train
```

- Trains for **1,000,000 environment steps** (~85 min on an NVIDIA GPU at ~200 fps).
- Best checkpoint (by RL/HEFT ratio) is saved to `checkpoints/best_model.zip`.
- Evaluation log is written to `checkpoints/eval_log.csv` every 20,000 steps.

### 5. Evaluate

```bash
python -m rl_scheduler.evaluate
```

Outputs:
- `evaluation_results.csv` - per-graph makespan for RL agent, HEFT, and Round Robin.
- `evaluation_plot.png` - 3-panel figure: scatter plot, ratio histogram, ratio vs. graph size.

---

## Configuration

All hyperparameters and hardware settings are in [`rl_scheduler/config.py`](rl_scheduler/config.py):

| Parameter | Value | Notes |
|---|---|---|
| `GPU_FLOPS_CAP` | 10 TFLOPS | Total GPU compute |
| `MAX_STREAMS` | 8 | Fixed obs/action space upper bound |
| `N_STREAMS_MIN / MAX` | 2 / 8 | Training range (randomised per episode) |
| `N_STREAMS_DEFAULT` | 4 | Evaluation default |
| `GPU_MEMORY_MIN / MAX` | 4 GB / 100 GB | Training range (randomised per episode) |
| `GPU_MEMORY_DEFAULT` | 8 GB | Evaluation default |
| `HIDDEN_DIM` | 128 | GNN embedding size |
| `N_GAT_LAYERS` | 3 | GNN depth |
| `PPO_LR` | 3e-4 | Learning rate |
| `PPO_N_STEPS` | 1,024 | Rollout length |
| `PPO_BATCH_SIZE` | 64 | Minibatch size |
| `TOTAL_TIMESTEPS` | 1,000,000 | Total training steps |
| `TRAIN_RATIO` | 0.80 | 137 train / 35 test graphs |

---

## Model Design

### Observation Space (fixed at MAX_STREAMS=8)

| Key | Shape | Description |
|---|---|---|
| `task_emb` | `(128,)` | GNN embedding of current ready operator |
| `proc_feat` | `(24,)` | `MAX_STREAMS × 3`: avail-time, load, active-indicator per slot |
| `gpu_state` | `(4,)` | `[vram_used_norm, gpu_cap_norm, n_streams_norm, progress_norm]` |
| `valid_mask` | `(8,)` | Binary: 0 = inactive or VRAM-violating stream |

### Action Space

`Discrete(8)` - inactive stream slots and VRAM-violating streams are masked to 0.

### Reward

- **Step:** `-Δidle × 0.01` (penalise idle time created on chosen stream)
- **Terminal:** `-makespan / lower_bound` (normalised makespan penalty)
- **VRAM violation:** `-50`

---

## Hardware Generalisation

Both GPU memory and stream count are randomised per episode during training:

```
gpu_capacity ~ Uniform(4 GB, 100 GB)
n_streams    ~ randint(2, 9)            # {2, 3, 4, 5, 6, 7, 8}
```

Both values are encoded in the `gpu_state` observation, enabling the policy to adapt zero-shot to any hardware configuration in those ranges at evaluation time.

---

## Paper

The LaTeX source for the accompanying ICLR-format report is in [`main.tex`](main.tex).

To compile:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Dependencies

See [`requirements_rl.txt`](requirements_rl.txt). Key packages:

| Package | Purpose |
|---|---|
| `torch >= 2.0` | Neural network backend |
| `gymnasium >= 0.29` | RL environment interface |
| `stable-baselines3 >= 2.0` | PPO implementation |
| `sb3-contrib >= 2.0` | MaskablePPO (action masking) |
| `networkx >= 3.0` | DAG representation |
| `datasets` | HuggingFace dataset loading (cache build only) |
| `pandas`, `matplotlib` | Evaluation and plotting |

---

## License

This project was developed as a course final project at Hacettepe University (2025–2026).
