# Task Scheduling in High-Performance Computing Environments with Reinforcement Learning

> **Hacettepe University**  
> Fevzi Kılas · Ahmet Deniz Güner · Yunus Can Bilge

---

## Overview

This repository contains the full implementation of a **Reinforcement Learning-based DAG scheduler** for single-GPU inference. The agent (MaskablePPO + GNN encoder) learns to assign computation graph operators to parallel CUDA streams while minimising execution makespan and enforcing shared VRAM limits via action masking.

Key design goal: **zero-shot hardware generalisation** — both GPU memory capacity (4–100 GB) and CUDA stream count (2–8) are randomised per episode during training, so a single trained model adapts at inference time to any GPU in those ranges without retraining.
<!--  Aslinda random olmasinin asil amaci herhangi bir settingi ezberlemesini istemiyoruz   -->

- **Dataset:** 172 real-world DAGs extracted from CNNs, Vision Transformers, and BERT-family models — published as [`nieche/DL-Architectural-DAGs-2026`](https://huggingface.co/datasets/nieche/DL-Architectural-DAGs-2026) on HuggingFace Hub.
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
│   ├── models.ipynb            # Exploration notebook - datasets, libraries, etc.
│   └── upload_to_hub.py        # Pushes dataset to HuggingFace Hub
├── checkpoints/                # Saved model weights (generated at runtime)
├── build_cache.py              # One-time: downloads dataset → graphs_cache.pkl
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

- Trains for **2,000,000 environment steps**.
- Learning rate decays linearly from **3×10⁻⁴ → 3×10⁻⁵** over training.
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
| `PPO_LR` | 3e-4 → 3e-5 | Linear decay over full training |
| `PPO_N_STEPS` | 2,048 | Rollout length per update |
| `PPO_BATCH_SIZE` | 64 | Minibatch size |
| `PPO_N_EPOCHS` | 10 | Gradient steps per rollout |
| `PPO_ENT_COEF` | 0.05 | Entropy bonus for exploration |
| `IDLE_PENALTY` | 0.1 | Weight on normalised idle-time step penalty |
| `TOTAL_TIMESTEPS` | 2,000,000 | Total training steps |
| `N_EVAL_EPISODES` | 35 | Full test set evaluated every `EVAL_FREQ` steps |
| `TRAIN_RATIO` | 0.80 | 137 train / 35 test graphs |

---

## Model Design

### Observation Space (fixed at MAX_STREAMS=8)

| Key | Shape | Description |
|---|---|---|
| `task_emb` | `(128,)` | GNN embedding of current ready operator |
| `proc_feat` | `(32,)` | `MAX_STREAMS × 4`: `[avail_norm, eft_norm, load_norm, active_flag]` per slot |
| `gpu_state` | `(5,)` | `[vram_used_norm, gpu_cap_norm, n_streams_norm, progress_norm, queue_depth_norm]` |
| `valid_mask` | `(8,)` | Binary: 0 = inactive or VRAM-violating stream |

`eft_norm` is the estimated finish time for the current task on each stream — the same key signal HEFT uses when making decisions.

### Action Space

`Discrete(8)` — inactive stream slots are masked to 0. VRAM-violating assignments no longer hard-block (VRAM is tracked but the episode continues to prevent infinite loops from mask fallback).

<!-- 

REWARD AND Experiment Log ----------------------------------------------------------------



---

### NEXT TODO's (if v2 still doesn't beat HEFT)

- [ ] Curriculum learning: start with small DAGs (≤10 nodes), gradually increase complexity
- [ ] Reward shaping: add bonus for each task that matches HEFT's stream assignment
- [ ] Larger GNN: increase `HIDDEN_DIM` from 128 → 256, `N_GAT_LAYERS` 3 → 4
- [ ] Longer rollouts: try `PPO_N_STEPS = 4096`
- [ ] Frame stacking: include last-N task assignments in observation
- [ ] Try SAC or TD3 with continuous relaxation instead of MaskablePPO
- [ ] Pre-train GNN encoder jointly with PPO instead of keeping it frozen



LOG OF REWARD:

### ---- v1

- **Step:** `-Δidle × 0.01` (penalise idle time created on chosen stream)
- **Terminal:** `-makespan / lower_bound` (normalised makespan penalty)
- **VRAM violation:** `-50`

Current reward since 13.05.2026 ---- v2

| **Step** | `-idle_norm × IDLE_PENALTY` | `idle_norm = idle_created / lower_bound` — dimensionless |
| **Terminal** | `clip((heft_ms - rl_ms) / heft_ms × 10, −10, 10)` | Positive when RL beats HEFT, zero when equal |

 -->


 <!-- ---

## Experiment Log

This section tracks every training run, what we changed, and what we observed. Useful for not repeating the same mistakes.

---

### Run v1 — First attempt (abandoned at ~800k steps)

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `PPO_LR` | 3e-4 (constant) |
| `PPO_N_STEPS` | 1,024 |
| `PPO_BATCH_SIZE` | 64 |
| `PPO_ENT_COEF` | 0.01 |
| `TOTAL_TIMESTEPS` | 1,000,000 |
| `N_EVAL_EPISODES` | 20 (random sample each eval) |

**Reward design:**
- Step: `-Δidle × 0.01` (absolute seconds)
- Terminal: `-makespan / lower_bound`
- VRAM violation: `-50` hard penalty

**Observation:**
- `proc_feat`: `[avail_time, load, active_flag]` per stream (3 features × 8 = 24)
- `gpu_state`: `[vram_used_norm, gpu_cap_norm, n_streams_norm, progress_norm]` (4)

**Results:**
- Average RL/HEFT ratio ≈ **1.094** (9.4% worse than HEFT)
- `ep_rew_mean` showed `NaN` throughout the entire run
- Eval ratio std dev ≈ 0.22, range 0.55–1.51 (completely noisy)
- No visible learning curve improvement

**Root causes identified:**

| Bug / Issue | Description |
|---|---|
| Reward always ≈ -1.0 | `-(makespan/lower_bound)` collapses to the same value regardless of how good or bad the schedule is — beating HEFT by 20% vs losing by 20% gives nearly the same reward |
| Idle penalty irrelevant | Idle penalty was in absolute seconds (~1e-5 s), resulting in ~5×10⁻⁸ per step — basically zero gradient signal next to the terminal -1.0 |
| VRAM infinite loop | When VRAM was full, the mask fallback opened all actions; then `step()` saw VRAM violation and returned -50 without advancing the task → episode looped forever |
| Eval variance | `eval_agent` sampled 20 random graphs each eval → different graphs every time → std dev 0.22 completely masked real learning signal |
| `ep_rew_mean = NaN` | `self.logger.name_to_value.get("rollout/ep_rew_mean")` had wrong key or timing issue with VecMonitor — always returned NaN |
| Missing EFT in obs | Agent only saw `avail_time[s]` per stream, not the estimated finish time — HEFT's core signal for decision-making |

---

### Run v2 — Current (in progress)

**What changed from v1:**

| Change | Before | After |
|---|---|---|
| Terminal reward | `-makespan / lower_bound` (always ≈ -1.0) | `clip((heft_ms - rl_ms) / heft_ms × 10, -10, 10)` |
| Idle penalty | `-idle_seconds × 0.01` (~5e-8) | `-idle_norm × 0.1` where `idle_norm = idle / lower_bound` |
| VRAM violation handling | Return `-50`, episode stuck | Set `out_bytes = 0.0`, proceed normally |
| EFT in observation | Not included | Added `eft_norm` per stream in `proc_feat` (3→4 features per slot) |
| Queue depth in observation | Not included | Added `queue_depth_norm` to `gpu_state` (4→5 features) |
| Entropy coefficient | 0.01 | 0.05 |
| Rollout length | 1,024 | 2,048 |
| Total timesteps | 1,000,000 | 2,000,000 |
| Learning rate | 3e-4 constant | 3e-4 → 3e-5 linear decay |
| Eval graphs | 20 random each eval | Fixed 35-graph subset (seed=42) |
| `ep_rew_mean` source | `logger.name_to_value` (broken) | `model.ep_info_buffer` (correct) |
| HEFT comparison | Separate `eval_baselines` call | Computed per-episode inside `hpc_env.py` — always uses same graphs/hardware as RL |

**Current status:** Training — results pending.

--- -->
### Reward

| Signal | Formula | Notes |
|---|---|---|
| **Step** | `-idle_norm × IDLE_PENALTY` | `idle_norm = idle_created / lower_bound` — dimensionless |
| **Terminal** | `clip((heft_ms - rl_ms) / heft_ms × 10, −10, 10)` | Positive when RL beats HEFT, zero when equal |

The terminal reward is **HEFT-relative**: the agent receives +1 for every 10% improvement over HEFT and -1 for every 10% it falls behind. This gives a clear signal that scales regardless of absolute makespan magnitude.



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
