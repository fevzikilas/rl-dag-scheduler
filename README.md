# Task Scheduling in GPU Inference with Reinforcement Learning

> **Hacettepe University**  
> Fevzi Kılas · Ahmet Deniz Güner · Yunus Can Bilge

---

## What is this?

We train an RL agent (MaskablePPO) to schedule computation graph operators onto parallel CUDA streams on a single GPU. The agent learns to assign each operator to a stream in order to minimize total execution time while staying within the GPU memory limit.

**Key feature:** The agent trains on random hardware (4–100 GB memory, 2–8 streams) and works on any GPU in that range without retraining.

- **Dataset:** 172 computation graphs from real models (ResNet, ViT, BERT, etc.) — available on [HuggingFace Hub](https://huggingface.co/datasets/nieche/DL-Architectural-DAGs-2026)
- **Baselines:** HEFT (standard scheduling algorithm) and Round Robin

---

## Files

- `rl_scheduler/` — main code
  - `config.py` — hyperparameters and hardware settings
  - `data_loader.py` — loads DAGs from HuggingFace
  - `hpc_env.py` — GPU scheduling environment
  - `gnn_policy.py` — GNN encoder + policy network
  - `baselines.py` — HEFT and Round Robin (for comparison)
  - `train.py` — training script (PPO + BC pretraining)
  - `pretrain.py` — behavioral cloning (imitate HEFT)
  - `evaluate.py` — test the trained agent
- `checkpoints/` — saved models (created when you run training)
- `build_cache.py` — download dataset one time
- `requirements_rl.txt` — Python packages

---

## Quick Start

### 1. Setup

```bash
conda create -n rl_sch python=3.10 -y
conda activate rl_sch
pip install -r requirements_rl.txt
```

### 2. Download dataset (one time only)

```bash
python build_cache.py
```

This downloads 172 computation graphs and saves them locally.

### 3. Train the agent (optional: behavioral cloning first)

If you want the agent to start by imitating HEFT:
```bash
python -m rl_scheduler.pretrain
```

Then train with PPO:
```bash
python -m rl_scheduler.train
```

- Trains for 2 million steps
- Best model saved to `checkpoints/best_model.zip`
- Training log saved to `checkpoints/eval_log.csv`

### 4. Evaluate

```bash
python -m rl_scheduler.evaluate
```

Outputs CSV file with results and a plot comparing RL vs HEFT.

---

## Settings

Edit `rl_scheduler/config.py` to change training parameters:

| What | Value | 
|---|---|
| GPU memory (training range) | 4–100 GB (random per episode) |
| GPU memory (evaluation) | 8 GB |
| Parallel streams (training) | 2–8 (random per episode) |
| Parallel streams (evaluation) | 4 |
| GNN layers | 3 |
| PPO learning rate | 3e-4 → 3e-5 (linear decay) |
| Training steps | 2,000,000 |
| Batch size | 64 |

---

## How it works

1. **Graph representation:** Each computation graph is a set of operators (nodes) with data dependencies (edges).

2. **Scheduling problem:** At each step, pick the next operator from the ready queue and assign it to one of the GPU streams. Goal: minimize total time while respecting memory limits.

3. **RL solution:** The agent (MaskablePPO) learns which stream to assign each operator to, by observing:
   - The current operator's features
   - Each stream's current availability time
   - GPU memory usage
   
4. **Training:** The agent trains on random hardware (random memory, random stream count per episode) so it learns to adapt to different GPUs.

5. **Results:** On 35 test graphs, the agent beats HEFT on 25 of them (71%).

---

## Dependencies

- `torch >= 2.0`
- `gymnasium >= 0.29`
- `stable-baselines3 >= 2.0`
- `sb3-contrib >= 2.0`
- `networkx >= 3.0`
- `datasets` (for downloading data)
- `pandas`, `matplotlib` (for plots)

Install with:
```bash
pip install -r requirements_rl.txt
```

---

## Reference

For technical details, see `main.tex` (ICLR-format paper).
