# evaluate.py — per-graph comparison: RL vs HEFT vs Round Robin
#
# Run: python -m rl_scheduler.evaluate --checkpoint checkpoints/best_model

import os, sys, argparse, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    _USE_MASK = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    _USE_MASK = False

import config
from data_loader import load_all_graphs
from gnn_policy  import NodeEncoder
from hpc_env     import HPCClusterEnv
from baselines   import heft, round_robin


def _wrap(env):
    if _USE_MASK:
        return ActionMasker(env, lambda e: e.get_action_mask())
    return env


def run_agent(model, g, encoder, device, gpu_memory=None,
              n_streams=None) -> float:
    env  = _wrap(HPCClusterEnv([g], encoder, device=device,
                               gpu_memory=gpu_memory, n_streams=n_streams))
    obs, _ = env.reset()
    done   = False
    info   = {}
    while not done:
        if _USE_MASK:
            mask           = env.action_masks()
            action, _      = model.predict(obs, action_masks=mask, deterministic=True)
        else:
            action, _      = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info.get("makespan", float("inf"))


def main(checkpoint: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}\n")

    _, test_graphs = load_all_graphs()

    encoder = NodeEncoder().to(device)
    encoder.eval()

    print(f"Loading model from '{checkpoint}' …")
    model = MaskablePPO.load(checkpoint, device=device)

    rows = []
    gpu_mem  = config.GPU_MEMORY_DEFAULT
    n_s      = config.N_STREAMS_DEFAULT
    for i, g in enumerate(test_graphs):
        model_id = g.graph.get("model_id", f"graph_{i}")
        n_nodes  = len(g)

        rl_ms   = run_agent(model, g, encoder, device,
                            gpu_memory=gpu_mem, n_streams=n_s)
        heft_ms = heft(g, gpu_memory=gpu_mem, n_streams=n_s)["makespan"]
        rr_ms   = round_robin(g, gpu_memory=gpu_mem, n_streams=n_s)["makespan"]
        ratio   = rl_ms / max(heft_ms, 1e-9)

        rows.append({
            "model_id":   model_id,
            "n_nodes":    n_nodes,
            "rl_ms":      rl_ms,
            "heft_ms":    heft_ms,
            "rr_ms":      rr_ms,
            "rl_vs_heft": ratio,
        })

        tag = "✓" if ratio < 1.0 else "✗"
        print(
            f"[{i+1:3d}/{len(test_graphs)}] {tag} {model_id[:45]:<45} "
            f"RL={rl_ms:.3f}  HEFT={heft_ms:.3f}  RR={rr_ms:.3f}  "
            f"ratio={ratio:.3f}"
        )

    df = pd.DataFrame(rows)


    n_better_heft = (df["rl_vs_heft"] < 1.0).sum()
    n_better_rr   = (df["rl_ms"] < df["rr_ms"]).sum()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(df[["rl_ms", "heft_ms", "rr_ms", "rl_vs_heft"]]
          .describe().round(4).to_string())
    print(f"\n  RL beats HEFT : {n_better_heft}/{len(df)} graphs "
          f"({100*n_better_heft/len(df):.1f} %)")
    print(f"  RL beats RR   : {n_better_rr}/{len(df)} graphs "
          f"({100*n_better_rr/len(df):.1f} %)")
    print("=" * 65)

    out_csv = "evaluation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to '{out_csv}'")

    if _HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Scatter: RL vs HEFT
        ax = axes[0]
        ax.scatter(df["heft_ms"], df["rl_ms"], alpha=0.6, s=25, c="steelblue")
        lim = max(df["heft_ms"].max(), df["rl_ms"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x  (equal)")
        ax.set_xlabel("HEFT Makespan"); ax.set_ylabel("RL Makespan")
        ax.set_title("RL vs HEFT"); ax.legend()

        # Ratio histogram
        ax = axes[1]
        ax.hist(df["rl_vs_heft"], bins=25, edgecolor="black", color="steelblue")
        ax.axvline(1.0, color="red", linestyle="--", lw=1.5, label="HEFT = 1.0")
        ax.set_xlabel("RL / HEFT ratio"); ax.set_ylabel("Count")
        ax.set_title("RL/HEFT Ratio Distribution"); ax.legend()

        # Makespan by graph size
        ax = axes[2]
        ax.scatter(df["n_nodes"], df["rl_vs_heft"],   alpha=0.6, s=25, label="RL/HEFT")
        ax.axhline(1.0, color="red", linestyle="--", lw=1, label="HEFT baseline")
        ax.set_xlabel("Graph size (nodes)"); ax.set_ylabel("RL / HEFT ratio")
        ax.set_title("Ratio vs Graph Size"); ax.legend()

        plt.tight_layout()
        out_png = "evaluation_plot.png"
        plt.savefig(out_png, dpi=150)
        print(f"Plot saved to '{out_png}'")
    else:
        print("[evaluate] matplotlib not available — skipping plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(config.CHECKPOINT_DIR, "best_model"),
        help="Path to the saved MaskablePPO checkpoint (without .zip extension)",
    )
    args = parser.parse_args()
    main(args.checkpoint)
