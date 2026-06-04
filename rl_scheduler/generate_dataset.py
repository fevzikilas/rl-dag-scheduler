# generate_dataset.py — synthetic layered-DAG dataset generator
#
# Current dataset (DL-Arch-2026) has mean parallelism 2.19 and depth 337 —
# too sequential for RL to improve over HEFT.
#
# This generator produces wide, shallow DAGs where HEFT is genuinely suboptimal:
#   - n_layers:    Uniform[8, 30]
#   - layer_width: Uniform[4, 12]  →  parallelism (nodes/depth) typically 5–9×
#   - Task flops:  LogNormal  →  mean ≈ 1 GFLOP, range 0.05–50 GFLOP
#   - mem_byte:    LogNormal  →  mean ≈ 50 MB
#   - transfer_byte: LogNormal → mean ≈ 5 MB
#
# Usage:
#   python -m rl_scheduler.generate_dataset
#
# Writes graphs_cache.pkl (same path data_loader.py reads from).
# Run this ONCE before training; data_loader.py auto-generates if cache missing.

import os
import pickle
import random

import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "graphs_cache.pkl")
N_GRAPHS  = 500
SEED      = 42

# LogNormal parameters: E[X] = exp(mu + sigma^2/2)
# Target flops  ~1e9  → mu = ln(1e9) - 1.5^2/2 = 20.723 - 1.125 = 19.598
# Target mem    ~5e7  → mu = ln(5e7) - 1.125     = 17.727 - 1.125 = 16.602
# Target xfer   ~5e6  → mu = ln(5e6) - 1.125     = 15.425 - 1.125 = 14.300
_FLOPS_MU  = 19.598
_MEM_MU    = 16.602
_XFER_MU   = 14.300
_SIGMA     = 1.5      # shared sigma → range roughly 1/20× to 20× mean


# ---------------------------------------------------------------------------
def make_layered_dag(
    rng:    random.Random,
    np_rng: np.random.Generator,
) -> nx.DiGraph:
    """Build one random layered DAG with variable width and skip edges."""
    n_layers = rng.randint(8, 30)
    g        = nx.DiGraph()
    layers: list[list[str]] = []
    node_id  = 0

    # ---- create nodes ---------------------------------------------------- #
    for _ in range(n_layers):
        width = rng.randint(4, 12)
        layer: list[str] = []
        for _ in range(width):
            name = f"t{node_id}"
            g.add_node(
                name,
                flops         = float(np_rng.lognormal(_FLOPS_MU, _SIGMA)),
                mem_byte      = float(np_rng.lognormal(_MEM_MU,   _SIGMA)),
                transfer_byte = float(np_rng.lognormal(_XFER_MU,  _SIGMA)),
            )
            layer.append(name)
            node_id += 1
        layers.append(layer)

    # ---- layer-to-layer edges -------------------------------------------- #
    for i in range(n_layers - 1):
        cur, nxt = layers[i], layers[i + 1]

        # Guarantee every node in cur has ≥1 outgoing edge
        for src in cur:
            g.add_edge(src, rng.choice(nxt))

        # Guarantee every node in nxt has ≥1 incoming edge
        for dst in nxt:
            if g.in_degree(dst) == 0:
                g.add_edge(rng.choice(cur), dst)

        # Density fill: ~25% of remaining possible edges
        for src in cur:
            for dst in nxt:
                if not g.has_edge(src, dst) and rng.random() < 0.25:
                    g.add_edge(src, dst)

    # ---- skip-layer edges (layer i → i+2) -------------------------------- #
    for i in range(n_layers - 2):
        if rng.random() < 0.20:
            src = rng.choice(layers[i])
            dst = rng.choice(layers[i + 2])
            if not g.has_edge(src, dst):
                g.add_edge(src, dst)

    return g


# ---------------------------------------------------------------------------
def generate_all(n: int = N_GRAPHS, seed: int = SEED) -> list[nx.DiGraph]:
    """Generate n synthetic layered DAGs and return as a list."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    graphs: list[nx.DiGraph] = []
    attempts = 0

    while len(graphs) < n:
        g = make_layered_dag(rng, np_rng)
        attempts += 1
        if nx.is_directed_acyclic_graph(g) and len(g) >= 10:
            g.graph["model_id"] = f"synthetic_{len(graphs)}"
            graphs.append(g)

    print(f"  Generated {n} valid DAGs in {attempts} attempts.")
    return graphs


def print_stats(graphs: list[nx.DiGraph]) -> None:
    depths       = [nx.dag_longest_path_length(g) for g in graphs]
    node_counts  = [len(g) for g in graphs]
    parallelisms = [nc / max(d, 1) for nc, d in zip(node_counts, depths)]
    print(f"  Graphs:      {len(graphs)}")
    print(f"  Node count:  {min(node_counts)}–{max(node_counts)}"
          f"  (mean {np.mean(node_counts):.1f})")
    print(f"  Depth:       {min(depths)}–{max(depths)}"
          f"  (mean {np.mean(depths):.1f})")
    print(f"  Parallelism: {min(parallelisms):.2f}–{max(parallelisms):.2f}"
          f"  (mean {np.mean(parallelisms):.2f})")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Generating {N_GRAPHS} synthetic layered DAGs (seed={SEED}) …")
    graphs = generate_all()
    print_stats(graphs)
    save_path = os.path.normpath(SAVE_PATH)
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"  Saved → '{save_path}'")
    print("Done. Run `python -m rl_scheduler.pretrain` then `python -m rl_scheduler.train`.")
