# data_loader.py — load HuggingFace dataset into NetworkX DAGs

import random
import collections
import pickle
import numpy as np
import networkx as nx

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

HF_DATASET  = "nieche/DL-Architectural-DAGs-2026"
# Local cache so we never import `datasets` (→ TF) during training
CACHE_FILE  = os.path.join(os.path.dirname(__file__), "..", "graphs_cache.pkl")


def _download_and_cache():
    """Download from HuggingFace and save to a local pickle for future runs."""
    # TF is loaded transitively by `datasets`. Keep it on CPU so it doesn't
    # interfere with PyTorch's GPU context.
    os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    from datasets import load_dataset
    print("  Downloading from HuggingFace (one-time) …")
    ds = load_dataset(HF_DATASET, "nodes", split="train")

    groups: dict = collections.defaultdict(list)
    for row in ds:
        groups[row["model_id"]].append(row)

    print(f"  Found {len(groups)} model graphs.  Building nx.DiGraph objects …")
    graphs = []
    skipped = 0
    for model_id, rows in groups.items():
        g = nx.DiGraph()
        g.graph["model_id"] = model_id
        for row in rows:
            node_name = row["node_name"]
            g.add_node(
                node_name,
                flops         = float(row["flops"]),
                mem_byte      = float(row["mem_byte"]),
                transfer_byte = float(row["transfer_byte"]),
            )
            for parent in row["parents"]:
                if parent:
                    g.add_edge(parent, node_name)
        if len(g) == 0 or not nx.is_directed_acyclic_graph(g):
            skipped += 1
            continue
        graphs.append(g)

    print(f"  Valid graphs: {len(graphs)}  |  Skipped: {skipped}")
    cache_path = os.path.normpath(CACHE_FILE)
    with open(cache_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"  Graphs cached to '{cache_path}'")
    return graphs


def load_all_graphs(seed: int = config.SEED):
    """Return (train_graphs, test_graphs). Downloads and caches on first call."""
    cache_path = os.path.normpath(CACHE_FILE)
    if os.path.exists(cache_path):
        print(f"Loading graphs from cache '{cache_path}' …")
        with open(cache_path, "rb") as f:
            graphs = pickle.load(f)
        print(f"  {len(graphs)} graphs loaded.")
    else:
        print("No cache found — downloading from HuggingFace …")
        graphs = _download_and_cache()

    rng = random.Random(seed)
    rng.shuffle(graphs)
    split = int(len(graphs) * config.TRAIN_RATIO)
    train_graphs = graphs[:split]
    test_graphs  = graphs[split:]

    print(f"  Train: {len(train_graphs)}  |  Test: {len(test_graphs)}")
    return train_graphs, test_graphs


def compute_node_features(g: nx.DiGraph):
    """Build (n, NODE_FEAT_DIM) float32 feature matrix in topological order.

    Columns: flops, mem_byte, transfer_byte, topo_depth, in_degree
    Returns (feat, nodes).
    """
    nodes = list(nx.topological_sort(g))
    n     = len(nodes)

    depth: dict = {}
    for nd in nx.topological_sort(g):
        preds = list(g.predecessors(nd))
        depth[nd] = 0 if not preds else max(depth[p] for p in preds) + 1

    feat = np.zeros((n, config.NODE_FEAT_DIM), dtype=np.float32)
    for i, nd in enumerate(nodes):
        d = g.nodes[nd]
        feat[i, 0] = d.get("flops",         0.0)
        feat[i, 1] = d.get("mem_byte",      0.0)
        feat[i, 2] = d.get("transfer_byte", 0.0)
        feat[i, 3] = depth[nd]
        feat[i, 4] = g.in_degree(nd)

    col_max = feat.max(axis=0)
    col_max[col_max == 0] = 1.0
    feat /= col_max

    return feat, nodes


def build_edge_index(g: nx.DiGraph, node_idx: dict) -> np.ndarray:
    """Return (2, E) COO edge index with self-loops added for each node."""
    src, dst = [], []

    for u, v in g.edges():
        src.append(node_idx[u])
        dst.append(node_idx[v])

    for i in range(len(g)):
        src.append(i)
        dst.append(i)

    return np.array([src, dst], dtype=np.int64)
