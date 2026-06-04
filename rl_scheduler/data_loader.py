"""Load computation graphs from cache or HuggingFace."""

import random
import pickle
import json
import numpy as np
import networkx as nx

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "graphs_cache.pkl")
HF_DATASET = "nieche/DL-Architectural-DAGs-2026"


def _download_and_cache():
    """Download dataset from HuggingFace Hub, convert to NetworkX, save locally."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"Downloading {HF_DATASET} from HuggingFace Hub …")
    ds = load_dataset(HF_DATASET, trust_remote_code=True)

    graphs = []
    split = ds.keys() if isinstance(ds, dict) else ["train"]
    for split_name in split:
        subset = ds[split_name] if isinstance(ds, dict) else ds
        for example in subset:
            g = nx.DiGraph()

            if "edges" in example:
                edges = example["edges"]
                if isinstance(edges, str):
                    edges = json.loads(edges)
                for src, dst in edges:
                    g.add_edge(str(src), str(dst))

            if "nodes" in example:
                nodes = example["nodes"]
                if isinstance(nodes, str):
                    nodes = json.loads(nodes)
                for node_id, attrs in nodes.items():
                    if isinstance(attrs, dict):
                        g.add_node(str(node_id), **attrs)
                    else:
                        g.add_node(str(node_id))

            if "model_id" in example:
                g.graph["model_id"] = example["model_id"]

            if nx.is_directed_acyclic_graph(g) and len(g) > 0:
                graphs.append(g)

    print(f"  Loaded {len(graphs)} graphs")

    cache_path = os.path.normpath(CACHE_FILE)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"  Saved to '{cache_path}'")

    return graphs


def load_all_graphs(seed: int = config.SEED):
    """Load all graphs from cache, split into train/test."""
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
    """Compute node features: [flops, mem, transfer_size, depth, in_degree]."""
    nodes = list(nx.topological_sort(g))
    n = len(nodes)

    depth = {}
    for nd in nx.topological_sort(g):
        preds = list(g.predecessors(nd))
        depth[nd] = 0 if not preds else max(depth[p] for p in preds) + 1

    feat = np.zeros((n, config.NODE_FEAT_DIM), dtype=np.float32)
    for i, nd in enumerate(nodes):
        d = g.nodes[nd]
        feat[i, 0] = d.get("flops", 0.0)
        feat[i, 1] = d.get("mem_byte", 0.0)
        feat[i, 2] = d.get("transfer_byte", 0.0)
        feat[i, 3] = depth[nd]
        feat[i, 4] = g.in_degree(nd)

    col_max = feat.max(axis=0)
    col_max[col_max == 0] = 1.0
    feat /= col_max

    return feat, nodes


def build_edge_index(g: nx.DiGraph, node_idx: dict) -> np.ndarray:
    """Build COO-format edge index with self-loops."""
    src, dst = [], []

    for u, v in g.edges():
        src.append(node_idx[u])
        dst.append(node_idx[v])

    for i in range(len(g)):
        src.append(i)
        dst.append(i)

    return np.array([src, dst], dtype=np.int64)
