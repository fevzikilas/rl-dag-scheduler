# baselines.py — HEFT and Round Robin schedulers
#
# Both run on a single GPU modelled as n_streams parallel CUDA streams
# sharing one VRAM pool.  Inter-stream comm cost = 0 (same device).
#
# Reference: Topcuoglu, Hariri, Wu (2002). IEEE TPDS.

import numpy as np
import networkx as nx

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config


def _eft_on_stream(
    task_node: str,
    stream_id: int,
    g:         nx.DiGraph,
    task_finish_time:   np.ndarray,
    task_proc_assigned: np.ndarray,
    proc_avail_time:    np.ndarray,
    node_idx:           dict,
    stream_flops:       float,
) -> float:
    """Compute earliest finish time on stream_id. No comm cost (same GPU)."""
    earliest_start = proc_avail_time[stream_id]
    for parent in g.predecessors(task_node):
        p_idx    = node_idx[parent]
        p_finish = task_finish_time[p_idx]
        earliest_start = max(earliest_start, p_finish)

    flops        = g.nodes[task_node].get("flops", 0.0)
    compute_time = max(flops / stream_flops, 1e-9)
    return earliest_start + compute_time


def round_robin(g: nx.DiGraph, gpu_memory: float = None,
                n_streams: int = None) -> dict:
    """Assign tasks in topological order, cycling through streams."""
    gpu_mem      = gpu_memory if gpu_memory is not None else config.GPU_MEMORY_DEFAULT
    n_s          = n_streams  if n_streams  is not None else config.N_STREAMS_DEFAULT
    stream_flops = config.GPU_FLOPS_CAP / n_s

    nodes    = list(nx.topological_sort(g))
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    n        = len(nodes)

    task_finish_time   = np.zeros(n, dtype=np.float64)
    task_proc_assigned = np.full(n, -1, dtype=np.int32)
    proc_avail_time    = np.zeros(n_s, dtype=np.float64)
    assignment         = {}
    vram_used          = 0.0

    for t, nd in enumerate(nodes):
        stream_id = t % n_s
        out_bytes = g.nodes[nd].get("transfer_byte", 0.0)
        if vram_used + out_bytes > gpu_mem:
            stream_id = int(np.argmin(proc_avail_time))
        eft = _eft_on_stream(
            nd, stream_id, g,
            task_finish_time, task_proc_assigned, proc_avail_time, node_idx,
            stream_flops,
        )
        task_finish_time[node_idx[nd]]   = eft
        task_proc_assigned[node_idx[nd]] = stream_id
        proc_avail_time[stream_id]       = eft
        vram_used = max(0.0, vram_used + out_bytes)
        assignment[nd] = stream_id

    return {"makespan": float(np.max(task_finish_time)), "assignment": assignment}


def heft(g: nx.DiGraph, gpu_memory: float = None,
         n_streams: int = None) -> dict:
    """
    HEFT for single-GPU variable-stream scheduling.
    Ranks tasks by upward rank then greedily assigns to the earliest-finish stream.
    """
    gpu_mem      = gpu_memory if gpu_memory is not None else config.GPU_MEMORY_DEFAULT
    n_s          = n_streams  if n_streams  is not None else config.N_STREAMS_DEFAULT
    stream_flops = config.GPU_FLOPS_CAP / n_s

    nodes    = list(nx.topological_sort(g))
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    n        = len(nodes)

    rank_u: dict = {}
    for nd in reversed(nodes):
        flops   = g.nodes[nd].get("flops", 0.0)
        compute = max(flops / stream_flops, 1e-9)
        succs   = list(g.successors(nd))
        rank_u[nd] = compute + (max(rank_u[s] for s in succs) if succs else 0.0)

    priority_list = sorted(g.nodes, key=lambda nd: rank_u[nd], reverse=True)

    task_finish_time   = np.zeros(n, dtype=np.float64)
    task_proc_assigned = np.full(n, -1, dtype=np.int32)
    proc_avail_time    = np.zeros(n_s, dtype=np.float64)
    assignment         = {}
    vram_used          = 0.0

    for nd in priority_list:
        nd_idx    = node_idx[nd]
        out_bytes = g.nodes[nd].get("transfer_byte", 0.0)
        fits_vram = (vram_used + out_bytes) <= gpu_mem

        best_stream = -1
        best_eft    = float("inf")

        for s in range(n_s):
            if not fits_vram:
                continue
            eft = _eft_on_stream(
                nd, s, g,
                task_finish_time, task_proc_assigned, proc_avail_time, node_idx,
                stream_flops,
            )
            if eft < best_eft:
                best_eft    = eft
                best_stream = s

        if best_stream == -1:   # VRAM full — use earliest-available stream
            best_stream = int(np.argmin(proc_avail_time))
            best_eft    = _eft_on_stream(
                nd, best_stream, g,
                task_finish_time, task_proc_assigned, proc_avail_time, node_idx,
                stream_flops,
            )

        task_finish_time[nd_idx]     = best_eft
        task_proc_assigned[nd_idx]   = best_stream
        proc_avail_time[best_stream] = best_eft
        vram_used = max(0.0, vram_used + out_bytes)
        assignment[nd] = best_stream

    return {"makespan": float(np.max(task_finish_time)), "assignment": assignment}
