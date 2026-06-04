"""Single-GPU task scheduling environment."""

import warnings
import random
from collections import deque

import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config
from baselines import heft as _heft


class HPCClusterEnv(gym.Env):
    """List-scheduling MDP: assign tasks to GPU streams."""

    metadata = {"render_modes": []}

    def __init__(self, graphs: list, device: str = "cpu",
                 gpu_memory: float = None, n_streams: int = None):
        super().__init__()
        self.graphs           = graphs
        self.device           = device
        self.gpu_memory_fixed = gpu_memory    # None → randomise each episode
        self.n_streams_fixed  = n_streams     # None → randomise each episode

        # Fixed to MAX_STREAMS so Gymnasium shapes stay constant across episodes.
        # Inactive slots are zeroed and masked at runtime.
        self.action_space = spaces.Discrete(config.MAX_STREAMS)
        self.observation_space = spaces.Dict({
            # 6 structural features per task: rank_u, compute, in/out degree, mem, transfer
            "task_emb":  spaces.Box(-np.inf, np.inf,
                                    (config.TASK_FEAT_DIM,), dtype=np.float32),
            # 4 features per stream: [avail_norm, eft_norm, load_norm, active_flag]
            "proc_feat": spaces.Box(-np.inf, np.inf,
                                    (config.MAX_STREAMS * 4,), dtype=np.float32),
            # [vram_used_norm, gpu_cap_norm, n_streams_norm, progress_norm, queue_depth_norm]
            "gpu_state": spaces.Box(0.0, 1.0, (5,), dtype=np.float32),
            "valid_mask": spaces.MultiBinary(config.MAX_STREAMS),
        })

        # episode state
        self.g                   = None
        self.nodes               = None
        self.node_idx: dict      = {}
        self.task_features       = None   # (n, TASK_FEAT_DIM) structural features
        self.rank_u_vals: dict   = {}     # node_name → raw rank_u (for queue sorting)
        self.ready_queue         = None
        self.task_status         = None
        self.task_finish_time    = None
        self.task_proc_assigned  = None
        self.proc_avail_time     = None
        self.proc_n_tasks        = None
        self.n_assigned          = 0
        self.n_total             = 0
        self.lower_bound         = 1.0
        self.heft_makespan       = 1.0   # HEFT baseline for this episode's graph+hardware
        self.gpu_capacity        = config.GPU_MEMORY_DEFAULT
        self.vram_used           = 0.0
        self.vram_live: dict     = {}  # task_idx → output bytes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.g = random.choice(self.graphs)
        self.nodes    = list(nx.topological_sort(self.g))
        self.node_idx = {nd: i for i, nd in enumerate(self.nodes)}
        n = len(self.nodes)

        if self.gpu_memory_fixed is not None:
            self.gpu_capacity = float(self.gpu_memory_fixed)
        else:
            self.gpu_capacity = float(
                np.random.uniform(config.GPU_MEMORY_MIN, config.GPU_MEMORY_MAX)
            )

        if self.n_streams_fixed is not None:
            self.n_streams = int(self.n_streams_fixed)
        else:
            self.n_streams = int(
                np.random.randint(config.N_STREAMS_MIN, config.N_STREAMS_MAX + 1)
            )
        self.stream_flops = config.GPU_FLOPS_CAP / self.n_streams

        # Pre-compute rank_u so ready_queue can be priority-sorted each step.
        # rank_u[v] = w[v] + max(rank_u[successors]) — identical to HEFT's priority.
        self.rank_u_vals = {}
        for nd in reversed(list(nx.topological_sort(self.g))):
            compute  = self.g.nodes[nd].get("flops", 0.0) / self.stream_flops
            succ_max = max((self.rank_u_vals[s] for s in self.g.successors(nd)), default=0.0)
            self.rank_u_vals[nd] = compute + succ_max

        # Structural features computed after stream_flops is set (rank_u depends on it)
        self.task_features = self._compute_task_features()

        # always MAX_STREAMS wide; slots >= n_streams are inactive (zeroed, masked)
        self.proc_avail_time = np.zeros(config.MAX_STREAMS, dtype=np.float64)
        self.proc_n_tasks    = np.zeros(config.MAX_STREAMS, dtype=np.int32)

        self.task_status         = np.zeros(n, dtype=np.int8)
        self.task_finish_time    = np.zeros(n, dtype=np.float64)
        self.task_proc_assigned  = np.full(n, -1, dtype=np.int32)

        self.vram_used  = 0.0
        self.vram_live  = {}

        self.ready_queue = deque()
        for i, nd in enumerate(self.nodes):
            if self.g.in_degree(nd) == 0:
                self.task_status[i] = 1
                self.ready_queue.append(i)
        # Sort initial queue by rank_u descending (most urgent task first)
        self._sort_ready_queue()

        self.n_assigned  = 0
        self.n_total     = n
        self.lower_bound = self._compute_lower_bound()
        self.heft_makespan = _heft(
            self.g, gpu_memory=self.gpu_capacity, n_streams=self.n_streams
        )["makespan"]

        self._check_deadlock()
        return self._get_obs(), {}

    def step(self, action: int):
        assert len(self.ready_queue) > 0, "step() called with empty ready queue!"
        stream_id = int(action)
        task_idx  = self.ready_queue[0]
        task_node = self.nodes[task_idx]

        out_bytes = self.g.nodes[task_node].get("transfer_byte", 0.0)
        if (self.vram_used + out_bytes) > self.gpu_capacity:
            # VRAM full: mask fallback opened this stream, proceed without charging VRAM
            # (prevents infinite loop; a warning was already issued by get_action_mask)
            out_bytes = 0.0

        # no inter-stream comm cost (same device)
        earliest_start = self.proc_avail_time[stream_id]
        for parent in self.g.predecessors(task_node):
            p_idx    = self.node_idx[parent]
            p_finish = self.task_finish_time[p_idx]
            earliest_start = max(earliest_start, p_finish)

        flops        = self.g.nodes[task_node].get("flops", 0.0)
        compute_time = max(flops / self.stream_flops, 1e-9)
        finish_time  = earliest_start + compute_time

        self.vram_used += out_bytes
        self.vram_live[task_idx] = out_bytes

        # 
        # 
        # 
        # normalize by lower_bound so idle penalty is on same scale as terminal reward


        idle_created = max(0.0, earliest_start - self.proc_avail_time[stream_id])
        idle_norm    = idle_created / max(self.lower_bound, 1e-9)
        reward       = -idle_norm * config.IDLE_PENALTY

        self.task_finish_time[task_idx]   = finish_time
        self.task_proc_assigned[task_idx] = stream_id
        self.task_status[task_idx]        = 2
        self.proc_avail_time[stream_id]   = finish_time
        self.proc_n_tasks[stream_id]     += 1
        self.n_assigned                  += 1
        self.ready_queue.popleft()

        for child in self.g.successors(task_node):
            c_idx = self.node_idx[child]
            if self.task_status[c_idx] == 0:
                if all(
                    self.task_status[self.node_idx[p]] == 2
                    for p in self.g.predecessors(child)
                ):
                    self.task_status[c_idx] = 1
                    self.ready_queue.append(c_idx)

        # Re-sort by rank_u so the most urgent task is always at the front
        if len(self.ready_queue) > 1:
            self._sort_ready_queue()

        # free output tensors of predecessors whose consumers are all done
        for parent in self.g.predecessors(task_node):
            p_idx = self.node_idx[parent]
            if all(
                self.task_status[self.node_idx[s]] == 2
                for s in self.g.successors(parent)
            ):
                freed = self.vram_live.pop(p_idx, 0.0)
                self.vram_used = max(0.0, self.vram_used - freed)

        terminated = (self.n_assigned == self.n_total)
        info: dict = {}

        if terminated:
            makespan = float(np.max(self.task_finish_time))

            #
            #     HEFT-relative terminal reward -> positive when RL beats HEFT, negative otherwise 
            #
            #
            #
            
            terminal_r = float(np.clip(
                (self.heft_makespan - makespan) / max(self.heft_makespan, 1e-9) * 10,
                -10.0, 10.0
            ))
            reward += terminal_r
            info = {
                "makespan":        makespan,
                "heft_makespan":   self.heft_makespan,
                "lower_bound":     self.lower_bound,
                "ratio":           makespan / max(self.lower_bound, 1e-9),
                "rl_vs_heft":      makespan / max(self.heft_makespan, 1e-9),
                "gpu_capacity_gb": self.gpu_capacity / 1e9,
                "n_streams":       self.n_streams,
            }
        else:
            self._check_deadlock()

        obs = self._get_obs() if not terminated else self._zero_obs()
        return obs, float(reward), terminated, False, info

    def get_action_mask(self) -> np.ndarray:
        """Shape (MAX_STREAMS,); True if stream is active and VRAM fits."""
        mask = np.zeros(config.MAX_STREAMS, dtype=bool)

        if not self.ready_queue:
            mask[:self.n_streams] = True
            return mask

        task_idx  = self.ready_queue[0]
        task_node = self.nodes[task_idx]
        out_bytes = self.g.nodes[task_node].get("transfer_byte", 0.0)

        fits = (self.vram_used + out_bytes) <= self.gpu_capacity
        mask[:self.n_streams] = fits

        #  - if nothing fits, open all active streams to avoid a deadlock


        if not mask.any():
            warnings.warn(
                f"VRAM full ({self.vram_used/1e9:.2f}/{self.gpu_capacity/1e9:.2f} GB) "
                f"for task '{task_node}' — allowing all active streams."
            )
            mask[:self.n_streams] = True
        return mask

    def render(self):
        pass

    def _get_obs(self) -> dict:
        if not self.ready_queue:
            return self._zero_obs()

        task_idx  = self.ready_queue[0]
        task_node = self.nodes[task_idx]
        task_emb  = self.task_features[task_idx]

        task_flops   = self.g.nodes[task_node].get("flops", 0.0)
        task_compute = max(task_flops / self.stream_flops, 1e-9)

        # earliest data-ready time (max finish of all parents)

        parent_max = max(
            (self.task_finish_time[self.node_idx[p]]
             for p in self.g.predecessors(task_node)),
            default=0.0
        )

        # EFT per stream -> when would this task finish if assigned here

        eft = np.array([
            max(self.proc_avail_time[s], parent_max) + task_compute
            for s in range(self.n_streams)
        ], dtype=np.float64)

        time_scale = max(float(eft.max()), 1e-9)
        max_load   = max(int(self.n_total), 1)

        # 4 features per slot: [ avail_norm , eft_norm, load_norm, active_flag ]

        proc_feat = np.zeros(config.MAX_STREAMS * 4, dtype=np.float32)
        for s in range(self.n_streams):
            base = s * 4
            proc_feat[base]     = self.proc_avail_time[s] / time_scale
            proc_feat[base + 1] = eft[s] / time_scale
            proc_feat[base + 2] = self.proc_n_tasks[s] / max_load
            proc_feat[base + 3] = 1.0

        queue_depth = len(self.ready_queue) / max(self.n_total, 1)
        gpu_state = np.array([
            min(self.vram_used / max(self.gpu_capacity, 1.0), 1.0),
            min(self.gpu_capacity / config.GPU_MEMORY_MAX, 1.0),
            (self.n_streams - config.N_STREAMS_MIN) /
                max(config.N_STREAMS_MAX - config.N_STREAMS_MIN, 1),
            self.n_assigned / max(self.n_total, 1),
            min(queue_depth, 1.0),
        ], dtype=np.float32)

        valid_mask = self.get_action_mask().astype(np.int8)

        return {"task_emb": task_emb, "proc_feat": proc_feat,
                "gpu_state": gpu_state, "valid_mask": valid_mask}

    def _zero_obs(self) -> dict:
        return {
            "task_emb":   np.zeros(config.TASK_FEAT_DIM, dtype=np.float32),
            "proc_feat":  np.zeros(config.MAX_STREAMS * 4, dtype=np.float32),
            "gpu_state":  np.zeros(5, dtype=np.float32),
            "valid_mask": np.ones(config.MAX_STREAMS, dtype=np.int8),
        }

    def _sort_ready_queue(self):
        """Sort ready queue by rank_u descending — highest priority task first.
        This mirrors HEFT's task ordering, so the agent only needs to learn
        which stream to use (not which task to schedule next)."""
        if len(self.ready_queue) > 1:
            self.ready_queue = deque(
                sorted(self.ready_queue,
                       key=lambda i: self.rank_u_vals[self.nodes[i]],
                       reverse=True)
            )

    def _compute_task_features(self) -> np.ndarray:
        """Per-node structural features that give the policy HEFT-relevant information.

        Features (TASK_FEAT_DIM = 6):
          0  rank_u_norm    — HEFT upward rank: higher = schedule this task sooner
          1  compute_norm   — task compute time relative to the heaviest task
          2  in_degree_norm — number of predecessors / (n-1)
          3  out_degree_norm— number of successors / (n-1)  (fan-out = impact)
          4  mem_norm       — output tensor size / GPU_MEMORY_MAX
          5  transfer_norm  — activation transfer size / GPU_MEMORY_MAX
        """
        n    = len(self.nodes)
        feat = np.zeros((n, config.TASK_FEAT_DIM), dtype=np.float32)

        # Reuse rank_u already computed in reset() — no redundant traversal
        rank_u    = self.rank_u_vals
        max_rank  = max(rank_u.values()) if rank_u else 1.0
        max_flops = max(
            (self.g.nodes[nd].get("flops", 0.0) for nd in self.nodes), default=1.0
        )

        for i, nd in enumerate(self.nodes):
            feat[i, 0] = rank_u[nd] / max(max_rank, 1e-9)
            feat[i, 1] = self.g.nodes[nd].get("flops", 0.0) / max(max_flops, 1e-9)
            feat[i, 2] = self.g.in_degree(nd)  / max(n - 1, 1)
            feat[i, 3] = self.g.out_degree(nd) / max(n - 1, 1)
            feat[i, 4] = self.g.nodes[nd].get("mem_byte",      0.0) / config.GPU_MEMORY_MAX
            feat[i, 5] = self.g.nodes[nd].get("transfer_byte", 0.0) / config.GPU_MEMORY_MAX

        return feat

    def _compute_lower_bound(self) -> float:
        """max(critical_path_time, total_flops / GPU_FLOPS_CAP)"""
        total_stream_cap = self.stream_flops * self.n_streams

        total_flops = sum(
            max(self.g.nodes[nd].get("flops", 0.0), 0.0)
            for nd in self.g.nodes
        )
        lb_parallel = total_flops / max(total_stream_cap, 1e-9)

        cp: dict = {}
        for nd in nx.topological_sort(self.g):
            t_nd = max(self.g.nodes[nd].get("flops", 0.0), 0.0) / self.stream_flops
            preds = list(self.g.predecessors(nd))
            cp[nd] = t_nd if not preds else t_nd + max(cp[p] for p in preds)

        lb_critical = max(cp.values()) if cp else 0.0
        return max(lb_parallel, lb_critical, 1e-9)

    def _check_deadlock(self):
        """Raises if the ready queue is empty with tasks remaining (should never happen)."""
        if not self.ready_queue and self.n_assigned < self.n_total:
            pending = [
                self.nodes[i]
                for i, s in enumerate(self.task_status)
                if s != 2
            ]
            raise RuntimeError(
                f"DEADLOCK detected in graph "
                f"'{self.g.graph.get('model_id', '?')}': "
                f"{len(pending)} tasks remain but ready queue is empty. "
                f"First few: {pending[:5]}"
            )
