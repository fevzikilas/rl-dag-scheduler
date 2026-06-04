"""GNN encoder and policy feature extractor."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

# PyTorch Geometric can segfault on Windows; use mean aggregation fallback
_HAS_PYG = False
print("[gnn_policy] Using mean aggregation GNN.")


class _GATLayer(nn.Module):
    """Single GAT layer (requires PyG)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        per_head = out_dim // config.N_GAT_HEADS
        self.gat = GATConv(
            in_dim, per_head,
            heads=config.N_GAT_HEADS,
            dropout=config.GAT_DROPOUT,
            add_self_loops=False,
        )

    def forward(self, h, edge_index):
        return self.gat(h, edge_index)   # returns (n, out_dim)


class _SimpleMPLayer(nn.Module):
    """Mean-aggregation message-passing layer (no PyG dependency)."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, h, edge_index):
        n = h.shape[0]
        if edge_index.shape[1] == 0:
            agg = torch.zeros_like(h)
        else:
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros_like(h)
            count = torch.zeros(n, 1, device=h.device, dtype=h.dtype)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]), h[src])
            count.scatter_add_(0, dst.unsqueeze(1),
                             torch.ones(src.shape[0], 1, device=h.device, dtype=h.dtype))
            agg = agg / count.clamp(min=1.0)

        return F.elu(self.linear(torch.cat([h, agg], dim=-1)))


class NodeEncoder(nn.Module):
    """GNN encoder: transforms node features to embeddings via message passing."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(config.NODE_FEAT_DIM, config.HIDDEN_DIM)

        if _HAS_PYG:
            self.layers = nn.ModuleList([
                _GATLayer(config.HIDDEN_DIM, config.HIDDEN_DIM)
                for _ in range(config.N_GAT_LAYERS)
            ])
        else:
            self.layers = nn.ModuleList([
                _SimpleMPLayer(config.HIDDEN_DIM)
                for _ in range(config.N_GAT_LAYERS)
            ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(config.HIDDEN_DIM) for _ in range(config.N_GAT_LAYERS)
        ])
        self.dropout = nn.Dropout(config.GAT_DROPOUT)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for layer, norm in zip(self.layers, self.norms):
            h = norm(layer(h, edge_index) + h)
            h = self.dropout(h)
        return h


class HPCFeaturesExtractor(BaseFeaturesExtractor):
    """Merge task, stream, and GPU state observations into joint feature vector."""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        task_dim = observation_space["task_emb"].shape[0]
        proc_dim = observation_space["proc_feat"].shape[0]
        gpu_dim = observation_space["gpu_state"].shape[0]

        self.task_net = nn.Sequential(
            nn.Linear(task_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.proc_net = nn.Sequential(
            nn.Linear(proc_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.gpu_net = nn.Sequential(
            nn.Linear(gpu_dim, 32), nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(32 + 64 + 32, features_dim), nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        t = self.task_net(observations["task_emb"].float())
        p = self.proc_net(observations["proc_feat"].float())
        g = self.gpu_net(observations["gpu_state"].float())
        return self.combine(torch.cat([t, p, g], dim=-1))
