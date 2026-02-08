"""
cde.py

Clinical Dependency Encoder (CDE) skeleton.

Provides:
- A lightweight graph-attention layer (GAT-like) implemented with dense adjacency
  (avoids external graph libraries to keep the dependency surface small).
- A CDE module that stacks multiple GAT layers, supports dropout and residuals.
- Utility to build adjacency from pairwise similarity matrix (or accept precomputed adj).

Notes:
- For large graphs or sparse adjacency, replace dense ops with sparse implementations
  or use PyTorch Geometric / DGL for efficiency.
- TODO: Replace attention scoring function with the exact paper formulation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGraphAttentionLayer(nn.Module):
    """
    Single-head dense GAT-style layer.
    Input: node features X (N, F_in), adjacency A (N, N) with 0/1 or weights.
    Output: new node features (N, F_out)
    """
    def __init__(self, in_dim: int, out_dim: int, leaky_relu_negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (N, F_in)
        adj: (N, N) adjacency matrix (0/1 or weights). Self-loops should be included if desired.
        returns: (N, F_out)
        """
        h = self.W(x)  # (N, F_out)
        # compute attention scores e_ij = a([Wh_i || Wh_j]) but implemented as a_src(Wh_i)+a_dst(Wh_j)
        f_src = self.a_src(h)  # (N,1)
        f_dst = self.a_dst(h)  # (N,1)
        e = f_src + f_dst.T     # broadcasting -> (N, N)
        e = self.leaky_relu(e)
        # mask with adjacency (set to -inf where no edge)
        if adj is not None:
            neg_inf = -9e15
            e = torch.where(adj > 0, e, torch.full_like(e, neg_inf))
        # attention coefficients
        alpha = F.softmax(e, dim=1)  # row-normalized
        out = alpha @ h  # (N, F_out)
        return out


class CDE(nn.Module):
    """
    Clinical Dependency Encoder: stack of DenseGraphAttentionLayer with optional residuals and layernorm.
    Expected usage:
        cde = CDE(node_feat_dim, hidden_dims=[64, 64], dropout=0.1)
        z = cde(x, adj)
    """
    def __init__(self, in_dim: int, hidden_dims: Optional[list] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(DenseGraphAttentionLayer(cur, h))
            cur = h
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden_dims])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (N, F)
        adj: (N, N) adjacency matrix; if None, fully connected attention will be used.
        returns: (N, F_out) where F_out = last hidden dim
        """
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, adj)
            h = norm(h + self.dropout(h_new))  # residual + norm
        return h


# Helper: adjacency builder
def adjacency_from_similarity(sim: torch.Tensor, threshold: Optional[float] = None, topk: Optional[int] = None) -> torch.Tensor:
    """
    Build adjacency matrix from pairwise similarity matrix 'sim' (N, N).
    Either threshold (float) or topk (int) must be provided. Output is (N, N) float matrix.
    """
    if threshold is not None:
        adj = (sim >= threshold).float()
    elif topk is not None:
        # for each row, keep topk values
        N = sim.shape[0]
        adj = torch.zeros_like(sim)
        if topk <= 0:
            return adj
        vals, idxs = torch.topk(sim, k=min(topk, sim.shape[1]), dim=1)
        row_idx = torch.arange(N).unsqueeze(1).expand(-1, idxs.size(1))
        adj[row_idx.flatten(), idxs.flatten()] = 1.0
    else:
        # fallback: fully connected
        adj = torch.ones_like(sim)
    # ensure self-loops
    adj.fill_diagonal_(1.0)
    return adj
