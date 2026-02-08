"""
pcrl.py

Prototype-Consistent Representation Learning (PCRL) skeleton.

Provides:
- NT-Xent contrastive loss (normalized temperature-scaled cross-entropy)
- Prototype store and simple update rule (moving average)
- Utilities to compute prototype deltas (for upload) and basic compression hooks (top-k sparsify)

TODO / Notes:
- Paper-specific prototype alignment and update rules should replace the placeholders.
- Compression hooks are intentionally minimal; adapt to the two-mode compression scheme described in the paper.
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute NT-Xent loss between two augmented batches z_i and z_j.
    z_i, z_j: (B, D) embeddings (not necessarily normalized)
    Returns average loss scalar.
    """
    z_i = F.normalize(z_i, p=2, dim=1)
    z_j = F.normalize(z_j, p=2, dim=1)
    representations = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim_matrix = torch.matmul(representations, representations.T)  # (2B,2B)
    sim_matrix = sim_matrix / temperature
    B = z_i.shape[0]
    labels = torch.arange(B, device=z_i.device)
    labels = torch.cat([labels + B, labels], dim=0)  # positive indices for each example

    # mask self
    mask = (~torch.eye(2 * B, dtype=torch.bool, device=z_i.device)).float()

    exp_sim = torch.exp(sim_matrix) * mask
    denom = exp_sim.sum(dim=1)
    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = -torch.log(pos_sim / (denom + 1e-12))
    return loss.mean()


class PrototypeStore(nn.Module):
    """
    Prototype store managing K prototypes of dimension D.
    Supports simple moving-average updates and computing deltas for upload.
    """
    def __init__(self, num_prototypes: int, dim: int, momentum: float = 0.9):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.momentum = momentum
        # prototypes are registered as buffers (not trained by optimizer)
        self.register_buffer("prototypes", torch.randn(num_prototypes, dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.01)

    @torch.no_grad()
    def update_with_batch(self, embeddings: torch.Tensor, assignments: torch.Tensor):
        """
        Update prototypes given batch embeddings and assignment indices.
        embeddings: (B, D)
        assignments: (B,) values in [0, num_prototypes)
        Update rule: prototype_k = momentum * prototype_k + (1 - momentum) * mean(embeddings assigned to k)
        """
        device = self.prototypes.device
        for k in range(self.num_prototypes):
            mask = (assignments == k)
            if mask.any():
                mean_emb = embeddings[mask].mean(dim=0)
                self.prototypes[k] = self.momentum * self.prototypes[k] + (1.0 - self.momentum) * mean_emb.to(device)

    def assign(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Assign each embedding to nearest prototype by cosine similarity.
        returns: assignments (B,) indices
        """
        emb_norm = F.normalize(embeddings, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sims = emb_norm @ proto_norm.T  # (B, K)
        assignments = sims.argmax(dim=1)
        return assignments

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes.detach().clone()

    def compute_delta(self, previous_prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute delta to upload: current - previous
        """
        return self.get_prototypes() - previous_prototypes


# Simple compression utilities
def topk_sparsify(tensor: torch.Tensor, fraction: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep top-k absolute values, return indices and values for sparse representation.
    Returns (indices, values) where indices is 1D flattened index tensor and values are corresponding floats.
    """
    flat = tensor.flatten()
    k = max(1, int(round(flat.numel() * fraction)))
    if k >= flat.numel():
        idx = torch.arange(flat.numel(), device=flat.device)
        vals = flat
    else:
        vals, idx = torch.topk(flat.abs(), k=k, largest=True)
        idx = idx.long()
        vals = flat[idx]
    return idx, vals


def decompress_topk(indices: torch.Tensor, values: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Reconstruct dense tensor from topk indices and values.
    """
    flat = torch.zeros(int(torch.prod(torch.tensor(shape))), device=values.device)
    flat[indices] = values
    return flat.view(shape)
