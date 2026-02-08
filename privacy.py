"""
privacy.py

Differential privacy utilities for federated aggregation.

Provides:
- compute_noise_std: Gaussian mechanism std calculation given clip, epsilon, delta
- clip_update: per-client vector clipping to L2 bound
- secure_aggregate: placeholder for server-side aggregation that adds Gaussian noise
- Note: For production use, integrate with a secure aggregator (e.g., secure multi-party sum)
  and use a formal accountant (Moments Accountant / RDP) for privacy accounting.
"""

from typing import Tuple, Optional
import math
import torch


def compute_noise_std(clip: float, epsilon: float, delta: float) -> float:
    """
    Compute Gaussian mechanism standard deviation sigma for given L2 clipping bound C,
    epsilon, and delta using the analytic bound:
        sigma = C * sqrt(2 * ln(1.25/delta)) / epsilon
    This is a common conservative bound. For tighter accounting, use RDP/analytic accountant.
    """
    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be positive")
    sigma = clip * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    return sigma


def clip_vector(vec: torch.Tensor, clip: float) -> torch.Tensor:
    """
    L2 clip a vector to norm 'clip'.
    """
    norm = torch.norm(vec)
    if norm <= clip:
        return vec
    return vec * (clip / (norm + 1e-12))


def secure_aggregate(updates: list, weights: Optional[list] = None, clip: float = 1.0, epsilon: float = 1.0, delta: float = 1e-5) -> torch.Tensor:
    """
    Aggregate a list of client update tensors (all same shape).
    Steps:
        - optionally clip each update to L2 norm bound
        - average with weights
        - add Gaussian noise with std computed from clip, eps, delta

    Returns aggregated tensor.
    Note: This is a simple centralized simulation of DP aggregation; real federated DP should
    use per-client clipping and a secure aggregation primitive to hide individual updates.
    """
    if len(updates) == 0:
        raise ValueError("No updates to aggregate")
    device = updates[0].device
    stacked = torch.stack([clip_vector(u, clip) for u in updates], dim=0)  # (K, ...)
    if weights is None:
        weights_t = torch.ones(len(updates), device=device) / len(updates)
    else:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        weights_t = w / (w.sum() + 1e-12)
    # weighted average
    agg = (weights_t.view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(dim=0)
    # add Gaussian noise
    sigma = compute_noise_std(clip, epsilon, delta)
    noise = torch.normal(mean=0.0, std=sigma, size=agg.shape, device=device)
    agg_noisy = agg + noise
    return agg_noisy
