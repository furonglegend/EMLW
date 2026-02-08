"""
compression.py

Two-mode compression utilities for federated updates:
- sparsify_topk: keep top-k absolute values (structured/unstructured)
- quantize_tensor / dequantize_tensor: uniform quantization with optional stochastic rounding
- residual compensation helper to maintain error feedback across rounds

This is a lightweight implementation intended for testing and prototyping.
"""

from typing import Tuple, Dict, Any
import torch
import numpy as np


def sparsify_topk(tensor: torch.Tensor, fraction: float = 0.01) -> Dict[str, torch.Tensor]:
    """
    Keep top-k absolute elements of the tensor and return a sparse representation.
    Returns a dict: {"indices": idx_tensor, "values": val_tensor, "shape": tensor.shape}
    Indices are flattened indices (long).
    """
    flat = tensor.flatten()
    k = max(1, int(round(flat.numel() * fraction)))
    if k >= flat.numel():
        idx = torch.arange(flat.numel(), device=flat.device)
        vals = flat
    else:
        vals_abs, idx = torch.topk(flat.abs(), k=k, largest=True)
        vals = flat[idx]
    return {"indices": idx.long(), "values": vals, "shape": torch.tensor(tensor.shape)}


def decompress_sparse(sparse: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Reconstruct dense tensor from sparse dict returned by sparsify_topk.
    """
    shape = tuple(int(x.item()) for x in sparse["shape"])
    flat = torch.zeros(np.prod(shape), device=sparse["values"].device, dtype=sparse["values"].dtype)
    flat[sparse["indices"]] = sparse["values"]
    return flat.view(shape)


def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8, stochastic: bool = False) -> Dict[str, Any]:
    """
    Uniform symmetric quantization to num_bits (including sign).
    Returns a dict with scale and quantized integers.
    Note: this simple quantizer scales by max abs value.
    """
    device = tensor.device
    t = tensor.detach()
    max_val = t.abs().max().item() + 1e-12
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = max_val / qmax
    if scale == 0:
        q = torch.zeros_like(t, dtype=torch.int32)
    else:
        scaled = t / scale
        if stochastic:
            # stochastic rounding
            frac = scaled - scaled.floor()
            rnd = torch.rand_like(frac)
            q = (scaled.floor() + (rnd < frac).int()).clamp(qmin, qmax).to(torch.int32)
        else:
            q = scaled.round().clamp(qmin, qmax).to(torch.int32)
    return {"q": q, "scale": float(scale), "shape": torch.tensor(tensor.shape)}


def dequantize_tensor(qdict: Dict[str, Any]) -> torch.Tensor:
    """
    Dequantize to float tensor using stored scale.
    """
    q = qdict["q"]
    scale = qdict["scale"]
    shape = tuple(int(x.item()) for x in qdict["shape"])
    return (q.to(torch.float32) * float(scale)).view(shape)


class ResidualCompensator:
    """
    Simple residual compensator for error-feedback compression.
    Stores residual per-client (keyed) and applies compensation on next update.
    """
    def __init__(self):
        self._residuals = {}

    def apply_and_update(self, client_id: str, tensor: torch.Tensor, compressed_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply residual to outgoing tensor, and update stored residual based on compression error.
        - tensor: original tensor to send
        - compressed_tensor: decompressed tensor that would be sent (approximation)
        Returns: compensated tensor to actually send (tensor + residual)
        """
        prev = self._residuals.get(client_id)
        if prev is None:
            prev = torch.zeros_like(tensor)
        compensated = tensor + prev
        # Compute new residual: compensated - compressed_tensor (error)
        new_residual = compensated - compressed_tensor
        self._residuals[client_id] = new_residual.detach()
        return compensated

    def clear(self, client_id: str):
        if client_id in self._residuals:
            del self._residuals[client_id]
