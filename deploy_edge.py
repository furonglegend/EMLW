"""
deploy_edge.py

Edge deployment helpers:
- export lightweight parts of model for edge inference (prune projector, keep encoder+classifier)
- simple inference wrapper that loads a checkpoint and runs on numpy input
- mock function to simulate Raspberry Pi constraints (force CPU, limit threads)

This is a helper script; adapt to your deployment pipeline.
"""

import torch
import numpy as np
from typing import Tuple

from medalign.models import MedAlignModel


def build_edge_model(full_model: MedAlignModel, keep_projector: bool = False) -> torch.nn.Module:
    """
    Build a compact model suitable for edge inference.
    If keep_projector False: return encoder + classifier only.
    """
    class EdgeModel(torch.nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, x):
            emb = self.encoder(x)
            return self.classifier(emb)

    return EdgeModel(full_model.encoder, full_model.classifier)


def run_inference(edge_model: torch.nn.Module, numpy_input: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Run inference on a single numpy input shaped (C, T) or batch (B, C, T).
    Returns logits as numpy array.
    """
    edge_model.to(device)
    edge_model.eval()
    if numpy_input.ndim == 2:
        x = numpy_input[None, ...]
    else:
        x = numpy_input
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = edge_model(x_t).cpu().numpy()
    return logits


def simulate_rpi_environment(set_threads: int = 1):
    """
    Simulate Raspberry Pi constraints by limiting OpenMP threads and forcing CPU usage.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = str(set_threads)
    os.environ["MKL_NUM_THREADS"] = str(set_threads)
    # Force CPU by setting device preference in config if needed
