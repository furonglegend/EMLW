"""
cafw.py

Context-Aware Feature Weighting (CAFW) skeleton.

This module provides:
- a small ontology loader (JSON) -> embeddings
- a function to compute per-feature/context weights given an input example
- a callable wrapper that applies weights to features

This is a high-level, pluggable interface: replace weighting logic with the paper's method.
"""

from typing import Dict, Optional, List
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class OntologyEmbedder(nn.Module):
    """
    Simple learnable embedding table for ontology tokens / feature names.
    Expects an ontology JSON mapping feature_name -> token_id or a list of feature names.
    """
    def __init__(self, feature_names: List[str], emb_dim: int = 32):
        super().__init__()
        self.feature_names = list(feature_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        self.emb = nn.Embedding(len(self.feature_names), emb_dim)

    def forward(self, feature_names: List[str]) -> torch.Tensor:
        """
        Return embeddings for the requested feature_names as (num_features, emb_dim)
        """
        idxs = [self.name_to_idx[n] for n in feature_names]
        idxs_tensor = torch.tensor(idxs, dtype=torch.long, device=self.emb.weight.device)
        return self.emb(idxs_tensor)


class CAFW(object):
    """
    CAFW: compute per-feature weights using ontology + simple scoring network.
    Usage:
        cafw = CAFW(feature_names, emb_dim=32, hidden=64)
        weights = cafw.compute_weights(example_features)  # shape (num_features,)
    """
    def __init__(self, feature_names: List[str], emb_dim: int = 32, hidden_dim: int = 64, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = OntologyEmbedder(feature_names, emb_dim=emb_dim).to(self.device)
        # scoring network: maps (feature_embedding + simple stats) -> scalar weight
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim + 2, hidden_dim),  # +2 for example-level stats (mean/std) as an example
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)

    @staticmethod
    def _example_stats(x: torch.Tensor):
        """
        Compute simple per-feature stats.
        x: (C, T) or (T, C) as torch tensor. We expect (C, T).
        Returns (mean, std) per feature vector of shape (C, 2)
        """
        if x.dim() == 2:
            feat = x  # (C, T)
        elif x.dim() == 1:
            raise ValueError("Expected 2D feature array")
        else:
            feat = x
        mean = feat.mean(dim=1, keepdim=True)
        std = feat.std(dim=1, keepdim=True)
        return torch.cat([mean, std], dim=1)  # (C, 2)

    def compute_weights(self, x: torch.Tensor, feature_order: Optional[List[str]] = None) -> torch.Tensor:
        """
        x: (C, T) tensor representing one example (channels x time)
        feature_order: list of feature names matching the order of channels in x; if None, uses embedder.feature_names
        returns: weights tensor (C,) values in [0,1]
        """
        # prepare feature embeddings
        names = feature_order if feature_order is not None else self.embedder.feature_names
        emb = self.embedder(names.to(self.device) if isinstance(names, torch.Tensor) else names)  # (C, emb_dim)
        stats = self._example_stats(x.to(self.device))  # (C, 2)
        inp = torch.cat([emb, stats], dim=1)  # (C, emb_dim+2)
        weights = self.scorer(inp).squeeze(-1)  # (C,)
        return weights

    @classmethod
    def from_ontology_file(cls, ontology_path: str, emb_dim: int = 32, hidden_dim: int = 64, device: Optional[str] = None):
        """
        Load ontology JSON file that contains a list of feature names or a mapping.
        """
        p = Path(ontology_path)
        data = json.loads(p.read_text())
        # Allow either list or dict
        if isinstance(data, dict):
            feature_names = list(data.keys())
        elif isinstance(data, list):
            feature_names = data
        else:
            raise ValueError("Ontology file must contain a list or dict of feature names.")
        return cls(feature_names=feature_names, emb_dim=emb_dim, hidden_dim=hidden_dim, device=device)
