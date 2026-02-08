"""
models.py

Core neural building blocks:
- Encoder (lightweight conv / 1D or Transformer encoder)
- Projection head for contrastive/prototype learning
- Classifier head
- Small policy network skeleton (for RL gating)

All modules are deliberately minimal and easy to extend.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesEncoder(nn.Module):
    """
    Simple 1D-convolutional encoder for multivariate time-series.
    Input shape: (batch, channels, seq_len)
    Output: embedding vector per example (batch, embed_dim)
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, out_dim: int = 128, kernel_sizes=(5, 3, 3)):
        super().__init__()
        layers = []
        cur_ch = in_channels
        for i, k in enumerate(kernel_sizes):
            layers.append(nn.Conv1d(cur_ch, hidden_dim, kernel_size=k, padding=k // 2))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            cur_ch = hidden_dim
        self.conv = nn.Sequential(*layers)
        # global pooling -> linear projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)  # (B, hidden_dim)
        out = self.fc(h)  # (B, out_dim)
        return out


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive / prototype learning.
    """
    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    """
    Simple classifier head that maps embedding to logits.
    """
    def __init__(self, in_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class PolicyNetwork(nn.Module):
    """
    Lightweight policy network for RL gating decisions.
    Input: state vector (could be device stats + feature stats)
    Output: action logits or distribution parameters
    """
    def __init__(self, state_dim: int = 32, hidden_dim: int = 64, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return action logits. For stochastic policy, apply softmax externally.
        """
        h = self.net(state)
        return self.logits(h)


# Convenience wrapper: full model used in client/server contexts
class MedAlignModel(nn.Module):
    """
    Combined model that exposes encoder, projector and classifier for easy use.
    """
    def __init__(self,
                 in_channels: int = 3,
                 encoder_dim: int = 128,
                 proj_dim: int = 64,
                 num_classes: int = 2):
        super().__init__()
        self.encoder = TimeSeriesEncoder(in_channels=in_channels, out_dim=encoder_dim)
        self.projector = ProjectionHead(in_dim=encoder_dim, out_dim=proj_dim)
        self.classifier = ClassifierHead(in_dim=encoder_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor, return_proj: bool = False):
        """
        x: (B, C, T)
        returns: logits (B, num_classes). If return_proj: also returns projection (B, proj_dim)
        """
        emb = self.encoder(x)
        logits = self.classifier(emb)
        if return_proj:
            proj = self.projector(emb)
            return logits, proj
        return logits
