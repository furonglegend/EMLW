"""
utils.py

Common utilities:
- basic logging setup
- checkpoint save/load for PyTorch models
- evaluation metrics: accuracy, MCC (Matthews correlation coefficient), SNR (dB)
- small helper functions (seeding, maybe device helper)
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across torch, numpy and python.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: str, name: str = "medalign", level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger that writes to both stdout and a file in log_dir.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# -------------------------
# Checkpoint utilities
# -------------------------
def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """
    Save checkpoint dict using torch.save. `state` typically contains model.state_dict, optimizer.state_dict, epoch, etc.
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """
    Load checkpoint saved by save_checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


# -------------------------
# Metrics
# -------------------------
def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.
    logits: shape (N, C) or (N,) for binary logits.
    labels: shape (N,)
    """
    if logits.ndim == 1 or logits.shape[1] == 1:
        preds = (logits.ravel() > 0).astype(int)
    else:
        preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return float(acc)


def matthews_corrcoef(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Matthews correlation coefficient (MCC).
    preds: predicted class labels (0..C-1)
    labels: ground-truth labels
    """
    if preds.ndim != 1:
        preds = preds.ravel()
    if labels.ndim != 1:
        labels = labels.ravel()
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    numerator = tp * tn - fp * fn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float(numerator / denom)


def snr_db(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Signal-to-Noise Ratio in decibels between clean and noisy signals.
    Both arrays must be same shape.
    SNR(dB) = 10 * log10( sum(clean^2) / sum((clean - noisy)^2) )
    """
    power_signal = np.sum(clean.astype(np.float64) ** 2)
    power_noise = np.sum((clean.astype(np.float64) - noisy.astype(np.float64)) ** 2) + eps
    snr = 10.0 * np.log10(power_signal / power_noise + eps)
    return float(snr)


# -------------------------
# Small device helper
# -------------------------
def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Return torch.device instance. preferred: "cpu" | "cuda" | None
    """
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
