"""
data.py

Data loading and augmentation utilities.

- Implements a PyTorch Dataset class expecting multivariate time-series examples.
- Provides augmentation transforms:
    * temporal_warp: randomly warp along the time axis within a given percentage
    * add_noise_snr: add Gaussian noise to achieve a target SNR in dB
    * channel_mask: randomly mask channels with specified probability

This file uses numpy and torch. It is intentionally minimal and easy to extend.
"""

from typing import Callable, Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


Array = np.ndarray


# -------------------------
# Augmentations
# -------------------------
def temporal_warp(x: Array, warp_pct: float = 0.15) -> Array:
    """
    Apply a simple temporal warp by resampling the time axis.
    x: (T, C) or (C, T) - this function expects (T, C)
    warp_pct: maximum relative warp (+/-). Example: 0.15 -> resample length in [0.85T, 1.15T]
    """
    if x.ndim != 2:
        raise ValueError("temporal_warp expects input with shape (T, C)")
    T, C = x.shape
    scale = 1.0 + np.random.uniform(-warp_pct, warp_pct)
    new_T = max(2, int(round(T * scale)))
    # linear resampling using interpolation
    orig_idx = np.linspace(0, 1, T)
    new_idx = np.linspace(0, 1, new_T)
    warped = np.zeros((new_T, C), dtype=x.dtype)
    for c in range(C):
        warped[:, c] = np.interp(new_idx, orig_idx, x[:, c])
    # If needed, pad or trim back to T
    if new_T > T:
        # trim center
        start = (new_T - T) // 2
        warped = warped[start:start + T, :]
    elif new_T < T:
        # pad symmetric
        pad_before = (T - new_T) // 2
        pad_after = T - new_T - pad_before
        warped = np.pad(warped, ((pad_before, pad_after), (0, 0)), mode="edge")
    return warped


def add_noise_to_snr(clean: Array, target_snr_db: float = 20.0) -> Array:
    """
    Add zero-mean Gaussian noise to 'clean' signal to reach target SNR in dB.
    clean: (T, C) numpy array
    """
    power_signal = np.mean(clean.astype(np.float64) ** 2)
    target_linear = 10 ** (target_snr_db / 10.0)
    power_noise = power_signal / (target_linear + 1e-12)
    # Gaussian noise with computed variance
    noise = np.random.normal(loc=0.0, scale=np.sqrt(power_noise), size=clean.shape)
    return clean + noise


def channel_mask(x: Array, mask_prob: float = 0.1) -> Array:
    """
    Randomly zero-out entire channels with probability mask_prob.
    x: (T, C)
    """
    T, C = x.shape
    mask = np.random.rand(C) >= mask_prob
    return x * mask.astype(x.dtype)[None, :]


# -------------------------
# Dataset
# -------------------------
class TimeSeriesDataset(Dataset):
    """
    Simple dataset wrapper for multivariate time-series.
    Expects a list/array of examples, targets.

    Each example shape: (T, C) numpy array.
    """

    def __init__(
        self,
        examples: List[Array],
        labels: List[int],
        augment: bool = False,
        aug_params: Optional[dict] = None,
        transform: Optional[Callable[[Array], Array]] = None,
    ):
        assert len(examples) == len(labels)
        self.examples = examples
        self.labels = labels
        self.augment = augment
        self.aug_params = aug_params or {}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        x = self.examples[idx].astype(np.float32)  # (T, C)
        y = int(self.labels[idx])
        if self.augment:
            # temporal warp
            warp_pct = float(self.aug_params.get("temporal_warp_pct", 0.15))
            x = temporal_warp(x, warp_pct=warp_pct)
            # add noise
            snr_db = float(self.aug_params.get("noise_snr_db", 20.0))
            x = add_noise_to_snr(x, target_snr_db=snr_db)
            # channel mask
            mask_prob = float(self.aug_params.get("channel_mask_prob", 0.1))
            x = channel_mask(x, mask_prob=mask_prob)
        if self.transform:
            x = self.transform(x)
        # convert to torch tensor with shape (C, T)
        x = torch.from_numpy(x).permute(1, 0).contiguous()
        return x, y


# -------------------------
# Data loader helper
# -------------------------
def make_dataloader(
    examples: List[Array],
    labels: List[int],
    batch_size: int = 64,
    shuffle: bool = True,
    augment: bool = False,
    aug_params: Optional[dict] = None,
    num_workers: int = 4,
) -> DataLoader:
    """
    Utility to build a DataLoader from raw arrays/lists.
    """
    dataset = TimeSeriesDataset(
        examples=examples,
        labels=labels,
        augment=augment,
        aug_params=aug_params,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# -------------------------
# Toy loader for quick debugging (synthetic data)
# -------------------------
def synthetic_dataset(num_examples: int = 1000, seq_len: int = 200, channels: int = 3, n_classes: int = 2):
    """
    Create a toy synthetic dataset for quick tests.
    Each class will have a different sinusoidal frequency.
    """
    xs = []
    ys = []
    t = np.linspace(0, 1, seq_len)
    for i in range(num_examples):
        label = np.random.randint(0, n_classes)
        freq = 1.0 + label * 0.5 + np.random.randn() * 0.05
        signal = np.sin(2 * np.pi * freq * t)[:, None]
        # add small channel variations
        channels_signal = np.concatenate([signal * (1 + 0.1 * np.random.randn()) for _ in range(channels)], axis=1)
        xs.append(channels_signal.astype(np.float32))
        ys.append(label)
    return xs, ys
