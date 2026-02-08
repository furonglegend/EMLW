"""
eval.py

Evaluation utilities:
- evaluate_global: compute metrics (accuracy, mcc, snr) for a global model or prototype set
- reconstruction attack placeholder: compute SSIM/PSNR for a reconstruction vs original (toy)
- visualization helpers for saliency heatmaps (placeholder)

This module is intentionally minimal. Replace dataset/model hooks with actual implementations.
"""

from typing import Dict, Any
import numpy as np
import torch

from medalign.utils import accuracy_from_logits, matthews_corrcoef, snr_db


def evaluate_global(model: torch.nn.Module, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate model on a small synthetic test set for demonstration.
    Returns a dict of metrics.
    """
    from medalign.data import synthetic_dataset, make_dataloader
    xs, ys = synthetic_dataset(num_examples=200, seq_len=128, channels=3, n_classes=config.get("num_classes", 2))
    dl = make_dataloader(xs, ys, batch_size=64, shuffle=False, augment=False, num_workers=0)
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(next(model.parameters()).device))
            logits = logits.cpu().numpy()
            logits_list.append(logits)
            labels_list.append(yb.numpy())
    logits_all = np.concatenate(logits_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    acc = accuracy_from_logits(logits_all, labels_all)
    preds = logits_all.argmax(axis=1)
    mcc = matthews_corrcoef(preds, labels_all)
    # For SNR demo, compute snr between a clean synthetic and one with noise (simple)
    clean = np.stack(xs[:50], axis=0)[0]  # pick first example
    noisy = clean + 0.01 * np.random.randn(*clean.shape)
    snr = snr_db(clean, noisy)
    return {"accuracy": acc, "mcc": mcc, "snr_db": snr}


def compute_psnr(clean: np.ndarray, recon: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute PSNR for reconstructed signal vs clean signal.
    """
    mse = np.mean((clean - recon) ** 2)
    if mse <= 0:
        return float("inf")
    maxval = np.max(np.abs(clean))
    psnr = 20 * np.log10(maxval) - 10 * np.log10(mse + eps)
    return float(psnr)


def compute_ssim(clean: np.ndarray, recon: np.ndarray) -> float:
    """
    Placeholder SSIM computation for 1D/2D signals. For production, use skimage.metrics.structural_similarity.
    Here we return a dummy value.
    """
    # TODO: replace with real SSIM computation from scikit-image if available
    return 0.5
