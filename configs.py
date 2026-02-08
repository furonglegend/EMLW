"""
configs.py

Global configuration module for the medalign project.
Contains default hyperparameters, file paths, privacy budgets, and experiment options.

All values are intended to be simple defaults; override via command-line or experiment config files.
"""

from pathlib import Path
from typing import Dict, Any

# Project root (default relative)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model / training defaults
DEFAULTS: Dict[str, Any] = {
    # optimization
    "seed": 42,
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    "batch_size": 64,
    "num_workers": 4,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "num_epochs": 100,

    # encoder / model sizes
    "hidden_dim": 128,
    "proj_dim": 64,
    "num_classes": 2,

    # federated settings (defaults)
    "clients_per_round": 10,
    "federated_rounds": 100,
    "local_epochs": 1,

    # privacy (differential privacy Gaussian mechanism)
    # eps, delta should be tuned per experiment
    "dp": {
        "enabled": True,
        "epsilon": 1.0,
        "delta": 1e-5,
        # clipping bound C used for per-client updates
        "clip": 1.0,
    },

    # compression
    "compression": {
        "topk_fraction": 0.01,
        "quant_bits": 8,
    },

    # RL gating defaults
    "rl": {
        "gamma": 0.99,
        "entropy_coeff": 0.01,
    },

    # augmentation
    "augmentation": {
        # temporal warp percent: sample warp factor in [1 - warp_pct, 1 + warp_pct]
        "temporal_warp_pct": 0.15,
        "noise_snr_db": 20.0,
        "channel_mask_prob": 0.1,
    },
}

# Utility to get a merged configuration dictionary (defaults can be overridden)
def get_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Return a copy of DEFAULTS optionally merged with overrides.
    """
    import copy
    conf = copy.deepcopy(DEFAULTS)
    if overrides:
        # shallow update is usually enough for experiments; for nested merge use custom logic.
        for k, v in overrides.items():
            if isinstance(v, dict) and k in conf and isinstance(conf[k], dict):
                conf[k].update(v)
            else:
                conf[k] = v
    conf["paths"] = {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "checkpoints": str(CHECKPOINT_DIR),
        "logs": str(LOG_DIR),
        "results": str(RESULTS_DIR),
    }
    return conf
