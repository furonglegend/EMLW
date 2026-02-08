"""
rl_gating.py

RL-Gating skeleton.

Provides:
- Simple environment wrapper to build a state vector from device stats and feature stats.
- A lightweight REINFORCE policy trainer and inference utilities.
- PolicyNetwork class is expected to be compatible with the one from models.PolicyNetwork,
  but this module also contains a small Agent wrapper to handle sampling and policy updates.

Notes:
- Replace REINFORCE with actor-critic / PPO as needed for sample efficiency.
- The reward function and state composition must be implemented according to the paper.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class GatingEnv:
    """
    Small wrapper that converts device + feature stats into a flat state vector.
    Device_info: dict with keys like cpu_load, mem_free, battery, latency_ms
    Feature_stats: dict with aggregated stats like mean_confidence, feature_count, etc.
    The env only provides state construction + a step() placeholder; the RL training loop is external.
    """
    def __init__(self, device_keys: Optional[list] = None, feature_keys: Optional[list] = None):
        self.device_keys = device_keys or ["cpu_load", "mem_free", "battery", "latency_ms"]
        self.feature_keys = feature_keys or ["feat_mean", "feat_std", "num_active_feats"]
        self.state_dim = len(self.device_keys) + len(self.feature_keys)

    def build_state(self, device_info: Dict[str, float], feature_stats: Dict[str, float]) -> torch.Tensor:
        vals = []
        for k in self.device_keys:
            vals.append(float(device_info.get(k, 0.0)))
        for k in self.feature_keys:
            vals.append(float(feature_stats.get(k, 0.0)))
        state = torch.tensor(vals, dtype=torch.float32)
        return state


class REINFORCEAgent:
    """
    Simple REINFORCE agent wrapper for policy training.
    Assumes policy is a PyTorch nn.Module returning action logits.
    """
    def __init__(self, policy: nn.Module, lr: float = 1e-3, gamma: float = 0.99, entropy_coeff: float = 0.0):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> (int, float):
        """
        Sample action from policy given state.
        Returns (action, log_prob)
        """
        logits = self.policy(state.unsqueeze(0))  # (1, A)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs.squeeze(0))
        if deterministic:
            action = int(probs.argmax(dim=-1).item())
            logp = dist.log_prob(torch.tensor(action))
        else:
            action = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(action))
        return action, logp

    def update(self, trajectories: list):
        """
        trajectories: list of dicts with keys: 'logps' (list of tensors), 'rewards' (list of floats)
        Implements Vanilla REINFORCE with baseline=0. Use advantage / baseline for variance reduction.
        """
        self.optimizer.zero_grad()
        loss_total = 0.0
        for traj in trajectories:
            logps = traj["logps"]  # list of tensors
            rewards = traj["rewards"]  # list of floats
            # compute discounted returns
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            for logp, G in zip(logps, returns):
                loss_total += -logp * G
            if self.entropy_coeff > 0:
                # entropy bonus across the trajectory (approx)
                # compute approximate entropy from logits stored? As a simple fallback we omit
                pass
        loss_total = loss_total / max(1, len(trajectories))
        loss_total.backward()
        self.optimizer.step()
        return loss_total.item()


# Simple inference helper
def run_policy_inference(policy: nn.Module, state: torch.Tensor, device: Optional[str] = None, deterministic: bool = True) -> int:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    state = state.to(device)
    with torch.no_grad():
        logits = policy(state.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            return int(probs.argmax(dim=-1).cpu().item())
        else:
            dist = torch.distributions.Categorical(probs.squeeze(0))
            return int(dist.sample().item())
