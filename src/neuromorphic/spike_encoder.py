"""
Rate encoder: converts float features → spike trains over T timesteps.
Feature value = spike probability per timestep.
0.9 → fires ~90% of steps. 0.0 → never fires. 1.0 → always fires.
"""
import torch
import numpy as np

class RateEncoder:
    def __init__(self, T=10):
        self.T = T

    def encode(self, x):
        # x: (batch, features) → output: (T, batch, features)
        x_clamped = x.clamp(0.0, 1.0)
        x_expanded = x_clamped.unsqueeze(0).repeat(self.T, 1, 1)
        spikes = torch.rand_like(x_expanded) < x_expanded
        return spikes.float()

    def decode(self, spike_train):
        return spike_train.mean(dim=0)

class TemporalEncoder:
    """Time-to-first-spike: high value → early spike."""
    def __init__(self, T=10):
        self.T = T

    def encode(self, x):
        x_clamped = x.clamp(0.0, 1.0)
        spike_times = ((1.0 - x_clamped) * (self.T - 1)).long()
        T, B, F = self.T, x.shape[0], x.shape[1]
        spikes = torch.zeros(T, B, F, device=x.device)
        spikes.scatter_(0, spike_times.unsqueeze(0), 1.0)
        return spikes

def normalise_features(X):
    X_min = X.min(axis=0); X_max = X.max(axis=0)
    denom = X_max - X_min; denom[denom == 0] = 1.0
    return (X - X_min) / denom
