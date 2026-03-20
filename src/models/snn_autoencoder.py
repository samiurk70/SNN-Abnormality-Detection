"""
SNN Autoencoder using Leaky Integrate-and-Fire (LIF) neurons.
Trains on normal samples only. Anomaly score = reconstruction error.
Uses surrogate gradients (ATan) for backprop through spike threshold.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional
from spikingjelly.activation_based import surrogate

class SNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, T=8, tau=2.0, threshold=1.0):
        super().__init__()
        self.T = T
        self.input_dim = input_dim
        sg = surrogate.ATan()

        self.encoder = nn.Sequential(
            layer.Linear(input_dim, hidden_dim),
            neuron.LIFNode(tau=tau, v_threshold=threshold, surrogate_function=sg, detach_reset=True),
            layer.Linear(hidden_dim, latent_dim),
            neuron.LIFNode(tau=tau, v_threshold=threshold, surrogate_function=sg, detach_reset=True),
        )
        self.decoder = nn.Sequential(
            layer.Linear(latent_dim, hidden_dim),
            neuron.LIFNode(tau=tau, v_threshold=threshold, surrogate_function=sg, detach_reset=True),
            layer.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        functional.reset_net(self)
        out_sum = torch.zeros(x.shape[0], self.input_dim, device=x.device)
        for _ in range(self.T):
            out_sum += self.decoder(self.encoder(x))
        return out_sum / self.T

    def encode_latent(self, x):
        functional.reset_net(self)
        z_sum = None
        for _ in range(self.T):
            h = self.encoder[1](self.encoder[0](x))
            z = self.encoder[3](self.encoder[2](h))
            z_sum = z if z_sum is None else z_sum + z
        return z_sum / self.T

    def count_synaptic_operations(self, x):
        functional.reset_net(self)
        total = 0
        with torch.no_grad():
            for _ in range(self.T):
                s1 = self.encoder[1](self.encoder[0](x))
                total += s1.sum().item() * self.encoder[2].weight.shape[0]
                s2 = self.encoder[3](self.encoder[2](s1))
                total += s2.sum().item() * self.decoder[0].weight.shape[0]
                s3 = self.decoder[1](self.decoder[0](s2))
                total += s3.sum().item() * self.decoder[2].weight.shape[0]
        n = x.shape[0]
        synops = total / n
        return {
            'total_synops_per_sample': synops,
            'estimated_energy_fJ': synops * 4.6,
            'estimated_energy_nJ': synops * 4.6 / 1e6,
        }
