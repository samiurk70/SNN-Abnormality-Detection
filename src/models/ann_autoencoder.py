"""ANN baseline autoencoder — identical architecture to SNN but with ReLU."""
import torch
import torch.nn as nn

class ANNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def count_multiply_accumulate(self, x):
        total = 0; n = x.shape[0]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                total += m.in_features * m.out_features * n
        flops = total * 2 / n
        return {
            'total_flops_per_sample': flops,
            'gpu_energy_fJ': flops * 200,
            'gpu_energy_nJ': flops * 200 / 1e6,
        }
