"""VAE encoder/decoder for ARC-AGI-3 frames.

Frame format: (3, 64, 64) color-index arrays (0-15 range)
Output: 64-d latent vector per frame
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """CNN encoder: (3, 64, 64) -> 64-d latent."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    """CNN decoder: 64-d latent -> (3, 64, 64)."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z)
        z = z.view(z.size(0), 256, 4, 4)
        return self.deconv(z)


class VAE(nn.Module):
    """VAE: encoder + decoder with reparameterization."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss: reconstruction + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
