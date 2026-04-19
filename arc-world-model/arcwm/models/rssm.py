"""RSSM: Recurrent State-Space Model for latent dynamics.

Transition model: p(z_{t+1} | z_t, a_t)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    """RSSM transition: z_t + a_t -> z_{t+1}."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.rnn = nn.GRU(latent_dim + action_dim, latent_dim, batch_first=True)

        self.fc_h = nn.Linear(latent_dim, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next latent; returns (next_z, mu, logvar)."""
        batch_size = z.size(0)

        a_one_hot = torch.zeros(batch_size, self.action_dim, device=z.device)
        a_one_hot.scatter_(1, action.long().clamp(0, self.action_dim - 1), 1)

        combined = torch.cat([z, a_one_hot], dim=-1)
        combined = combined.unsqueeze(1)

        h, _ = self.rnn(combined)
        h = h.squeeze(1)

        h = torch.tanh(self.fc_h(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mu + eps * std

        return z_next, mu, logvar

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Deterministic prediction (mean)."""
        batch_size = z.size(0)

        a_one_hot = torch.zeros(batch_size, self.action_dim, device=z.device)
        a_one_hot.scatter_(1, action.long().clamp(0, self.action_dim - 1), 1)

        combined = torch.cat([z, a_one_hot], dim=-1)
        combined = combined.unsqueeze(1)

        h, _ = self.rnn(combined)
        h = h.squeeze(1)

        h = torch.tanh(self.fc_h(h))
        mu = self.fc_mu(h)

        return mu
