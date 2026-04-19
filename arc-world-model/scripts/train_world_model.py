"""World model training: VAE + Transition Model.

Trains encoder/decoder to reconstruct frames, plus transition model
to predict future latents. Success criterion: 5-step open-loop
prediction produces recognizable frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from arcwm.env.replay_buffer import ReplayBuffer
from arcwm.models.vae import VAE, vae_loss
from arcwm.models.rssm import TransitionModel


def load_buffer(path: str, batch_size: int = 32):
    """Load replay buffer and return DataLoader."""
    buf = ReplayBuffer.load(path)

    frames = torch.tensor(buf.frames[: len(buf)], dtype=torch.float32) / 15.0
    next_frames = torch.tensor(buf.next_frames[: len(buf)], dtype=torch.float32) / 15.0
    actions = torch.tensor(buf.fields["action"][: len(buf)], dtype=torch.long)
    rewards = torch.tensor(buf.fields["reward"][: len(buf)], dtype=torch.float32)

    print(f"Loaded {len(buf)} transitions")
    print(f"  Frames: {frames.shape}")
    print(f"  Actions: {actions.min()} - {actions.max()}")

    dataset = TensorDataset(frames, next_frames, actions, rewards)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_vae epoch(vae, dataloader, optimizer, device):
    """Train VAE for one epoch."""
    vae.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch in dataloader:
        frames, _, _, _ = batch
        frames = frames.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = vae(frames)
        loss, recon_loss, kl_loss = vae_loss(recon, frames, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


def train_transition(trans_model, vae, dataloader, optimizer, device):
    """Train transition model: p(z_{t+1} | z_t, a_t)."""
    trans_model.train()

    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        frames, next_frames, actions, _ = batch
        frames = frames.to(device)
        next_frames = next_frames.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            z_t = vae.encode(frames)[0]
            z_next = vae.encode(next_frames)[0]

        z_pred, mu, logvar = trans_model(z_t, actions)

        loss = nn.functional.mse_loss(z_pred, z_next)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train world model")
    parser.add_argument("--buffer-path", default="./data/replay", help="Replay buffer path")
    parser.add_argument("--epochs", type=int, default=10, help="VAE epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading data...")
    dataloader = load_buffer(args.buffer_path, args.batch_size)

    print(f"\n=== Training VAE ({args.epochs} epochs) ===")
    vae = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss, recon, kl = train_vae_epoch(vae, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}: loss={loss:.2f} recon={recon:.2f} kl={kl:.2f}")

    print(f"\n=== Training Transition Model ===")
    trans_model = TransitionModel(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(trans_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_transition(trans_model, vae, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}: trans_loss={loss:.4f}")

    print("\n=== Checking 5-step open-loop prediction ===")
    vae.eval()
    trans_model.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))
        frames, _, actions, _ = batch
        frames = frames[:8].to(device)
        actions = actions[:8].to(device)

        z = vae.encode(frames)[0]

        for step in range(5):
            z = trans_model.predict(z, actions[step % len(actions)])
            recon = vae.decode(z)
            if step == 0:
                print(f"Step {step + 1}: shape={recon.shape}, mean={recon.mean():.3f}")
            else:
                print(f"Step {step + 1}: mean={recon.mean():.3f}")

    print("\nWorld model training complete!")
    print(f"Test by: decode random z and see if it looks like game frames")

    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(vae.state_dict(), checkpoint_dir / "vae.pt")
    torch.save(trans_model.state_dict(), checkpoint_dir / "transition.pt")
    print(f"Saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()