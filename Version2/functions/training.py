"""Training loop for EngressNet, functionized from the notebook's
"Training loop (Engression)" cell.

The notebook had TWO training loop cells: one referencing undefined
``masked_loss`` / ``base_loss`` / ``USE_MASKED_LOSS`` (would NameError
immediately, and doesn't pass ``z`` to the model at all -- looks like an
earlier, now-superseded draft), and the Engression one with
``energy_loss`` + ``land_loss`` that's actually complete and consistent
with the rest of the notebook (the eval cells assume stochastic ensemble
predictions exist). This module implements only the Engression loop. If
you actually want the masked-loss variant revived, it needs `masked_loss`
and `base_loss` defined from scratch -- they don't exist anywhere in your
notebook as uploaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .losses import energy_loss, land_loss


@dataclass
class TrainConfig:
    num_epochs: int = 20
    batch_size: int = 32
    k_ensemble: int = 6  # number of stochastic samples per training step
    learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0
    latent_channels: int = 8  # must match the model's latent_channels


@dataclass
class TrainHistory:
    loss: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    land: list[float] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    M_train: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one epoch of Engression training. Returns (loss, energy, land)
    averaged over the epoch, matching what the notebook printed per epoch.
    """
    model.train()
    epoch_energy = epoch_land = epoch_loss = 0.0
    n = X_train.size(0)
    idx = torch.randperm(n)

    for i in range(0, n, config.batch_size):
        optimizer.zero_grad()

        batch_idx = idx[i:i + config.batch_size]
        X_batch = X_train[batch_idx].to(device)
        Y_batch = Y_train[batch_idx].to(device)
        M_batch = M_train[batch_idx].to(device)

        B, C, H, W = X_batch.shape

        X_rep = X_batch.repeat_interleave(config.k_ensemble, dim=0)
        z = torch.randn(
            B * config.k_ensemble, config.latent_channels, H // 8, W // 8,
            device=device,
        )

        preds = model(X_rep, Y_batch.shape[-2:], z=z)
        preds = preds.reshape(B, config.k_ensemble, *preds.shape[1:])

        loss_energy = energy_loss(preds, Y_batch)
        loss_land = land_loss(preds, M_batch)
        loss = loss_energy + loss_land

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        epoch_energy += loss_energy.item() * B
        epoch_land += loss_land.item() * B
        epoch_loss += loss.item() * B

        del X_batch, Y_batch, preds, loss, loss_energy, loss_land

    return epoch_loss / n, epoch_energy / n, epoch_land / n


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    M_train: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> TrainHistory:
    """Run the full training loop, replacing the notebook's bare for-loop
    over epochs.
    """
    history = TrainHistory()

    for epoch in range(config.num_epochs):
        loss, energy, land = train_one_epoch(
            model, optimizer, X_train, Y_train, M_train, config, device
        )
        history.loss.append(loss)
        history.energy.append(energy)
        history.land.append(land)
        if verbose:
            print(f"Epoch {epoch + 1}/{config.num_epochs} | Loss: {loss:.6f}")

    return history
