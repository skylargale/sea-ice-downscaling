"""Loss functions for Engression-style stochastic training.

Both functions are unchanged math from your notebook -- ``energy_loss``
implements the energy score for an ensemble of stochastic predictions
against one truth, and ``land_loss`` penalizes positive ice thickness
predicted over land. The only change is making them importable instead of
notebook-global.
"""

from __future__ import annotations

import torch


def energy_loss(preds: torch.Tensor, y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Energy-score loss used for stochastic prediction.

    Args:
        preds: Ensemble predictions, shape [B, K, C, H, W] (B=batch size,
            K=number of stochastic samples).
        y: Ground-truth targets, shape [B, C, H, W].
        beta: Power parameter of the energy score. Default 1.

    Returns:
        Scalar energy-score loss.
    """
    B, K, C, H, W = preds.shape

    flat_preds = preds.reshape(B, K, -1)
    flat_y = y.reshape(B, 1, -1)

    eps = 0.0 if float(beta).is_integer() else 1e-5

    # Average distance between each ensemble member and truth
    s1 = (torch.linalg.vector_norm(flat_preds - flat_y, ord=2, dim=2) + eps).pow(beta).mean()

    # Average pairwise distance between ensemble members
    s2 = (torch.cdist(flat_preds, flat_preds, p=2) + eps).pow(beta).mean() * K / (K - 1)

    return s1 - 0.5 * s2


def land_loss(preds: torch.Tensor, land_mask: torch.Tensor) -> torch.Tensor:
    """Penalize positive ice thickness predicted over land.

    ``land_mask`` should already be broadcastable against ``preds``
    (e.g. shape [B, 1, H, W] or [1, 1, H, W]); this matches how the
    notebook called it with a pre-expanded mask batch.
    """
    land_mask = land_mask.to(preds.device)
    return ((torch.relu(preds) * land_mask) ** 2).mean()
