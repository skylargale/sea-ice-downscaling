"""Evaluation, baseline comparison, and skill metrics for EngressNet,
functionized from the notebook's evaluation cells.

Same math throughout (energy-score-style ensemble eval, bilinear
baseline, MAE/RMSE/gradient-MAE/spread-skill metrics) -- the only fix is
removing the ``model.module.latent_channels`` access (see model.py) and
the ``sit_idx = 0`` hardcoded literal (now looked up by name via
``channels.find_channel_index``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from .patches import NormalizationStats


@dataclass
class EnsembleEvalResult:
    preds_all: torch.Tensor  # [N, K, C, H, W] -- every stochastic sample
    pred_mean: torch.Tensor  # [N, C, H, W]
    pred_std: torch.Tensor  # [N, C, H, W]
    pred_det: torch.Tensor  # [N, C, H, W] -- deterministic (z=0) prediction


@torch.inference_mode()
def evaluate_ensemble(
    model: nn.Module,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    latent_channels: int,
    device: torch.device,
    batch_size: int = 16,
    k_eval: int = 6,
) -> EnsembleEvalResult:
    """Run the model K times per sample (stochastic ensemble) plus once
    deterministically (z=0), matching the notebook's evaluation cell.
    """
    model.eval()

    preds_all, preds_mean, preds_std, preds_det = [], [], [], []

    for i in range(0, X_test.shape[0], batch_size):
        X_batch = X_test[i:i + batch_size].to(device)
        Y_batch = Y_test[i:i + batch_size].to(device)
        B, _, H, W = X_batch.shape

        ensemble_preds = []
        for _ in range(k_eval):
            z = torch.randn(B, latent_channels, H // 8, W // 8, device=device)
            pred = model(X_batch, Y_batch.shape[-2:], z=z)
            ensemble_preds.append(pred)

        preds = torch.stack(ensemble_preds, dim=0).permute(1, 0, 2, 3, 4)
        pred_mean = preds.mean(dim=1)
        pred_std = preds.std(dim=1)

        z0 = torch.zeros(B, latent_channels, H // 8, W // 8, device=device)
        pred_det = model(X_batch, Y_batch.shape[-2:], z=z0)

        preds_all.append(preds.cpu())
        preds_mean.append(pred_mean.cpu())
        preds_std.append(pred_std.cpu())
        preds_det.append(pred_det.cpu())

    return EnsembleEvalResult(
        preds_all=torch.cat(preds_all, dim=0),
        pred_mean=torch.cat(preds_mean, dim=0),
        pred_std=torch.cat(preds_std, dim=0),
        pred_det=torch.cat(preds_det, dim=0),
    )


def bilinear_baseline(
    X_test: torch.Tensor,
    target_hw: tuple[int, int],
    channel_idx: int,
) -> torch.Tensor:
    """Plain bilinear upsample of one LR channel to the target resolution
    -- the baseline your notebook compared against. ``channel_idx`` should
    come from ``channels.find_channel_index(channel_order, "hi")`` rather
    than a hardcoded ``sit_idx = 0`` literal, since channel position
    depends on what order you built X in.
    """
    return F.interpolate(
        X_test[:, channel_idx:channel_idx + 1],
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )


def denormalize_all(
    Y_test: torch.Tensor,
    eval_result: EnsembleEvalResult,
    X_test: torch.Tensor,
    baseline: torch.Tensor,
    y_stats: NormalizationStats,
    x_stats: NormalizationStats,
    target_channel_idx: int,
) -> dict[str, torch.Tensor]:
    """Convert everything back to physical units, matching the notebook's
    block of ``Y_pred_phys = Y_pred * Y_std + Y_mean`` lines.

    Note: the baseline uses ONLY the target channel's mean/std (it's a
    1-channel field), so we slice x_stats down to that channel rather than
    denormalizing with the full multi-channel stats tensor.
    """
    x_mean_c = x_stats.mean[:, target_channel_idx:target_channel_idx + 1]
    x_std_c = x_stats.std[:, target_channel_idx:target_channel_idx + 1]

    return {
        "Y_test_phys": y_stats.denormalize(Y_test),
        "Y_pred_phys": y_stats.denormalize(eval_result.pred_mean),
        "Y_spread_phys": eval_result.pred_std * y_stats.std,
        "Y_pred_det_phys": y_stats.denormalize(eval_result.pred_det),
        "preds_all_phys": eval_result.preds_all * y_stats.std + y_stats.mean,
        "X_test_target_phys": X_test[:, target_channel_idx:target_channel_idx + 1] * x_std_c + x_mean_c,
        "Y_base_phys": baseline * x_std_c + x_mean_c,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mae(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - truth)).item()


def rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - truth) ** 2)).item()


def grad_mae(pred: torch.Tensor, truth: torch.Tensor) -> float:
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = truth[:, :, :, 1:] - truth[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = truth[:, :, 1:, :] - truth[:, :, :-1, :]
    grad_error = torch.mean(torch.abs(dx_p - dx_t)) + torch.mean(torch.abs(dy_p - dy_t))
    return grad_error.item()


def spread_skill_ratio(pred_mean: torch.Tensor, pred_std: torch.Tensor, truth: torch.Tensor) -> float:
    error = torch.abs(pred_mean - truth)
    return (pred_std.mean() / error.mean()).item()


def spatial_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-sample spatial Pearson correlation, averaged over samples,
    skipping samples with zero variance in either field (a constant
    patch -- e.g. all-land or all-zero-ice -- has undefined correlation).
    """
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    corrs = []
    valid = 0
    total = y_true.shape[0]

    for i in range(y_true.shape[0]):
        std_true = np.std(y_true_flat[i])
        std_pred = np.std(y_pred_flat[i])
        if std_true > 0 and std_pred > 0:
            valid += 1
            corrs.append(np.corrcoef(y_true_flat[i], y_pred_flat[i])[0, 1])

    print(f"Valid patches: {valid}/{total}")
    return float(np.nanmean(corrs)) if corrs else float("nan")


def compute_metrics_table(phys: dict[str, torch.Tensor]) -> list[dict]:
    """Build the same comparison table (Bilinear / Deterministic UNet /
    Stochastic UNet Mean) the notebook printed via pandas.
    """
    return [
        {
            "Method": "Bilinear",
            "MAE": mae(phys["Y_base_phys"], phys["Y_test_phys"]),
            "RMSE": rmse(phys["Y_base_phys"], phys["Y_test_phys"]),
            "Grad MAE": grad_mae(phys["Y_base_phys"], phys["Y_test_phys"]),
            "Spread/Error": np.nan,
        },
        {
            "Method": "Deterministic UNet",
            "MAE": mae(phys["Y_pred_det_phys"], phys["Y_test_phys"]),
            "RMSE": rmse(phys["Y_pred_det_phys"], phys["Y_test_phys"]),
            "Grad MAE": grad_mae(phys["Y_pred_det_phys"], phys["Y_test_phys"]),
            "Spread/Error": np.nan,
        },
        {
            "Method": "Stochastic UNet Mean",
            "MAE": mae(phys["Y_pred_phys"], phys["Y_test_phys"]),
            "RMSE": rmse(phys["Y_pred_phys"], phys["Y_test_phys"]),
            "Grad MAE": grad_mae(phys["Y_pred_phys"], phys["Y_test_phys"]),
            "Spread/Error": spread_skill_ratio(
                phys["Y_pred_phys"], phys["Y_spread_phys"], phys["Y_test_phys"]
            ),
        },
    ]
