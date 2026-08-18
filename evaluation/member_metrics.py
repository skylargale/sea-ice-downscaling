"""
member_metrics.py

Per-member-then-average metric recomputation for MESACLIP runs, shared by
functions_engressnet.py (compute_metrics_table's per_member path, applied
automatically at training time for future runs) and the three evaluation
notebooks (evaluation_plots.ipynb, evaluation_plots_daily.ipynb,
compare_all_batches.ipynb), which retrofit already-finished MESACLIP runs
by recomputing from the preds_all_phys already saved in eval_data/fields.npz.

Why this exists: MESACLIP's saved "truth" (Y_test_phys) is a single CESM1
ensemble-member realization, not an ensemble mean or a deterministic
physical target. Comparing the model's ensemble-MEAN prediction
(Y_pred_phys) against that single realization folds member-to-member
internal variability into an artificially smaller error than any
individual, physically-realizable member would actually show (for any
per-sample squared error, E_k[(member_k - truth)^2] = (mean - truth)^2 +
Var_k(member_k) >= (mean - truth)^2, by the usual bias-variance
decomposition -- the ensemble mean is mathematically smoother than any
single draw). compute_member_avg_metrics() instead computes each metric on
every individual saved ensemble member against truth, then averages the
resulting *metric values* across members -- matching standard
single-realization ensemble verification practice. This is NOT applied to
FOSI runs, whose single-realization "truth" is the one and only physical
trajectory, not one draw from a family of equally-valid members. Spread/Error
is left alone in both cases: it's already designed around a single-truth
comparison (Fortin et al. 2014-style spread-skill: spread vs. ensemble-mean
error against one observation), not the plain ensemble-mean accuracy that
MAE/RMSE/Bias/etc. measure, so it doesn't have the same problem.

Mirrors functions_engressnet.py's mae/rmse/bias/grad_mae/pattern_corr/ssim/
ice_edge_error/masked_mae/masked_rmse/coastal_band_mask formula-for-formula
(reusing torch, already a hard dependency of this project's training
pipeline and the `downscaling_env` these notebooks run in) rather than a
from-scratch numpy port, specifically so the two implementations of the
same metric can't quietly drift apart. Deliberately does NOT import
functions_engressnet.py itself -- that would pull in xesmf/pop_tools/
cartopy at import time, which these lightweight plotting notebooks are
designed to avoid needing (they read only eval_data/, no GPU/xesmf/
pop_tools).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F


def is_mesaclip_run(name_or_path):
    """
    True for the "MESA_" run/batch-folder naming convention used
    throughout Version4 for MESACLIP-sourced data (e.g.
    "MESA_daily_combo_avg", "MESA_el1_dmed_es1_..."), vs. "FOSI_" for the
    single-realization FOSI dataset. Checks only the final path component,
    so it works whether given a batch folder, a run folder, or a full
    output_dir path.
    """
    return os.path.basename(os.path.normpath(str(name_or_path))).startswith("MESA_")


def _as_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x.float()


def mae(pred, truth):
    return torch.mean(torch.abs(pred - truth)).item()


def rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth) ** 2)).item()


def bias(pred, truth):
    return torch.mean(pred - truth).item()


def grad_mae(pred, truth):
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = truth[:, :, :, 1:] - truth[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = truth[:, :, 1:, :] - truth[:, :, :-1, :]
    grad_error = torch.mean(torch.abs(dx_p - dx_t)) + torch.mean(torch.abs(dy_p - dy_t))
    return grad_error.item()


def pattern_corr(pred, truth):
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = truth.reshape(B, -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p ** 2).sum(dim=1) * (t ** 2).sum(dim=1)) + 1e-8
    return (num / den).mean().item()


def _gaussian_window(window_size, sigma, device_):
    coords = torch.arange(window_size, dtype=torch.float32, device=device_) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(pred, truth, window_size=11, sigma=1.5, data_range=None):
    device_ = pred.device
    C = pred.shape[1]
    if data_range is None:
        data_range = (truth.max() - truth.min()).clamp(min=1e-6)

    window = _gaussian_window(window_size, sigma, device_)
    window = window.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=C)
    mu_t = F.conv2d(truth, window, padding=pad, groups=C)
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_p_sq
    sigma_t_sq = F.conv2d(truth * truth, window, padding=pad, groups=C) - mu_t_sq
    sigma_pt = F.conv2d(pred * truth, window, padding=pad, groups=C) - mu_pt

    k1, k2 = 0.01, 0.03
    C1, C2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean().item()


def _ssim_per_member(preds_all, truth, window_size=11, sigma=1.5, chunk_size=1):
    """
    Equivalent of calling ssim(preds_all[:, k], truth) for each of the K
    members and collecting the results, but computed in fewer conv2d
    passes: the truth-only convolutions (mu_t, sigma_t_sq) don't depend on
    the member index, so they're computed once (not K times) and reused --
    a free win, since it doesn't change peak memory at all. This is a pure
    performance optimization -- it reproduces plain per-member ssim()
    exactly (conv2d treats each batch item independently), which mattered
    here because SSIM's 5 convolution passes per member turned out to
    dominate compare_all_batches.ipynb's runtime across 147 runs (hours,
    not minutes) before this was added.

    `chunk_size` (default 1, i.e. no member-batching) exists to fold more
    than one member into a single conv2d call for extra speed, but this
    environment's shared Casper node showed real, unpredictable kills for
    even chunk_size=2 on a large run (N=2190, K=20, 150x310 grid) that had
    plenty of headroom on paper (>800 GB system-free at the time) -- the
    same kind of transient node memory pressure already documented for the
    ProcessPoolExecutor path in this notebook. Default stays at 1 (no extra
    memory over the original per-member loop) since that's proven robust;
    raise it only if this environment's contention improves.

    Returns a (K,) tensor of per-member mean SSIM values.
    """
    N, K, C, H, W = preds_all.shape
    device_ = preds_all.device
    data_range = (truth.max() - truth.min()).clamp(min=1e-6)

    window = _gaussian_window(window_size, sigma, device_)
    window = window.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_t = F.conv2d(truth, window, padding=pad, groups=C)
    mu_t_sq = mu_t ** 2
    sigma_t_sq = F.conv2d(truth * truth, window, padding=pad, groups=C) - mu_t_sq

    k1, k2 = 0.01, 0.03
    C1, C2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2

    per_member = []
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        b = end - start
        preds_chunk = preds_all[:, start:end].reshape(N * b, C, H, W)
        truth_rep = truth.unsqueeze(1).expand(N, b, C, H, W).reshape(N * b, C, H, W)
        mu_t_rep = mu_t.unsqueeze(1).expand(N, b, C, H, W).reshape(N * b, C, H, W)
        mu_t_sq_rep = mu_t_sq.unsqueeze(1).expand(N, b, C, H, W).reshape(N * b, C, H, W)
        sigma_t_sq_rep = sigma_t_sq.unsqueeze(1).expand(N, b, C, H, W).reshape(N * b, C, H, W)

        mu_p = F.conv2d(preds_chunk, window, padding=pad, groups=C)
        mu_p_sq = mu_p ** 2
        mu_pt = mu_p * mu_t_rep
        sigma_p_sq = F.conv2d(preds_chunk * preds_chunk, window, padding=pad, groups=C) - mu_p_sq
        sigma_pt = F.conv2d(preds_chunk * truth_rep, window, padding=pad, groups=C) - mu_pt

        ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
            (mu_p_sq + mu_t_sq_rep + C1) * (sigma_p_sq + sigma_t_sq_rep + C2)
        )
        per_member.append(ssim_map.reshape(N, b, C, H, W).mean(dim=(0, 2, 3, 4)))

    return torch.cat(per_member)


def ice_edge_error(pred, truth, threshold=0.0, mask_bool=None):
    ice_pred = pred[:, 0] > threshold
    ice_truth = truth[:, 0] > threshold
    if mask_bool is not None:
        ice_pred = ice_pred[mask_bool]
        ice_truth = ice_truth[mask_bool]
    overestimate = (ice_pred & ~ice_truth).sum()
    underestimate = (~ice_pred & ice_truth).sum()
    return ((overestimate + underestimate).float() / ice_pred.numel()).item()


def masked_mae(pred, truth, mask_bool):
    sel = torch.abs(pred[:, 0] - truth[:, 0])[mask_bool]
    return sel.mean().item() if sel.numel() > 0 else float("nan")


def masked_rmse(pred, truth, mask_bool):
    sel = ((pred[:, 0] - truth[:, 0]) ** 2)[mask_bool]
    return torch.sqrt(sel.mean()).item() if sel.numel() > 0 else float("nan")


def coastal_band_mask(land_mask, coastal_width=5):
    land = _as_tensor(land_mask)
    lead_shape = land.shape[:-2]
    H, W = land.shape[-2:]
    land_dilated = F.max_pool2d(
        land.reshape(-1, 1, H, W), kernel_size=2 * coastal_width + 1, stride=1, padding=coastal_width
    )
    land_dilated = land_dilated.reshape(*lead_shape, H, W)
    return (land_dilated > 0.5) & (land <= 0.5)


def compute_member_avg_metrics(preds_all_phys, truth_phys, mask_test=None, coastal_width=5):
    """
    preds_all_phys: (N, K, C, H, W) array/tensor of individual ensemble-member
    predictions, already de-normalized and land-zeroed (as saved in
    eval_data/fields.npz's "preds_all_phys", or as computed live in
    run_pipeline()).
    truth_phys: (N, C, H, W) truth.
    mask_test: (N, C_mask, H, W), optional -- same land mask used for
    metrics.csv's Coastal MAE/RMSE and IIEE.

    Returns a dict of {metric_name: member-averaged value} using the exact
    same metric names as functions_engressnet.compute_metrics_table's
    columns, so it drops straight into a "Stochastic UNet Mean" row. Does
    NOT include "Spread/Error" -- that's still computed from the ensemble
    mean/std, not per-member (see module docstring).
    """
    preds_all = _as_tensor(preds_all_phys)
    truth = _as_tensor(truth_phys)
    K = preds_all.shape[1]

    mask_t = _as_tensor(mask_test) if mask_test is not None else None
    coastal_band = coastal_band_mask(mask_t[:, 0], coastal_width=coastal_width) if mask_t is not None else None
    ocean_bool = (mask_t[:, 0] <= 0.5) if mask_t is not None else None

    def avg(fn, **kwargs):
        vals = [fn(preds_all[:, k], truth, **kwargs) for k in range(K)]
        return float(np.mean(vals))

    return {
        "MAE": avg(mae),
        "RMSE": avg(rmse),
        "Bias": avg(bias),
        "Grad MAE": avg(grad_mae),
        "Pattern Corr": avg(pattern_corr),
        "SSIM": float(_ssim_per_member(preds_all, truth).mean().item()),
        "IIEE": avg(ice_edge_error, mask_bool=ocean_bool),
        "Coastal MAE": avg(masked_mae, mask_bool=coastal_band) if coastal_band is not None else float("nan"),
        "Coastal RMSE": avg(masked_rmse, mask_bool=coastal_band) if coastal_band is not None else float("nan"),
    }
