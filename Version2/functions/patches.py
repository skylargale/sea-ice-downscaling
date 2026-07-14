"""Patch extraction, train/test split, and normalization for EngressNet
training, functionized from the notebook.

The patch-extraction math itself (``extract_patches``) is unchanged from
your notebook -- same context/target/stride logic, same scale-factor
approach. One thing flagged rather than silently handled: this scale
factor (``H_hi / H``) only means "pixels of Y per pixel of X" if X and Y
are both on equal-spacing rectilinear grids with a roughly constant
ratio. If you regrid X's ice channels onto EASE-Grid 2.0
(``dest_grid="ease2_n25km"`` in this package's PipelineConfig) while Y
stays on a rectilinear high-res destination, X and Y are on fundamentally
different grid types (equal-area-in-meters vs. equal-angle-in-degrees) and
a single scalar scale factor no longer correctly maps a patch in X-space
to the matching patch in Y-space. ``extract_patches`` now warns (not
raises -- you may know what you're doing) if it detects a strongly
non-square or unusual aspect ratio that suggests this mismatch; it cannot
detect every case, so this is documentation as much as a runtime check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NormalizationStats:
    """Per-channel mean/std computed from TRAINING data only.

    Save these alongside the model checkpoint -- you need the exact same
    stats to denormalize predictions later (see your notebook's
    ``Y_pred_phys = Y_pred * Y_std + Y_mean`` pattern in the eval cell).
    """

    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def compute_normalization(x: torch.Tensor, dims: tuple[int, ...] = (0, 2, 3)) -> NormalizationStats:
    """Compute per-channel mean/std over the given dims, matching the
    original notebook's ``X_train_fields.mean(dim=(0, 2, 3), keepdim=True)``.
    """
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)
    return NormalizationStats(mean=mean, std=std)


def train_test_split_fields(
    X: torch.Tensor,
    Y: torch.Tensor,
    train_frac: float = 0.7,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten (ensemble, time) into one sample axis and randomly split.

    Equivalent to the original notebook's reshape + ``torch.randperm``
    split. ``seed`` lets you make the split reproducible (the original
    relied on whatever ``torch.manual_seed(0)`` was set globally at the
    top of the notebook, which is fragile if anything else consumes random
    state beforehand).
    """
    N, T, C, H, W = X.shape
    _, _, C_out, H_hi, W_hi = Y.shape

    X_fields = X.reshape(N * T, C, H, W).float()
    Y_fields = Y.reshape(N * T, C_out, H_hi, W_hi).float()

    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    indices = torch.randperm(N * T, generator=gen)
    split = int(train_frac * N * T)

    train_idx, test_idx = indices[:split], indices[split:]
    return (
        X_fields[train_idx], Y_fields[train_idx],
        X_fields[test_idx], Y_fields[test_idx],
    )


def extract_patches(
    X: torch.Tensor,
    Y: torch.Tensor,
    mask: torch.Tensor,
    context_size: tuple[int, int],
    target_size: tuple[int, int],
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slide a (context_h, context_w) window over the LR field X, and for
    each window extract the corresponding centered (target_h, target_w)
    region of the HR field Y (and the matching region of the land mask).

    Unchanged math from your notebook. See module docstring for the
    EASE-Grid caveat on what the LR->HR scale factor actually means.
    """
    N, C, H, W = X.shape
    _, C_out, H_hi, W_hi = Y.shape

    context_h, context_w = context_size
    target_h, target_w = target_size

    if context_h > H or context_w > W:
        raise ValueError(f"Context size {context_size} exceeds LR grid size ({H}, {W})")
    if target_h > context_h or target_w > context_w:
        raise ValueError(f"Target size {target_size} must be <= context size {context_size}")

    scale_y = H_hi / H
    scale_x = W_hi / W
    if abs(scale_y - scale_x) / max(scale_y, scale_x) > 0.05:
        print(
            f"[extract_patches] WARNING: LR->HR scale factors differ "
            f"noticeably between y ({scale_y:.3f}) and x ({scale_x:.3f}). "
            f"This is expected if X and Y are different grid TYPES (e.g. "
            f"X on EASE-Grid 2.0, Y on a rectilinear lat/lon grid) -- in "
            f"that case a single scalar scale factor does not correctly "
            f"map an LR patch to the matching HR region, and patches may "
            f"be spatially misaligned. See module docstring."
        )

    pad_h = (context_h - target_h) // 2
    pad_w = (context_w - target_w) // 2

    X_patches, Y_patches, M_patches = [], [], []

    for n in range(N):
        for i in range(0, H - context_h + 1, stride):
            for j in range(0, W - context_w + 1, stride):
                x_patch = X[n, :, i:i + context_h, j:j + context_w]

                y0 = round((i + pad_h) * scale_y)
                x0 = round((j + pad_w) * scale_x)
                y1 = y0 + int(round(target_h * scale_y))
                x1 = x0 + int(round(target_w * scale_x))

                y_patch = Y[n, :, y0:y1, x0:x1]
                mask_patch = mask[0, 0, y0:y1, x0:x1].unsqueeze(0)

                X_patches.append(x_patch)
                Y_patches.append(y_patch)
                M_patches.append(mask_patch)

    return torch.stack(X_patches), torch.stack(Y_patches), torch.stack(M_patches)
