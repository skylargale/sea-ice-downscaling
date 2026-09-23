"""
Recompute IIEE from already-saved eval_data/fields.npz, without re-running training.

Why this exists: run_pipeline's physical-space de-normalization used to multiply by a
bare `std` while normalization divided by `(std + 1e-6)` -- an asymmetric round trip
that left true-zero (open water) pixels at a tiny nonzero floor (~1e-6 m) instead of
exact 0.0 after de-normalization. That's since been fixed at the source in
training/functions_engressnet.py (normalize/de-normalize now use the same epsilon), so
any run kicked off after that fix computes IIEE correctly with the original
threshold=0.0. Runs already completed (or in flight) before the fix baked the floor
into their saved fields.npz -- and fields.npz never stored the mean/std needed to
invert the round trip exactly after the fact. Since the floor (<=1e-6 m here, confirmed
across multiple windows/fields) is ~1000x below any physically real ice thickness, a
small physically-negligible threshold (1e-4 m = 0.1 mm) recovers the correct IIEE from
the existing saved arrays without retraining.

Usage:
    python recompute_iiee.py <batch_dir>
    e.g. python recompute_iiee.py results/MESA_noise_cascade_avg

Also applies cos(lat) physical-area weighting (matching ice_edge_error's `lat` argument
and the same convention used throughout evaluation/run_daily_eval_batch.py) instead of a
plain per-pixel-count fraction -- fields.npz already stores "hlat" for every run, so this
applies retroactively without retraining, same as the threshold fix.

Rewrites the "IIEE" column in each window's metrics.csv in place (all other columns/
rows untouched) and prints an old-vs-new comparison per window.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from functions_engressnet import ice_edge_error  # noqa: E402

IIEE_THRESHOLD = 1e-4  # meters; see module docstring


def recompute_window(window_dir):
    fields_path = os.path.join(window_dir, "eval_data", "fields.npz")
    metrics_path = os.path.join(window_dir, "metrics.csv")
    if not (os.path.exists(fields_path) and os.path.exists(metrics_path)):
        print(f"  skipping {window_dir} (missing fields.npz or metrics.csv)")
        return

    d = np.load(fields_path)
    import torch
    Y_test_phys = torch.from_numpy(d["Y_test_phys"])
    mask_test = torch.from_numpy(d["mask_test"])
    ocean_bool = mask_test[:, 0] <= 0.5

    # fields.npz's own "hlat" is the pre-crop bbox_regrid grid used to build land_mask
    # (shape mismatches Y_test_phys's H), not the actual cropped grid the saved fields
    # live on -- that's only in tile_geometry.pkl's "target_lat" (see extract_full_domain
    # in training/functions_engressnet.py, which slices hlat[hi0:hi1] into tile_geometry
    # before hlat itself gets saved to fields.npz unsliced).
    tile_geometry_path = os.path.join(window_dir, "eval_data", "tile_geometry.pkl")
    with open(tile_geometry_path, "rb") as f:
        tile_geometry = pickle.load(f)
    hlat = np.asarray(tile_geometry[0]["target_lat"])

    preds = {
        "Bilinear": torch.from_numpy(d["Y_base_phys"]),
        "Deterministic UNet": torch.from_numpy(d["Y_pred_det_phys"]),
        "Stochastic UNet Mean": torch.from_numpy(d["Y_pred_phys"]),
    }

    df = pd.read_csv(metrics_path)
    old = df.set_index("Method")["IIEE"].to_dict()
    for method, pred in preds.items():
        new_iiee = ice_edge_error(pred, Y_test_phys, threshold=IIEE_THRESHOLD, mask_bool=ocean_bool, lat=hlat)
        df.loc[df["Method"] == method, "IIEE"] = new_iiee
        print(f"  {method:22s} IIEE: {old.get(method):.4f} -> {new_iiee:.4f}")

    df.to_csv(metrics_path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", help="e.g. results/MESA_noise_cascade_avg")
    args = parser.parse_args()

    window_dirs = sorted(glob.glob(os.path.join(args.batch_dir, "*/")))
    if not window_dirs:
        window_dirs = [args.batch_dir]

    for window_dir in window_dirs:
        print(f"{window_dir}")
        recompute_window(window_dir.rstrip("/"))


if __name__ == "__main__":
    main()
