"""
Recompute calibration and ocean-only skill from saved eval_data/fields.npz, without retraining.

metrics.csv's "Spread/Error" is mean(std)/mean|ens mean - truth| over every pixel, including
land, using the spread of members *before* the clamp at 0 m. That mixes land cells (spread > 0,
error = 0) into the average, and its calibrated value isn't 1 anyway (~1.21 for K=20, Gaussian).
This computes, over ocean cells only and from the saved (clamped, land-zeroed) members:
  spread-skill ratio = sqrt((K+1)/K * mean ens variance) / RMSE(ens mean)   [1 = calibrated]
plus ensemble-mean and deterministic RMSE/MAE and ensemble CRPS on the same cells, so CRPS can be
compared directly against the deterministic model's MAE (CRPS reduces to MAE for one member).

Writes one row per window to <out-dir>/<batch>.csv. Does not modify any run directory.

Usage:
    python evaluation/recompute_spread_skill.py --out-dir evaluation/corrected_calibration <batch_dir> [...]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

CHUNK = 64


def window_sums(fields_path):
    f = np.load(fields_path)
    preds = f["preds_all_phys"]  # (N, K, 1, H, W)
    truth_all = f["Y_test_phys"]
    det_all = f["Y_pred_det_phys"]
    mask_all = f["mask_test"]
    N, K = preds.shape[:2]
    s = dict(n=0.0, sum_var=0.0, sse_mean=0.0, sae_mean=0.0, sse_det=0.0, sae_det=0.0, sum_crps=0.0)
    w = 2 * np.arange(1, K + 1) - K - 1
    for i in range(0, N, CHUNK):
        ocean = mask_all[i:i + CHUNK, 0] <= 0.5
        ens = np.moveaxis(preds[i:i + CHUNK, :, 0], 1, -1)[ocean].astype(np.float64)
        y = truth_all[i:i + CHUNK, 0][ocean].astype(np.float64)
        det = det_all[i:i + CHUNK, 0][ocean].astype(np.float64)
        mu = ens.mean(1)
        s["n"] += y.size
        s["sum_var"] += ens.var(1, ddof=1).sum()
        s["sse_mean"] += ((mu - y) ** 2).sum()
        s["sae_mean"] += np.abs(mu - y).sum()
        s["sse_det"] += ((det - y) ** 2).sum()
        s["sae_det"] += np.abs(det - y).sum()
        crps = np.abs(ens - y[:, None]).mean(1) - (np.sort(ens, 1) * w).sum(1) / K ** 2
        s["sum_crps"] += crps.sum()
    s["K"] = K
    return s


def summarize(s):
    n, K = s["n"], s["K"]
    rmse = np.sqrt(s["sse_mean"] / n)
    return {
        "K": K, "N ocean cells": int(n),
        "Spread-skill ratio": np.sqrt((K + 1) / K * s["sum_var"] / n) / rmse,
        "Ens-mean RMSE (ocean)": rmse, "Ens-mean MAE (ocean)": s["sae_mean"] / n,
        "Det RMSE (ocean)": np.sqrt(s["sse_det"] / n), "Det MAE (ocean)": s["sae_det"] / n,
        "CRPS (ocean)": s["sum_crps"] / n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("batch_dirs", nargs="+")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for batch in args.batch_dirs:
        rows, pooled = [], None
        for fp in sorted(glob.glob(os.path.join(batch, "*", "eval_data", "fields.npz"))):
            run = os.path.basename(os.path.dirname(os.path.dirname(fp)))
            print(f"{batch}: {run}", flush=True)
            s = window_sums(fp)
            rows.append({"run": run, **summarize(s)})
            if pooled is None:
                pooled = dict(s)
            else:
                assert pooled["K"] == s["K"]
                for k in s:
                    if k != "K":
                        pooled[k] += s[k]
        if pooled is None:
            print(f"{batch}: no fields.npz found", flush=True)
            continue
        rows.append({"run": "POOLED", **summarize(pooled)})
        df = pd.DataFrame(rows)
        out = os.path.join(args.out_dir, os.path.basename(os.path.normpath(batch)) + ".csv")
        df.to_csv(out, index=False)
        print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
