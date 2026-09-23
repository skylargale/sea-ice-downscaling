"""
Recompute headline skill metrics from saved eval_data/fields.npz with every method scored on the
same cells.

metrics.csv scores the bilinear baseline over land as well, where it is never zeroed, while the
network outputs are hard-zeroed on land (so they get free zero-error land cells). Here RMSE, MAE,
bias, pattern correlation, gradient MAE and coastal RMSE use ocean cells only; IIEE is ocean-masked
and cos(lat)-weighted as in ice_edge_error, with a 1e-4 m ice threshold (see recompute_iiee.py); SSIM is computed on full fields with land set to 0 for
every method. The EngressNet row is the ensemble mean (no per-member averaging).

Writes one row per (window, method) plus a 4-window mean to <out-dir>/<batch>_ocean_metrics.csv.

Usage:
    python evaluation/recompute_ocean_metrics.py --out-dir evaluation/corrected_calibration <batch_dir> [...]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
import functions_engressnet as fe  # noqa: E402

CHUNK = 128


def window_metrics(fp):
    f = np.load(fp)
    truth = torch.from_numpy(f["Y_test_phys"])
    mask = torch.from_numpy(f["mask_test"])
    ocean = mask[:, 0] <= 0.5  # (N, H, W)
    hlat_full = torch.from_numpy(f["hlat"])
    H = truth.shape[-2]
    hlat = hlat_full[:H] if hlat_full.shape[0] != H else hlat_full
    coastal = fe.coastal_band_mask(mask[:, 0], coastal_width=5)
    methods = {
        "Bilinear": torch.from_numpy(f["Y_base_phys"]) * ocean.unsqueeze(1),
        "Deterministic (zero-noise pass)": torch.from_numpy(f["Y_pred_det_phys"]),
        "EngressNet ensemble mean": torch.from_numpy(f["Y_pred_phys"]),
    }
    out = {}
    for name, pred in methods.items():
        err = (pred - truth)[:, 0]
        e_o = err[ocean].double()
        r_vals = []
        for i in range(pred.shape[0]):
            o = ocean[i]
            a, b = pred[i, 0][o].double(), truth[i, 0][o].double()
            if a.std() > 0 and b.std() > 0:
                r_vals.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
        gx_ok = ocean[:, :, 1:] & ocean[:, :, :-1]
        gy_ok = ocean[:, 1:, :] & ocean[:, :-1, :]
        p0, t0 = pred[:, 0], truth[:, 0]
        gx = ((p0[:, :, 1:] - p0[:, :, :-1]) - (t0[:, :, 1:] - t0[:, :, :-1]))[gx_ok].abs().double()
        gy = ((p0[:, 1:, :] - p0[:, :-1, :]) - (t0[:, 1:, :] - t0[:, :-1, :]))[gy_ok].abs().double()
        ssim_vals = [fe.ssim(pred[i:i + CHUNK], truth[i:i + CHUNK]) for i in range(0, pred.shape[0], CHUNK)]
        ssim_w = [min(CHUNK, pred.shape[0] - i) for i in range(0, pred.shape[0], CHUNK)]
        ssim_v = [v.item() if torch.is_tensor(v) else float(v) for v in ssim_vals]
        e_c = err[coastal.bool() & ocean].double()
        out[name] = {
            "RMSE": e_o.pow(2).mean().sqrt().item(), "MAE": e_o.abs().mean().item(), "Bias": e_o.mean().item(),
            "Pattern Corr": float(np.mean(r_vals)),
            "Grad MAE": torch.cat([gx, gy]).mean().item(),
            "SSIM": float(np.average(ssim_v, weights=ssim_w)),
            "IIEE": float(fe.ice_edge_error(pred, truth, threshold=1e-4, mask_bool=ocean, lat=hlat)),
            "Coastal RMSE": e_c.pow(2).mean().sqrt().item(),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("batch_dirs", nargs="+")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for batch in args.batch_dirs:
        rows = []
        for fp in sorted(glob.glob(os.path.join(batch, "*", "eval_data", "fields.npz"))):
            run = os.path.basename(os.path.dirname(os.path.dirname(fp)))
            print(f"{batch}: {run}", flush=True)
            for method, m in window_metrics(fp).items():
                rows.append({"run": run, "method": method, **m})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        mean = df.groupby("method", sort=False).mean(numeric_only=True).reset_index()
        mean.insert(0, "run", "MEAN over windows")
        df = pd.concat([df, mean], ignore_index=True)
        out = os.path.join(args.out_dir, os.path.basename(os.path.normpath(batch)) + "_ocean_metrics.csv")
        df.to_csv(out, index=False)
        print(mean.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
