#!/usr/bin/env python3
"""Apply the coarse-zero rule to already-saved evaluation outputs (2026-10-23).

The rule (training/functions_engressnet.py coarse_zero_mask, --coarse-zero): at evaluation, members are exactly 0 on ocean cells
whose bilinear coarse SIT (Y_base_phys) is exactly 0. Applying it to the saved fields.npz is identical to re-running inference
with --coarse-zero, except that every other cell keeps exactly the same members (a re-run would redraw the noise).

Per run (results/<batch>/<run>/eval_data/):
  - copies fields.npz and candidate_point_timeseries.csv to BACKUP/<batch>/<run>/ first (skipped if the copy exists);
  - rewrites fields.npz (compressed, as saved by run_pipeline): preds_all_phys and Y_pred_det_phys set to 0 on those cells,
    Y_pred_phys = mean of the constrained members; all other arrays unchanged;
  - updates the deterministic_unet / stochastic_unet_mean rows of candidate_point_timeseries.csv at their grid cells;
  - writes coarse_zero_applied.json (cell counts, max values zeroed). A run that already has it is skipped.
Training-time metrics.csv / sit_timeseries.csv are not rewritten (the paper's numbers are recomputed from fields.npz).

Undo: copy the backups back and delete coarse_zero_applied.json.

Usage: python evaluation/apply_coarse_zero_fields.py results/FOSI_final_ema [more batch dirs ...]
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

BACKUP = os.environ.get("CZ_BACKUP", "/glade/derecho/scratch/skygale/downscaling/coarse_zero_originals")


def update_point_rows(csv, ed, d, methods):
    """Rewrite the rows of candidate_point_timeseries.csv for `methods` ({csv method: fields key}) from the fields d. The file
    is sorted by (point, method, time), one row per sample; multi-member runs repeat each date once per member, in an
    arbitrary order (unstable sort), so the rows of every listed method are rewritten in date-then-member order (a stable
    sort of the samples by date), which keeps the methods aligned member by member. Returns the number of values changed."""
    df = pd.read_csv(csv)
    times = pd.to_datetime(pd.read_csv(os.path.join(ed, "sample_times.csv"))["time"]).to_numpy()
    order = np.argsort(times, kind="stable")
    n_chg = 0
    for (point, meth), g in df.groupby(["point", "method"], sort=False):
        if meth not in methods:
            continue
        if len(g) != len(times) or not (pd.to_datetime(g["time"]).to_numpy() == times[order]).all():
            raise ValueError(f"{csv}: rows for {point}/{meth} are not one per sample in date order")
        new = d[methods[meth]][order, 0, g["grid_iy"].astype(int).to_numpy(), g["grid_ix"].astype(int).to_numpy()]
        n_chg += int((g["value"].to_numpy() != new).sum())
        df.loc[g.index, "value"] = new
    df.to_csv(csv, index=False)
    return n_chg


def apply_run(run_dir):
    ed = os.path.join(run_dir, "eval_data")
    marker = os.path.join(ed, "coarse_zero_applied.json")
    if os.path.exists(marker):
        print(f"  skip (already applied): {run_dir}")
        return
    batch, run = os.path.basename(os.path.dirname(run_dir.rstrip("/"))), os.path.basename(run_dir.rstrip("/"))
    bdir = os.path.join(BACKUP, batch, run)
    os.makedirs(bdir, exist_ok=True)
    for name in ("fields.npz", "candidate_point_timeseries.csv"):
        src, dst = os.path.join(ed, name), os.path.join(bdir, name)
        if os.path.exists(src) and not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
            shutil.copy2(src, dst)

    f = np.load(os.path.join(ed, "fields.npz"))
    d = {k: f[k] for k in f.files}
    base, mask = d["Y_base_phys"], d["mask_test"]
    if mask.shape[0] != base.shape[0]:
        mask = np.broadcast_to(mask[:1], base.shape)
    zero = (base[:, :1] <= 0) & (mask[:, :1] <= 0.5)  # [N, 1, H, W], as coarse_zero_mask
    ocean = (mask[:, :1] <= 0.5).sum()
    stats = {"zero_cells_frac_of_ocean": float(zero.sum() / ocean),
             "truth_positive_in_zero_cells": int((d["Y_test_phys"][:, :1][zero] > 0).sum()),
             "truth_max_in_zero_cells": float(d["Y_test_phys"][:, :1][zero].max()) if zero.any() else 0.0}
    p = d["preds_all_phys"]  # [N, K, C, H, W]
    zk = np.broadcast_to(zero[:, None], p.shape)
    stats["members_positive_zeroed_frac"] = float((p[zk] > 0).mean()) if zero.any() else 0.0
    stats["members_max_zeroed"] = float(p[zk].max()) if zero.any() else 0.0
    p[zk] = 0.0
    d["preds_all_phys"] = p
    d["Y_pred_phys"] = p.mean(axis=1).astype(d["Y_pred_phys"].dtype)
    d["Y_pred_det_phys"] = np.where(zero, 0.0, d["Y_pred_det_phys"]).astype(d["Y_pred_det_phys"].dtype)
    tmp = os.path.join(ed, "fields.tmp.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, os.path.join(ed, "fields.npz"))

    csv = os.path.join(ed, "candidate_point_timeseries.csv")
    if os.path.exists(csv):
        stats["candidate_point_values_changed"] = update_point_rows(
            csv, ed, d, {"truth": "Y_test_phys", "bilinear": "Y_base_phys", "stochastic_unet_mean": "Y_pred_phys", "deterministic_unet": "Y_pred_det_phys"})
    stats["backup"] = bdir
    json.dump(stats, open(marker, "w"), indent=1)
    print(f"  {run}: " + ", ".join(f"{k}={v}" for k, v in stats.items() if k != "backup"))


if __name__ == "__main__":
    for bd in sys.argv[1:]:
        runs = sorted(os.path.dirname(os.path.dirname(r)) for r in glob.glob(os.path.join(bd, "*", "eval_data", "fields.npz")))
        print(f"=== {bd}: {len(runs)} runs")
        for r in runs:
            apply_run(r)
