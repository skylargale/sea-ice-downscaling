#!/usr/bin/env python3
"""Replace the saved truth with the original truth from the data file (2026-10-24).

run_pipeline saves Y_test_phys after a normalize / de-normalize round trip ((Y - mean) / (std + 1e-6), then back), which adds
float32 rounding of up to ~1e-6 m. Truth that is exactly 0 in the data file comes back as about +-2e-9 m in ~23% of those cells
(FOSI 2021), so in rank histograms it falls strictly below or above members that are exactly 0 (clamped, or the coarse-zero
rule). This script reads the original Y for the test samples (same year selection, crop, member-major order and 6 m cap as
run_pipeline) and writes it as Y_test_phys; nothing else in fields.npz changes. The truth and model rows of
candidate_point_timeseries.csv are rewritten from the fields (matched by position within each point/method).

Safety: a run is skipped (and reported) if the saved and original truth differ by more than 1e-5 m anywhere, which would mean a
sample-ordering or crop mismatch rather than rounding. fields.npz is copied to BACKUP first. Runs with
raw_truth_restored.json are skipped.

Usage: python evaluation/restore_raw_truth.py results/FOSI_final_ema [more perfect-model batch dirs ...]
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr

BACKUP = os.environ.get("RT_BACKUP", "/glade/derecho/scratch/skygale/downscaling/coarse_zero_originals/pre_rawtruth")
TOL = 1e-5


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


def crop(coord, lo, hi):
    inside = np.where((coord >= lo) & (coord <= hi))[0]
    return slice(int(inside.min()), int(inside.max()) + 1)


def raw_truth(cfg):
    path = cfg.get("test_y_path") or cfg["y_path"]
    da = xr.open_dataset(path)["Y"]
    years = set(int(y) for y in cfg["test_years"])
    tsel = np.array([t.year in years for t in da["time"].values])
    sd = cfg["subdomain"]
    la = crop(da.lat.values, sd["lat_min"], sd["lat_max"])
    lo = crop(da.lon.values % 360, sd["lon_min"] % 360, sd["lon_max"] % 360)
    y = da.isel(time=np.where(tsel)[0], lat=la, lon=lo).values  # (N, T, C, H, W)
    y = np.clip(y, None, 6.0)
    N, T, C, H, W = y.shape
    return y.reshape(N * T, C, H, W).astype(np.float32)  # member-major, as split_train_test


def restore_run(run_dir):
    ed = os.path.join(run_dir, "eval_data")
    marker = os.path.join(ed, "raw_truth_restored.json")
    if os.path.exists(marker):
        print(f"  skip (already restored): {run_dir}")
        return
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    f = np.load(os.path.join(ed, "fields.npz"))
    d = {k: f[k] for k in f.files}
    raw = raw_truth(cfg)
    saved = d["Y_test_phys"]
    if raw.shape != saved.shape:
        print(f"  !!! shape mismatch {raw.shape} vs {saved.shape}, skipped: {run_dir}")
        return
    diff = float(np.nanmax(np.abs(raw - saved)))
    if not np.isfinite(diff) or diff > TOL:
        print(f"  !!! max |saved - raw| = {diff:.3g} m > {TOL:g}, skipped: {run_dir}")
        return
    batch, run = os.path.basename(os.path.dirname(run_dir.rstrip("/"))), os.path.basename(run_dir.rstrip("/"))
    bdir = os.path.join(BACKUP, batch, run)
    os.makedirs(bdir, exist_ok=True)
    for name in ("fields.npz", "candidate_point_timeseries.csv"):
        src, dst = os.path.join(ed, name), os.path.join(bdir, name)
        if os.path.exists(src) and not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
            shutil.copy2(src, dst)
    mask = d["mask_test"]
    ocean = np.broadcast_to(mask[:1] if mask.shape[0] != saved.shape[0] else mask, saved.shape) <= 0.5
    stats = {"max_abs_change_m": diff,
             "ocean_exact_zero_before": float((saved[ocean] == 0).mean()),
             "ocean_exact_zero_after": float((raw[ocean] == 0).mean()),
             "ocean_negative_before": float((saved[ocean] < 0).mean()),
             "ocean_negative_after": float((raw[ocean] < 0).mean())}
    d["Y_test_phys"] = raw
    tmp = os.path.join(ed, "fields.tmp.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, os.path.join(ed, "fields.npz"))

    csv = os.path.join(ed, "candidate_point_timeseries.csv")
    if os.path.exists(csv):
        # truth from the original data; the model rows are rewritten too, which repairs multi-member (MESA) files that
        # apply_coarse_zero_fields.py had filled with the last member's value for every member (date lookup, fixed)
        stats["candidate_point_values_changed"] = update_point_rows(
            csv, ed, d, {"truth": "Y_test_phys", "bilinear": "Y_base_phys", "stochastic_unet_mean": "Y_pred_phys", "deterministic_unet": "Y_pred_det_phys"})
    stats["backup"] = bdir
    json.dump(stats, open(marker, "w"), indent=1)
    print(f"  {run}: " + ", ".join(f"{k}={v:.4g}" for k, v in stats.items() if k != "backup"))


if __name__ == "__main__":
    for bd in sys.argv[1:]:
        runs = sorted(os.path.dirname(os.path.dirname(r)) for r in glob.glob(os.path.join(bd, "*", "eval_data", "fields.npz")))
        print(f"=== {bd}: {len(runs)} runs")
        for r in runs:
            restore_run(r)
