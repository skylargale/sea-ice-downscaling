#!/usr/bin/env python3
"""What sets the far-field open-water low-rank peak? (2026-10-23)

Far field = ocean cells with ensemble mean < 0.15 m and coarse bilinear SIT < 1 cm (as in rank_hist_breakdown.py).
Pooled over a batch's windows, prints:
  1. the truth distribution (fraction exactly 0, quantiles of the positive values);
  2. the member distribution, conditioned on truth == 0 / truth > 0 (fraction of members exactly 0, positive quantiles);
  3. how well inputs-only quantities (coarse bilinear SIT, coarse == 0 exactly) and model quantities (fraction of members
     > 0, ensemble mean) separate truth > 0 from truth == 0 (ROC AUC);
  4. the rank histogram by month.

Usage: python evaluation/far_field_diagnostics.py results/FOSI_final_ema
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

QS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
CHUNK = 16


def auc(score, label):
    """ROC AUC of score for binary label (rank formula, ties averaged)."""
    r = pd.Series(score).rank().to_numpy()
    n1 = label.sum()
    n0 = label.size - n1
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def collect(run_dir, rng, sub=0.05):
    f = np.load(os.path.join(run_dir, "eval_data", "fields.npz"))
    preds, truth, mask, base = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"], f["Y_base_phys"]
    months = pd.read_csv(os.path.join(run_dir, "eval_data", "sample_times.csv"))
    tcol = [c for c in months.columns if "time" in c.lower() or "date" in c.lower()][0]
    months = pd.to_datetime(months[tcol]).dt.month.to_numpy()
    out = {k: [] for k in ["y", "b", "pos", "mu", "r", "mon", "ens_sub"]}
    for t0 in range(0, preds.shape[0], CHUNK):
        sl = slice(t0, t0 + CHUNK)
        ocean = mask[sl, 0] <= 0.5
        ens = np.moveaxis(preds[sl, :, 0], 1, -1)
        mu = ens.mean(axis=-1)
        b = base[sl, 0]
        ff = ocean & (mu < 0.15) & (b < 0.01)
        e = ens[ff]
        y = truth[sl, 0][ff]
        below = (e < y[:, None]).sum(axis=1)
        ties = (e == y[:, None]).sum(axis=1)
        out["r"].append(below + rng.integers(0, ties + 1))
        out["y"].append(y)
        out["b"].append(b[ff])
        out["pos"].append((e > 0).mean(axis=1))
        out["mu"].append(mu[ff])
        out["mon"].append(np.broadcast_to(months[sl][:, None, None], ff.shape)[ff])
        keep = rng.random(len(y)) < sub
        out["ens_sub"].append(np.concatenate([e[keep], y[keep, None]], axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}


def qline(x):
    if x.size == 0:
        return "(none)"
    return " ".join(f"q{int(100 * q)}={v:.2e}" for q, v in zip(QS, np.quantile(x, QS)))


def main(bd):
    rng = np.random.default_rng(0)
    runs = sorted(glob.glob(os.path.join(bd, "*", "eval_data", "fields.npz")))
    parts = [collect(os.path.dirname(os.path.dirname(r)), rng) for r in runs]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    y, b = d["y"], d["b"]
    K = d["ens_sub"].shape[1] - 1
    print(f"=== {os.path.basename(bd.rstrip('/'))} far field ({len(runs)} windows), {y.size:,} cells ===")

    print("\n1. truth")
    print(f"  exactly 0: {np.mean(y == 0):.3f}   in (0,1e-6]: {np.mean((y > 0) & (y <= 1e-6)):.3f}   "
          f"(1e-6,1e-3]: {np.mean((y > 1e-6) & (y <= 1e-3)):.3f}   (1e-3,1e-2]: {np.mean((y > 1e-3) & (y <= 1e-2)):.3f}   "
          f"(1e-2,0.05]: {np.mean((y > 1e-2) & (y <= 0.05)):.3f}   >0.05: {np.mean(y > 0.05):.3f}")
    print(f"  positive truth: {qline(y[y > 0])}")

    print("\n2. members (subsample)")
    es = d["ens_sub"]
    e, yt = es[:, :K], es[:, K]
    for lab, s in [("truth == 0", yt == 0), ("truth > 0", yt > 0)]:
        m = e[s]
        print(f"  {lab:10s}: members exactly 0 {np.mean(m == 0):.3f}; positive members: {qline(m[m > 0])}")
    s = yt > 0
    print(f"  truth > 0: fraction of members below truth {np.mean(e[s] < yt[s, None]):.3f}, "
          f"median |member - truth| {np.median(np.abs(e[s] - yt[s, None])):.2e} m")

    print("\n3. ROC AUC for truth > 0 (0.5 = no skill)")
    lab = y > 0
    print(f"  coarse bilinear SIT          {auc(b, lab):.3f}   (coarse exactly 0 in {np.mean(b == 0):.3f} of cells; "
          f"P(truth>0 | coarse==0) {lab[b == 0].mean() if np.any(b == 0) else np.nan:.3f}, "
          f"P(truth>0 | coarse>0) {lab[b > 0].mean():.3f})")
    print(f"  fraction of members > 0      {auc(d['pos'], lab):.3f}")
    print(f"  ensemble mean                {auc(d['mu'], lab):.3f}")
    for lo, hi in [(0, 0), (0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2)]:
        s = (b == 0) if hi == 0 else ((b > lo) & (b <= hi))
        if s.any():
            print(f"  coarse in {'{0}' if hi == 0 else f'({lo:g},{hi:g}]':14s} cells {np.mean(s):.3f}: "
                  f"P(truth>0) {lab[s].mean():.3f}, mean frac members>0 {d['pos'][s].mean():.3f}")

    print("\n4. rank histogram by month (7 groups of 3 ranks, 1.00 = flat)")
    for mo in range(1, 13):
        s = d["mon"] == mo
        if s.sum() == 0:
            continue
        h = np.bincount(d["r"][s], minlength=K + 1)
        g = h.reshape(7, 3).sum(axis=1) / h.sum() * 7
        print(f"  {mo:2d}  n={s.sum():9,d}  " + " ".join(f"{v:5.2f}" for v in g)
              + f"   truth0 {np.mean(y[s] == 0):.2f}")


if __name__ == "__main__":
    for bd in sys.argv[1:]:
        main(bd)
