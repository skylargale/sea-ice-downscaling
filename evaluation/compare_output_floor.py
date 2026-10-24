#!/usr/bin/env python3
"""Per-regime comparison of batches (written 2026-10-21 for the relu output-floor test; since 2026-10-24 also the regime table for
the final batches, see submit_cz_metrics.sh).

For each batch, pools the 4 windows' ocean cells (test year 2021) from eval_data/fields.npz and reports, per regime:
ensemble-mean RMSE and bias, fair CRPS, RMS spread-skill ratio sqrt((K+1)/K * mean var) / RMSE (1 = calibrated), and the
fraction of members that are exactly 0. Regimes: truth-binned for RMSE/CRPS/bias, ensemble-mean-binned for spread-skill
(binning by truth distorts calibration diagnostics, see build_4window_aggregate_figures.py).

Ice-edge risk check (written for the relu floor, where members clamped to 0 get no gradient, so a region can get stuck at zero ice):
"missed ice" = fraction of cells with truth above a threshold where every member is exactly 0.

Usage: python evaluation/compare_output_floor.py results/FOSI_final_ema results/FOSI_final_ema_relu results/FOSI_final_ema_relul05
"""
import glob
import os
import sys

import numpy as np

SIT_BIN_EDGES = [0.0, 0.15, 0.5, 1.0, np.inf]
SIT_BIN_LABELS = ["open (0-0.15)", "thin (0.15-0.5)", "moderate (0.5-1)", "thick (>1)"]
MISSED_THRESHOLDS = [0.05, 0.15, 0.5]


def fair_crps(ens, y):
    """Fair CRPS per point; ens (M, K) sorted along K."""
    K = ens.shape[1]
    w = (2 * np.arange(1, K + 1) - K - 1).astype(np.float64)
    mean_abs = np.abs(ens - y[:, None]).mean(axis=1)
    spread_term = (ens * w).sum(axis=1) / (K * (K - 1))  # = 0.5 * fair E|X - X'|
    return mean_abs - spread_term


def new_acc(nb):
    z = lambda: np.zeros(nb)
    return {"n_t": z(), "se_t": z(), "err_t": z(), "crps_t": z(), "zero_t": z(),
            "n_f": z(), "se_f": z(), "var_f": z(), "missed_n": np.zeros(len(MISSED_THRESHOLDS)),
            "missed_d": np.zeros(len(MISSED_THRESHOLDS)), "K": 0}


def accumulate(run_dir, acc, chunk=16, transforms=None):
    """transforms: optional list of (acc, fn) pairs; fn(ens (M, K)) -> modified ens, each scored into its own acc in the
    same pass over fields.npz (used by open_water_gate_check.py). None = score the saved members into `acc`."""
    f = np.load(os.path.join(run_dir, "eval_data", "fields.npz"))
    preds, truth, mask = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"]
    for t0 in range(0, preds.shape[0], chunk):
        ocean = mask[t0:t0 + chunk, 0] <= 0.5
        ens0 = np.moveaxis(preds[t0:t0 + chunk, :, 0], 1, -1)[ocean].astype(np.float64)  # (M, K)
        y = truth[t0:t0 + chunk, 0][ocean].astype(np.float64)
        for a, fn in (transforms or [(acc, None)]):
            _score_chunk(fn(ens0) if fn is not None else ens0.copy(), y, a)


def _score_chunk(ens, y, acc):
    acc["K"] = ens.shape[1]
    ens.sort(axis=1)
    mu = ens.mean(axis=1)
    var = ens.var(axis=1, ddof=1)
    crps = fair_crps(ens, y)
    allzero = (ens == 0).all(axis=1)
    zero_frac = (ens == 0).mean(axis=1)
    bt = np.digitize(y, SIT_BIN_EDGES[1:-1])
    bf = np.digitize(mu, SIT_BIN_EDGES[1:-1])
    for b in range(len(SIT_BIN_LABELS)):
        s = bt == b
        acc["n_t"][b] += s.sum()
        acc["se_t"][b] += ((mu[s] - y[s]) ** 2).sum()
        acc["err_t"][b] += (mu[s] - y[s]).sum()
        acc["crps_t"][b] += crps[s].sum()
        acc["zero_t"][b] += zero_frac[s].sum()
        s = bf == b
        acc["n_f"][b] += s.sum()
        acc["se_f"][b] += ((mu[s] - y[s]) ** 2).sum()
        acc["var_f"][b] += var[s].sum()
    for i, thr in enumerate(MISSED_THRESHOLDS):
        s = y > thr
        acc["missed_d"][i] += s.sum()
        acc["missed_n"][i] += (allzero & s).sum()


def report(name, acc):
    K = acc["K"]
    print(f"\n=== {name} (K={K}) ===")
    print(f"{'regime':18s} {'frac':>6s} {'RMSE':>7s} {'bias':>8s} {'CRPS':>7s} {'zero mbrs':>9s} | {'ens-mean bin SSR':>16s}")
    tot = acc["n_t"].sum()
    for b, lab in enumerate(SIT_BIN_LABELS):
        n = max(acc["n_t"][b], 1)
        ssr = np.sqrt((K + 1) / K * acc["var_f"][b] / max(acc["n_f"][b], 1)) / np.sqrt(acc["se_f"][b] / max(acc["n_f"][b], 1))
        print(f"{lab:18s} {acc['n_t'][b] / tot:6.3f} {np.sqrt(acc['se_t'][b] / n):7.4f} {acc['err_t'][b] / n:+8.4f} "
              f"{acc['crps_t'][b] / n:7.4f} {acc['zero_t'][b] / n:9.3f} | {ssr:16.3f}")
    ssr_all = np.sqrt((K + 1) / K * acc["var_f"].sum() / acc["n_f"].sum()) / np.sqrt(acc["se_f"].sum() / acc["n_f"].sum())
    print(f"{'all ocean':18s} {1.0:6.3f} {np.sqrt(acc['se_t'].sum() / tot):7.4f} {acc['err_t'].sum() / tot:+8.4f} "
          f"{acc['crps_t'].sum() / tot:7.4f} {acc['zero_t'].sum() / tot:9.3f} | {ssr_all:16.3f}")
    print("missed ice (all members exactly 0): " + ", ".join(
        f"truth>{thr}: {acc['missed_n'][i] / max(acc['missed_d'][i], 1):.4%}" for i, thr in enumerate(MISSED_THRESHOLDS)))


def main(batch_dirs):
    for bd in batch_dirs:
        runs = sorted(glob.glob(os.path.join(bd, "*", "eval_data", "fields.npz")))
        if not runs:
            print(f"\n=== {bd}: no fields.npz found ===")
            continue
        acc = new_acc(len(SIT_BIN_LABELS))
        for r in runs:
            print(f"  {r}", flush=True)
            accumulate(os.path.dirname(os.path.dirname(r)), acc)
        report(f"{os.path.basename(bd.rstrip('/'))} ({len(runs)} windows)", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
