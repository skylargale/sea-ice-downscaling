#!/usr/bin/env python3
"""Where does the open-water/frazil CRPS come from? (2026-10-21)

Open-water/frazil = ocean cells with truth SIT < 0.15 m (the regime used in compare_output_floor.py). For a batch's 4 windows,
pools these cells and splits their fair CRPS by:
  - truth value: < 1 mm, 1 mm-1 cm, 1-5 cm, 5-15 cm
  - distance to the ice edge: within EDGE_PX high-res cells of a truth >= 0.15 m cell, or farther
  - coarse input: bilinear baseline < 1 cm (coarse model has no ice here) or >= 1 cm
For each group: share of open-water cells and of open-water CRPS, mean CRPS, the CRPS of the clamped member median alone
(no spread), bias and MAE of the ensemble mean, the two CRPS terms (mean |member - truth| and the spread credit), the
fraction of members exactly 0, and how often the truth falls below / above every member.

"Shift oracle" rows: CRPS after subtracting one constant from every open-water member and re-clamping at 0. Uses the
truth to pick the cells, so it's an upper bound on what a learned downward shift (the proposed shift head) could gain.

Usage: python evaluation/open_water_crps_breakdown.py results/FOSI_final_ema [results/FOSI_final_ema_relu ...]
"""
import glob
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter

import compare_output_floor as cof

OPEN_MAX = 0.15
EDGE_PX = 11  # 11x11 window, i.e. within 5 cells (~40 km) of a truth >= 0.15 m cell
TRUTH_BINS = [(-np.inf, 0.001, "truth < 1 mm"), (0.001, 0.01, "1 mm - 1 cm"), (0.01, 0.05, "1 - 5 cm"), (0.05, OPEN_MAX, "5 - 15 cm")]
SHIFTS = [0.005, 0.01, 0.02, 0.04]
FIELDS = ["n", "crps", "crps_med", "bias", "mae", "term1", "term2", "zero", "below", "above"] + [f"shift_{s}" for s in SHIFTS]


def new_group():
    return {k: 0.0 for k in FIELDS}


def add(group, sel, d):
    group["n"] += sel.sum()
    for k, v in d.items():
        group[k] += v[sel].sum()


def accumulate(run_dir, groups, chunk=16):
    f = np.load(os.path.join(run_dir, "eval_data", "fields.npz"))
    preds, truth, mask, base = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"], f["Y_base_phys"]
    for t0 in range(0, preds.shape[0], chunk):
        tr = truth[t0:t0 + chunk, 0]
        ocean = mask[t0:t0 + chunk, 0] <= 0.5
        sel = ocean & (tr < OPEN_MAX)
        ice = (tr >= OPEN_MAX) & ocean
        near = np.stack([maximum_filter(ice[i].astype(np.uint8), size=EDGE_PX) > 0 for i in range(ice.shape[0])])
        ens = np.moveaxis(preds[t0:t0 + chunk, :, 0], 1, -1)[sel].astype(np.float64)
        y = tr[sel].astype(np.float64)
        b = base[t0:t0 + chunk, 0][sel]
        nr = near[sel]
        ens.sort(axis=1)
        K = ens.shape[1]
        mu = ens.mean(axis=1)
        med = np.median(ens, axis=1)
        crps = cof.fair_crps(ens, y)
        term1 = np.abs(ens - y[:, None]).mean(axis=1)
        d = {"crps": crps, "crps_med": np.abs(med - y), "bias": mu - y, "mae": np.abs(mu - y), "term1": term1,
             "term2": term1 - crps, "zero": (ens == 0).mean(axis=1), "below": (y < ens[:, 0]).astype(float),
             "above": (y > ens[:, -1]).astype(float)}
        for s in SHIFTS:
            d[f"shift_{s}"] = cof.fair_crps(np.clip(ens - s, 0, None), y)
        add(groups["all open water"], np.ones_like(y, bool), d)
        for lo, hi, lab in TRUTH_BINS:
            add(groups[lab], (y >= lo) & (y < hi), d)
        add(groups["near ice edge"], nr, d)
        add(groups["far from ice edge"], ~nr, d)
        add(groups["coarse SIT < 1 cm"], b < 0.01, d)
        add(groups["coarse SIT >= 1 cm"], b >= 0.01, d)
        add(groups["far edge & coarse < 1 cm"], ~nr & (b < 0.01), d)


def report(name, groups):
    tot_n, tot_c = groups["all open water"]["n"], groups["all open water"]["crps"]
    print(f"\n=== {name}: open water/frazil (truth < {OPEN_MAX} m) CRPS breakdown ===")
    print(f"{'group':26s} {'%cells':>6s} {'%CRPS':>6s} {'CRPS':>7s} {'medCRPS':>7s} {'bias':>8s} {'MAE':>7s} {'|x-y|':>7s} "
          f"{'spread':>7s} {'zero':>5s} {'<min':>5s} {'>max':>5s}")
    for lab, g in groups.items():
        n = max(g["n"], 1)
        print(f"{lab:26s} {100 * g['n'] / tot_n:6.1f} {100 * g['crps'] / tot_c:6.1f} {g['crps'] / n:7.4f} {g['crps_med'] / n:7.4f} "
              f"{g['bias'] / n:+8.4f} {g['mae'] / n:7.4f} {g['term1'] / n:7.4f} {g['term2'] / n:7.4f} {g['zero'] / n:5.2f} "
              f"{g['below'] / n:5.2f} {g['above'] / n:5.2f}")
    print("shift oracle (subtract s from every open-water member, re-clamp; truth picks the cells):")
    for lab, g in groups.items():
        n = max(g["n"], 1)
        print(f"  {lab:26s} " + "  ".join(f"s={s}: {g[f'shift_{s}'] / n:.4f}" for s in SHIFTS))


def main(batch_dirs):
    for bd in batch_dirs:
        runs = sorted(glob.glob(os.path.join(bd, "*", "eval_data", "fields.npz")))
        groups = {lab: new_group() for lab in ["all open water"] + [t[2] for t in TRUTH_BINS] +
                  ["near ice edge", "far from ice edge", "coarse SIT < 1 cm", "coarse SIT >= 1 cm", "far edge & coarse < 1 cm"]}
        for r in runs:
            print(f"  {r}", flush=True)
            accumulate(os.path.dirname(os.path.dirname(r)), groups)
        report(f"{os.path.basename(bd.rstrip('/'))} ({len(runs)} windows)", groups)


if __name__ == "__main__":
    main(sys.argv[1:])
