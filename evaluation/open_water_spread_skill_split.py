#!/usr/bin/env python3
"""Where does the open-water over-dispersion come from? (2026-10-24)

Open water = ocean cells with ensemble-mean SIT < 0.15 m (as in the regime figures). Split by the coarse bilinear SIT into far from
the ice edge (< 1 cm) and near it (>= 1 cm), pooled over a batch's windows, and report for each part: its share of open-water
cells, the spread-skill ratio sqrt((K+1)/K * mean member variance) / RMSE of the ensemble mean, and its shares of the open-water
summed member variance and summed squared error of the ensemble mean. These are the numbers behind the paper's open-water
calibration paragraph.

Usage: python evaluation/open_water_spread_skill_split.py results/MESA_final_ema results/FOSI_final_ema
"""
import glob
import os
import sys

import numpy as np

CHUNK = 64


def main(bd):
    acc, K = {}, None
    for path in sorted(glob.glob(os.path.join(bd, "*", "eval_data", "fields.npz"))):
        f = np.load(path)
        p, y, b, m = f["preds_all_phys"], f["Y_test_phys"][:, 0], f["Y_base_phys"][:, 0], f["mask_test"][:, 0] <= 0.5
        for t0 in range(0, p.shape[0], CHUNK):
            e = p[t0:t0 + CHUNK, :, 0]
            K = e.shape[1]
            mu, var = e.mean(axis=1), e.var(axis=1, ddof=1)
            err2 = (mu - y[t0:t0 + CHUNK]) ** 2
            opn = m[t0:t0 + CHUNK] & (mu < 0.15)
            bb = b[t0:t0 + CHUNK]
            for lab, s in [("far from edge (coarse < 1 cm)", opn & (bb < 0.01)),
                           ("near edge (coarse >= 1 cm)", opn & (bb >= 0.01)), ("all open water", opn)]:
                a = acc.setdefault(lab, [0.0, 0.0, 0])
                a[0] += var[s].sum()
                a[1] += err2[s].sum()
                a[2] += int(s.sum())
    tot = acc["all open water"]
    print(f"=== {os.path.basename(bd.rstrip('/'))} (K={K}) ===")
    for lab, (v, e2, n) in acc.items():
        print(f"  {lab:30s} cells {n / tot[2]:.2f} of open water   spread-skill {np.sqrt((K + 1) / K * v / e2):.2f}   "
              f"share of variance {v / tot[0]:.2f}   share of squared error {e2 / tot[1]:.2f}")


if __name__ == "__main__":
    for bd in sys.argv[1:]:
        main(bd)
