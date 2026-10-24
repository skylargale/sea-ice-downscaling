#!/usr/bin/env python3
"""No-training check (2026-10-23): far-field truth is exactly 0 wherever the coarse bilinear SIT is exactly 0. Rank
histograms of the baseline split by coarse == 0 / > 0 (input-only split), and the unsplit groups if members were exactly 0
wherever coarse == 0 (what a model with that constraint built in would give, if nothing else changed).

Usage: python evaluation/far_field_coarse_zero_check.py results/FOSI_final_ema
"""
import glob
import os
import sys

import numpy as np

EDGES = [0.0, 0.15, 0.5, 1.0, np.inf]
GROUPS = ["all ocean", "open (0-0.15)", "thin (0.15-0.5)", "moderate (0.5-1)", "thick (>1)", "open & coarse < 1 cm",
          "open & coarse >= 1 cm", "open & coarse == 0", "open & 0 < coarse < 1 cm"]


def ranks(ens, y, rng):
    below = (ens < y[:, None]).sum(axis=1)
    ties = (ens == y[:, None]).sum(axis=1)
    return below + rng.integers(0, ties + 1)


def accumulate(run_dir, acc, rng, chunk=16):
    f = np.load(os.path.join(run_dir, "eval_data", "fields.npz"))
    preds, truth, mask, base = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"], f["Y_base_phys"]
    for t0 in range(0, preds.shape[0], chunk):
        ocean = mask[t0:t0 + chunk, 0] <= 0.5
        ens0 = np.moveaxis(preds[t0:t0 + chunk, :, 0], 1, -1)[ocean]
        y = truth[t0:t0 + chunk, 0][ocean]
        b = base[t0:t0 + chunk, 0][ocean]
        for variant, ens in [("baseline", ens0), ("zero where coarse == 0", np.where((b == 0)[:, None], 0, ens0))]:
            r = ranks(ens, y, rng)
            reg = np.digitize(ens.mean(axis=1), EDGES[1:-1])
            opn = reg == 0
            sels = {"all ocean": np.ones_like(opn), "open & coarse < 1 cm": opn & (b < 0.01),
                    "open & coarse >= 1 cm": opn & (b >= 0.01), "open & coarse == 0": opn & (b == 0),
                    "open & 0 < coarse < 1 cm": opn & (b > 0) & (b < 0.01)}
            sels.update({g: reg == i for i, g in enumerate(GROUPS[1:5])})
            for g, s in sels.items():
                a = acc.setdefault(variant, {}).setdefault(g, {"h": np.zeros(ens.shape[1] + 1), "crps": 0.0})
                a["h"] += np.bincount(r[s], minlength=ens.shape[1] + 1)


def crps_all(run_dirs):
    """All-ocean ensemble CRPS (fair-free, K members) for both variants."""
    tot = {"baseline": 0.0, "zero where coarse == 0": 0.0}
    n = 0
    for rd in run_dirs:
        f = np.load(os.path.join(rd, "eval_data", "fields.npz"))
        preds, truth, mask, base = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"], f["Y_base_phys"]
        for t0 in range(0, preds.shape[0], 8):
            ocean = mask[t0:t0 + 8, 0] <= 0.5
            ens0 = np.moveaxis(preds[t0:t0 + 8, :, 0], 1, -1)[ocean]
            y = truth[t0:t0 + 8, 0][ocean]
            b = base[t0:t0 + 8, 0][ocean]
            for v, ens in [("baseline", ens0), ("zero where coarse == 0", np.where((b == 0)[:, None], 0, ens0))]:
                t1 = np.abs(ens - y[:, None]).mean(axis=1)
                s = np.sort(ens, axis=1)
                K = s.shape[1]
                w = 2 * np.arange(1, K + 1) - K - 1
                t2 = (s * w).sum(axis=1) / K ** 2
                tot[v] += (t1 - t2).sum()
            n += y.size
    return {v: t / n for v, t in tot.items()}


if __name__ == "__main__":
    bd = sys.argv[1]
    runs = sorted(os.path.dirname(os.path.dirname(r)) for r in glob.glob(os.path.join(bd, "*", "eval_data", "fields.npz")))
    acc, rng = {}, np.random.default_rng(0)
    for rd in runs:
        accumulate(rd, acc, rng)
    for variant, a in acc.items():
        tot = a["all ocean"]["h"].sum()
        print(f"\n=== {os.path.basename(bd.rstrip('/'))} ({len(runs)} windows), {variant}: 7 groups of 3 ranks, 1.00 = flat ===")
        for g in GROUPS:
            h = a[g]["h"]
            if h.sum() == 0:
                continue
            gg = h.reshape(7, 3).sum(axis=1) / h.sum() * 7
            print(f"{g:26s} {100 * h.sum() / tot:6.1f} " + " ".join(f"{v:6.2f}" for v in gg) + f"  out {(h[0] + h[-1]) / h.sum():.3f}")
    print("\n(flat 'out' = 0.095)")
    for v, c in crps_all(runs).items():
        print(f"all-ocean CRPS, {v}: {c:.4f}")
