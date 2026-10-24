#!/usr/bin/env python3
"""Which cells make the open-water rank histogram humped? (2026-10-22)

Rank of the truth among the K members (ties broken at random, as in build_4window_aggregate_figures.py), pooled over a
batch's 4 windows. Regimes are binned by ensemble-mean SIT. The open-water bin (ens mean < 0.15 m) is split by inputs only:
coarse bilinear SIT < 1 cm / >= 1 cm. Also split by whether the truth is exactly 0 (diagnostic only, uses truth).

Histogram printed as 7 groups of 3 ranks, each normalized so 1.00 = flat. "out" = fraction of truth outside the ensemble
(expected 2/(K+1)), "zt" = fraction of cells with truth exactly 0, "zm" = mean fraction of members exactly 0.

Optional env TOL (m): values below TOL in both truth and members are set to exactly 0 before ranking (detection-limit
convention; default 0 = no change).

Usage: [TOL=0.01] python evaluation/rank_hist_breakdown.py results/FOSI_final_ema [more batches ...]
"""
import glob
import os
import sys

import numpy as np

TOL = float(os.environ.get("TOL", "0"))
# CLAMP_TRUTH=1 (2026-10-24): clamp the truth at 0 like the members. Exact-zero truth comes back as about -2e-9 m from the
# normalize/de-normalize round trip in some cells, which then ranks below members that are clamped to exactly 0.
CLAMP_TRUTH = os.environ.get("CLAMP_TRUTH", "0") == "1"

EDGES = [0.0, 0.15, 0.5, 1.0, np.inf]
REGIMES = ["open (0-0.15)", "thin (0.15-0.5)", "moderate (0.5-1)", "thick (>1)"]
OPEN_SUB = ["open & coarse < 1 cm", "open & coarse >= 1 cm", "open & truth == 0", "open & truth > 0"]
GROUPS = ["all ocean"] + REGIMES + OPEN_SUB


def accumulate(run_dir, acc, rng, chunk=16):
    f = np.load(os.path.join(run_dir, "eval_data", "fields.npz"))
    preds, truth, mask, base = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"], f["Y_base_phys"]
    for t0 in range(0, preds.shape[0], chunk):
        ocean = mask[t0:t0 + chunk, 0] <= 0.5
        ens = np.moveaxis(preds[t0:t0 + chunk, :, 0], 1, -1)[ocean]
        y = truth[t0:t0 + chunk, 0][ocean]
        b = base[t0:t0 + chunk, 0][ocean]
        if CLAMP_TRUTH:
            y = np.maximum(y, 0)
        if TOL > 0:
            ens = np.where(ens < TOL, 0, ens)
            y = np.where(y < TOL, 0, y)
        K = ens.shape[1]
        below = (ens < y[:, None]).sum(axis=1)
        ties = (ens == y[:, None]).sum(axis=1)
        r = below + rng.integers(0, ties + 1)
        mu = ens.mean(axis=1)
        reg = np.digitize(mu, EDGES[1:-1])
        opn = reg == 0
        sels = {"all ocean": np.ones_like(opn)}
        sels.update({lab: reg == i for i, lab in enumerate(REGIMES)})
        sels.update({OPEN_SUB[0]: opn & (b < 0.01), OPEN_SUB[1]: opn & (b >= 0.01),
                     OPEN_SUB[2]: opn & (y == 0), OPEN_SUB[3]: opn & (y > 0)})
        zm = (ens == 0).mean(axis=1)
        for lab, s in sels.items():
            a = acc.setdefault(lab, {"h": np.zeros(K + 1), "zt": 0.0, "zm": 0.0})
            a["h"] += np.bincount(r[s], minlength=K + 1)
            a["zt"] += (y[s] == 0).sum()
            a["zm"] += zm[s].sum()


def report(name, acc):
    tot = acc["all ocean"]["h"].sum()
    K = len(acc["all ocean"]["h"]) - 1
    print(f"\n=== {name} TOL={TOL} CLAMP_TRUTH={int(CLAMP_TRUTH)}: rank histogram (K={K}), 7 groups of 3 ranks, 1.00 = flat ===")
    print(f"{'group':24s} {'%ocean':>6s} " + " ".join(f"{f'r{3*i}-{3*i+2}':>6s}" for i in range(7))
          + f" {'out':>6s} {'zt':>5s} {'zm':>5s}")
    for lab in GROUPS:
        a = acc[lab]
        n = a["h"].sum()
        if n == 0:
            continue
        g = a["h"].reshape(7, 3).sum(axis=1) / n * 7
        out = (a["h"][0] + a["h"][-1]) / n
        print(f"{lab:24s} {100 * n / tot:6.1f} " + " ".join(f"{v:6.2f}" for v in g)
              + f" {out:6.3f} {a['zt'] / n:5.2f} {a['zm'] / n:5.2f}")
    print(f"(flat 'out' = {2 / (K + 1):.3f})")


if __name__ == "__main__":
    for bd in sys.argv[1:]:
        acc, rng = {}, np.random.default_rng(0)
        runs = sorted(glob.glob(os.path.join(bd, "*", "eval_data", "fields.npz")))
        for r in runs:
            accumulate(os.path.dirname(os.path.dirname(r)), acc, rng)
        report(f"{os.path.basename(bd.rstrip('/'))} ({len(runs)} windows)", acc)
