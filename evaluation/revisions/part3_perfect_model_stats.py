#!/usr/bin/env python3
"""
Final round part 3 (2026-11-17): perfect-model numbers for the paper text (items 1-5), from saved eval_data/fields.npz.
Headline convention: test year 2021, ocean cells (mask <= 0.5), K = 20 members, ensemble mean = mean of the members (after the
clamp and the coarse-zero rule, as saved). Truth = Y_test_phys (rewritten from the raw Y files by restore_raw_truth.py).

Per EngressNet run:
  item 1  ensemble-mean bias (m); with --det, the det-MSE U-Net bias; bilinear bias (Y_base_phys)
  item 2  rank histogram by ice type, binned by ensemble-mean SIT: open water/frazil < 0.15 m, thin 0.15-0.5, moderate 0.5-1,
          thick > 1 m, and pooled over all ocean cells. Rank of the truth among the K members, ties broken at random (as in
          rank_hist_breakdown.py / the figures). Rank-0 and rank-K frequencies as multiples of the ideal 1/(K+1); % of truths
          outside the ensemble (ideal 2/(K+1) = 9.5% for K = 20).
  item 3  per type: spread = sqrt((K+1)/K * mean member variance), RMSE of the ensemble mean, spread-skill = spread / RMSE, CRPS
          (headline K^2 estimator). Open water split by the coarse input (bilinear SIT < 1 cm = far from the edge, >= 1 cm = near it):
          share of open-water cells, spread-skill, share of open-water squared error (as evaluation/diagnostics/
          open_water_spread_skill_split.py).
  item 4  cells where the bilinear coarse SIT is exactly 0 (coarse_zero_mask): their share of ocean cells and the maximum truth.

Usage (project root): python evaluation/revisions/part3_perfect_model_stats.py <label>=<eng run dir>[,<det run dir>] ...
Writes nothing; print to a file (submit_part3_perfect_model_stats.sh).
"""
import os
import sys

import numpy as np

EDGES = [0.15, 0.5, 1.0]
TYPES = ["open water/frazil (<0.15 m)", "thin (0.15-0.5 m)", "moderate (0.5-1 m)", "thick (>1 m)"]
CHUNK = 16


def crps_sum(ens, y):
    """Sum over cells of the CRPS, headline estimator: mean|X - y| - 1/(2K^2) sum_ij |X_i - X_j|. ens [n, K], y [n]."""
    K = ens.shape[1]
    s = np.sort(ens, axis=1)
    w = 2 * np.arange(1, K + 1) - K - 1
    return float((np.abs(ens - y[:, None]).mean(1) - (s * w).sum(1) / (K * K)).sum())


def stats(label, eng_dir, det_dir=None, seed=0):
    f = np.load(os.path.join(eng_dir, "eval_data", "fields.npz"))
    preds, truth, mask, base = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"], f["Y_base_phys"]
    det = np.load(os.path.join(det_dir, "eval_data", "fields.npz"))["Y_pred_phys"] if det_dir else None
    N, K = preds.shape[0], preds.shape[1]
    rng = np.random.default_rng(seed)
    groups = ["all ocean"] + TYPES + ["open, coarse < 1 cm", "open, coarse >= 1 cm"]
    A = {g: dict(n=0, h=np.zeros(K + 1), var=0.0, err2=0.0, crps=0.0) for g in groups}
    B = dict(n=0, eng=0.0, det=0.0, bil=0.0, cz_n=0, cz_max=-np.inf)
    for t0 in range(0, N, CHUNK):
        oc = mask[t0:t0 + CHUNK, 0] <= 0.5
        ens = np.moveaxis(preds[t0:t0 + CHUNK, :, 0], 1, -1)[oc].astype(np.float64)
        y = truth[t0:t0 + CHUNK, 0][oc].astype(np.float64)
        b = base[t0:t0 + CHUNK, 0][oc].astype(np.float64)
        mu = ens.mean(1)
        B["n"] += y.size
        B["eng"] += (mu - y).sum()
        B["bil"] += (b - y).sum()
        if det is not None:
            B["det"] += (det[t0:t0 + CHUNK, 0][oc] - y).sum()
        cz = b <= 0
        B["cz_n"] += int(cz.sum())
        if cz.any():
            B["cz_max"] = max(B["cz_max"], float(y[cz].max()))
        below = (ens < y[:, None]).sum(1)
        ties = (ens == y[:, None]).sum(1)
        r = below + rng.integers(0, ties + 1)
        var = ens.var(1, ddof=1)
        e2 = (mu - y) ** 2
        reg = np.digitize(mu, EDGES)
        sels = {"all ocean": np.ones(y.size, bool)}
        sels.update({t: reg == i for i, t in enumerate(TYPES)})
        sels["open, coarse < 1 cm"] = (reg == 0) & (b < 0.01)
        sels["open, coarse >= 1 cm"] = (reg == 0) & (b >= 0.01)
        for g, s in sels.items():
            a = A[g]
            a["n"] += int(s.sum())
            a["h"] += np.bincount(r[s], minlength=K + 1)
            a["var"] += var[s].sum()
            a["err2"] += e2[s].sum()
            a["crps"] += crps_sum(ens[s], y[s]) if s.any() else 0.0

    print(f"\n=== {label}: {eng_dir}" + (f"\n    det: {det_dir}" if det_dir else ""))
    print(f"K = {K}, days = {N}, ocean cells = {B['n']}")
    print(f"[1] bias (m): EngressNet ens mean {B['eng'] / B['n']:+.5f}" + (f" | det U-Net {B['det'] / B['n']:+.5f}" if det is not None else "")
          + f" | bilinear {B['bil'] / B['n']:+.5f}")
    print(f"[4] bilinear coarse SIT == 0: {100 * B['cz_n'] / B['n']:.1f}% of ocean cells, max truth there {B['cz_max']:.2e} m")
    ideal = 1 / (K + 1)
    print(f"[2,3] {'group':28s} {'%ocean':>6s} {'rank0 x':>8s} {'rank'+str(K)+' x':>8s} {'%out':>6s} {'spread':>7s} {'RMSE':>7s} "
          f"{'SSR':>6s} {'CRPS':>7s} {'%open':>6s} {'%open SE':>8s}")
    op = A[TYPES[0]]
    for g in groups:
        a = A[g]
        if a["n"] == 0:
            continue
        h = a["h"] / a["n"]
        spread = np.sqrt((K + 1) / K * a["var"] / a["n"])
        rmse = np.sqrt(a["err2"] / a["n"])
        extra = f" {100 * a['n'] / op['n']:6.1f} {100 * a['err2'] / op['err2']:8.1f}" if g.startswith("open") else ""
        print(f"      {g:28s} {100 * a['n'] / B['n']:6.1f} {h[0] / ideal:8.2f} {h[-1] / ideal:8.2f} {100 * (h[0] + h[-1]):6.1f} "
              f"{spread:7.4f} {rmse:7.4f} {spread / rmse:6.2f} {a['crps'] / a['n']:7.4f}{extra}")
    print(f"      (ideal: rank-0/rank-{K} multiple 1.00, % outside {200 * ideal:.1f})")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        label, dirs = arg.split("=", 1)
        d = dirs.split(",")
        stats(label, d[0], d[1] if len(d) > 1 else None)
