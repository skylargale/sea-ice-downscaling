"""
Final round, item 1: Table 5 "Spread/error" for the FOSI-trained EngressNet on PIOMAS input, against ICESat-2 pass-cell means
and medians (winters Oct 2018 - May 2022, same QC and cell set as Table 5 / is2_crps_basis.py: 161,376 pass-cells).

Definition (as in observations/notebook_src/engressnet_obs_evaluation_src.py, "spread/error + obs noise"):
    sqrt( (K+1)/K * mean member variance + mean sigma_obs(m)^2 ) / RMSE of the ensemble mean
sigma_obs(h): ICESat-2 random error per thickness bin, from the 10 km along-track structure function extrapolated to zero lag
(calibration_options_compare.noise_fn), built from the bin statistic matching the reference (means or medians).
Also reported without the observation term. Computed for final2 (1 model x 20 members) and, as a check against the old Table 5
value (0.445), for final1 (4 windows x 20 members pooled = 80) on the same cells.
Writes results/revisions_2026-11/is2_spread_error.csv.
"""
import sys

import numpy as np
import pandas as pd

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, f"{V6}/observations/processing")
sys.path.insert(0, f"{V6}/evaluation/revisions")
import calibration_options_compare as c  # noqa: E402
from final_set import SETS  # noqa: E402

pcd = c.PCD
qc = ((pcd.extent_km >= 2.0) & (pcd.n_seg >= 20) & (pcd.unc_med <= 1.0)).values & pcd.in_domain.values if "in_domain" in pcd \
    else ((pcd.extent_km >= 2.0) & (pcd.n_seg >= 20) & (pcd.unc_med <= 1.0)).values
L = {}
for name in ("final1", "final2"):
    E = np.load(f"{c.EV2}/{SETS[name]['cells_eng']}_cells.npz")
    D = np.load(f"{c.EV2}/{SETS[name]['cells_det']}_cells.npz")
    L[name] = (E["members"].astype(float), D["members"].astype(float), E["bilinear"].astype(float))
print("cells in file:", len(pcd), {k: v[0].shape for k, v in L.items()})

rows = []
for name, (M, Dm, B) in L.items():
    # Table 5 cell set for this model (is2_crps_basis.py): QC + every estimate finite + both references finite
    ok = qc & np.isfinite(M).all(1) & np.isfinite(Dm).all(1) & np.isfinite(B) & np.isfinite(pcd.h_mean.values) & np.isfinite(pcd.h_median.values)
    engb = np.load(f"{c.EV2}/{SETS[name]['cells_eng']}_bins.npz")["members"].astype(float)
    okb = c.qcB & np.isfinite(engb).all(1)
    W = c.contiguous_windows(okb)
    for ref, col, bcol in (("pass-cell mean", "h_mean", "h_mean"), ("pass-cell median", "h_median", "h_median")):
        sig = c.noise_fn(c.BIN[bcol].values.astype(float), W)
        y, X = pcd[col].values[ok], M[ok]
        K = X.shape[1]
        m = X.mean(1)
        rmse = np.sqrt(np.mean((m - y) ** 2))
        s2 = X.var(1, ddof=1).mean()
        rows.append({"model": name, "reference": ref, "N cells": int(ok.sum()), "members": K, "RMSE of mean": rmse,
                     "corrected spread": np.sqrt((K + 1) / K * s2), "mean obs sigma": np.sqrt(np.mean(sig(m) ** 2)),
                     "spread/error": np.sqrt((K + 1) / K * s2) / rmse,
                     "spread/error + obs noise": np.sqrt((K + 1) / K * s2 + np.mean(sig(m) ** 2)) / rmse})
df = pd.DataFrame(rows)
df.to_csv(f"{V6}/results/revisions_2026-11/is2_spread_error.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))
