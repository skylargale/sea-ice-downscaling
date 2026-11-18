"""
Final round part 3, item 6 (2026-11-17): Table 5 bias and correlation against ICESat-2 for the FOSI-trained EngressNet (ensemble
mean, 20 members), the det-MSE U-Net and bilinear PIOMAS, on PIOMAS input; same QC and cell set as Table 5 (is2_crps_basis.py /
is2_spread_error.py: winters 15 Oct 2018 - 30 Apr 2022, extent >= 2 km, >= 20 segments, median segment uncertainty <= 1 m, every
estimate finite). Bias = mean(estimate - reference); r = Pearson over pass-cells. References: pass-cell means (Table 5) and
pass-cell medians (the "median offset": how much thicker each estimate is than the medians).
Writes results/revisions_2026-11/part3_is2_bias_r.csv.
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
qc = ((pcd.extent_km >= 2.0) & (pcd.n_seg >= 20) & (pcd.unc_med <= 1.0)).values
if "in_domain" in pcd:
    qc = qc & pcd.in_domain.values
E = np.load(f"{c.EV2}/{SETS['final2']['cells_eng']}_cells.npz")
D = np.load(f"{c.EV2}/{SETS['final2']['cells_det']}_cells.npz")
M, Dm, B = E["members"].astype(float), D["members"].astype(float), E["bilinear"].astype(float)
ok = qc & np.isfinite(M).all(1) & np.isfinite(Dm).all(1) & np.isfinite(B) & np.isfinite(pcd.h_mean.values) & np.isfinite(pcd.h_median.values)
est = {"EngressNet (ensemble mean)": M[ok].mean(1), "Deterministic U-Net": Dm[ok].mean(1), "Bilinear PIOMAS": B[ok]}
dates = pd.to_datetime(pcd["date"].values[ok]) if "date" in pcd else None
rows = []
for ref, col in (("pass-cell mean", "h_mean"), ("pass-cell median", "h_median")):
    y = pcd[col].values[ok].astype(float)
    for name, x in est.items():
        rows.append({"reference": ref, "estimate": name, "N cells": int(ok.sum()), "bias (m)": np.mean(x - y),
                     "median of (estimate - ref) (m)": np.median(x - y), "r": np.corrcoef(x, y)[0, 1],
                     "RMSE (m)": np.sqrt(np.mean((x - y) ** 2)), "mean estimate (m)": x.mean(), "mean ref (m)": y.mean()})
df = pd.DataFrame(rows)
df.to_csv(f"{V6}/results/revisions_2026-11/part3_is2_bias_r.csv", index=False)
pd.set_option("display.width", 220)
print("members:", M.shape[1], "| det members:", Dm.shape[1])
if dates is not None:
    print("dates:", dates.min().date(), "-", dates.max().date())
print(df.round(4).to_string(index=False))
y = pcd.h_mean.values[ok]; z = pcd.h_median.values[ok]
print(f"pass-cell mean - median: mean {np.mean(y - z):.4f} m")
