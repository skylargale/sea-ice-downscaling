"""
Revisions 2026-11, follow-up 1: Table 5 CRPS on one consistent basis, against ICESat-2 pass-cell MEANS and MEDIANS.
PIOMAS input, winters Oct 2018 - May 2022, FOSI-trained final checkpoints (4 windows), same QC as Table 5 / scoreboard.csv
(extent >= 2 km, >= 20 segments, median segment uncertainty <= 1 m), cells where every estimate is finite.
  bilinear PIOMAS: MAE (point value)
  deterministic U-Net (det-MSE + EMA): MAE, per window then averaged over the 4 windows   [primary]
  EngressNet (EMA): CRPS of each window's K = 20 members (fair estimator, as Table 5), averaged over the 4 windows [primary]
  pooled versions for reference: deterministic = MAE of the 4-window mean; EngressNet = CRPS of all 80 members
Writes results/revisions_2026-11/is2_crps_basis.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/skygale/projects/SeaIceDownscaling/evaluation/revisions")
from final_set import FS, NAME as FINAL_SET  # noqa: E402
SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"
IS2C = "/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells"
OUT = f"/glade/work/skygale/projects/SeaIceDownscaling/results/revisions_2026-11/is2_crps_basis{SFX}.csv"
pcd = pd.read_parquet(f"{IS2C}/pass_cells.parquet").query("in_domain").reset_index(drop=True)
qc = ((pcd.extent_km >= 2.0) & (pcd.n_seg >= 20) & (pcd.unc_med <= 1.0)).values
E = np.load(f"{IS2C}/eval/{FS['cells_eng']}_cells.npz"); D = np.load(f"{IS2C}/eval/{FS['cells_det']}_cells.npz")
K = int(E["K"]); NU = len(E["windows"]); KD = int(D["K"])
M, B = E["members"].astype(float), E["bilinear"].astype(float)
Dm = D["members"].astype(float)[:, 0::KD]  # det: one distinct value per unit (K_EVAL identical passes)
assert M.shape[1] == NU * K and Dm.shape[1] == NU, (M.shape, Dm.shape, NU, K, KD)
ok = qc & np.isfinite(M).all(1) & np.isfinite(Dm).all(1) & np.isfinite(B) & np.isfinite(pcd.h_mean.values) & np.isfinite(pcd.h_median.values)


def crps_fair(X, y):
    k = X.shape[1]
    w = 2 * np.arange(1, k + 1) - k - 1
    return (np.abs(X - y[:, None]).mean(1) - (np.sort(X, 1) * w).sum(1) / (k * (k - 1))).mean()


rows = []
for ref, col in (("pass-cell mean", "h_mean"), ("pass-cell median", "h_median")):
    y = pcd[col].values[ok]
    m, d, b = M[ok], Dm[ok], B[ok]
    r = {"reference": ref, "N cells": int(ok.sum()),
         "Bilinear MAE": np.abs(b - y).mean(),
         "Deterministic MAE (per window, mean of 4)": np.mean([np.abs(d[:, w] - y).mean() for w in range(NU)]),
         "EngressNet CRPS K=20 (per window, mean of 4)": np.mean([crps_fair(m[:, w * K:(w + 1) * K], y) for w in range(NU)]),
         "Deterministic MAE (4-window mean)": np.abs(d.mean(1) - y).mean(),
         "EngressNet CRPS (80 members pooled)": crps_fair(m, y),
         "EngressNet RMSE of mean": np.sqrt(((m.mean(1) - y) ** 2).mean()), "EngressNet bias": (m.mean(1) - y).mean(),
         "Deterministic RMSE": np.sqrt(((d.mean(1) - y) ** 2).mean()), "Bilinear RMSE": np.sqrt(((b - y) ** 2).mean()),
         "units": NU}
    e, dd, bb = r["EngressNet CRPS K=20 (per window, mean of 4)"], r["Deterministic MAE (per window, mean of 4)"], r["Bilinear MAE"]
    r["EngressNet reduction vs deterministic (%)"] = 100 * (1 - e / dd)
    r["EngressNet reduction vs bilinear (%)"] = 100 * (1 - e / bb)
    rows.append(r)
df = pd.DataFrame(rows).set_index("reference")
df.to_csv(OUT)
pd.set_option("display.width", 250)
print(df.T.round(4).to_string())
