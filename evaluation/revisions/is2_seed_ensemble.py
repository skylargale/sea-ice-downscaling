"""
2026-11-17: EngressNet vs the seed-ensemble baseline on the ICESat-2 testbed (FOSI-trained models on PIOMAS input, winters
15 Oct 2018 - 30 Apr 2022). Seed ensemble = the 10 det-MSE U-Nets trained on the same 1,825 days (seed 0:
final2_detmse_piomas, seeds 1-9: final2_seedens_piomas, observations/inference/submit_seedens_obs_inference.py).

Same QC and definitions as Table 5 (is2_crps_basis.py / is2_spread_error.py): pass-cells with extent >= 2 km, >= 20 segments,
median segment uncertainty <= 1 m, every estimate finite (now also the 9 extra models), references = pass-cell means (Table 5) and
medians. Scores: CRPS (K^2 estimator as the paper, and the fair estimator, unbiased for ensemble size), RMSE / bias / r of the
ensemble mean, spread/error with and without the ICESat-2 observation-noise term, fraction of pass-cells outside the ensemble
(ideal 2/(K+1)). Also each single det model (mean of the 10 single-model CRPS = MAE).
Writes results/revisions_2026-11/is2_seed_ensemble.csv.
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
D0 = np.load(f"{c.EV2}/{SETS['final2']['cells_det']}_cells.npz")
S9 = np.load(f"{c.EV2}/final2_seedens_piomas_cells.npz")
M = E["members"].astype(float)                                   # EngressNet, 20 members
K9 = int(S9["K"])
seed = np.column_stack([D0["members"][:, 0]] + [S9["members"][:, u * K9] for u in range(len(S9["windows"]))]).astype(float)
B = E["bilinear"].astype(float)
assert seed.shape[1] == 10, seed.shape
ok = qc & np.isfinite(M).all(1) & np.isfinite(seed).all(1) & np.isfinite(B) & np.isfinite(pcd.h_mean.values) & np.isfinite(pcd.h_median.values)
engb = np.load(f"{c.EV2}/{SETS['final2']['cells_eng']}_bins.npz")["members"].astype(float)
W = c.contiguous_windows(c.qcB & np.isfinite(engb).all(1))
print(f"pass-cells: {ok.sum()} (Table 5: 161,376) | seed-ensemble units: {list(S9['windows'])}")


def crps(X, y, fair):
    K = X.shape[1]
    s = np.sort(X, 1)
    w = 2 * np.arange(1, K + 1) - K - 1
    return float((np.abs(X - y[:, None]).mean(1) - (s * w).sum(1) / (K * (K - 1) if fair else K * K)).mean())


rows = []
for ref, col in (("pass-cell mean", "h_mean"), ("pass-cell median", "h_median")):
    y = pcd[col].values[ok].astype(float)
    sig = c.noise_fn(c.BIN[col].values.astype(float), W)
    for name, X in (("EngressNet (1 model, 20 members)", M[ok]), ("Seed ensemble (10 det-MSE U-Nets)", seed[ok])):
        K = X.shape[1]
        m = X.mean(1)
        rmse = np.sqrt(np.mean((m - y) ** 2))
        s2 = X.var(1, ddof=1).mean()
        below, ties = (X < y[:, None]).sum(1), (X == y[:, None]).sum(1)
        r = below + np.random.default_rng(0).integers(0, ties + 1)
        rows.append({"reference": ref, "ensemble": name, "N cells": int(ok.sum()), "members": K,
                     "CRPS (K^2)": crps(X, y, False), "CRPS (fair)": crps(X, y, True), "RMSE of mean": rmse,
                     "bias of mean": float(np.mean(m - y)), "r of mean": float(np.corrcoef(m, y)[0, 1]),
                     "spread/error": np.sqrt((K + 1) / K * s2) / rmse,
                     "spread/error + obs noise": np.sqrt((K + 1) / K * s2 + np.mean(sig(m) ** 2)) / rmse,
                     "outside ensemble": float(((r == 0) | (r == K)).mean()), "ideal outside": 2 / (K + 1)})
    single = [np.abs(seed[ok][:, i] - y).mean() for i in range(seed.shape[1])]
    rows.append({"reference": ref, "ensemble": "single det-MSE U-Net (mean of the 10; CRPS = MAE)", "N cells": int(ok.sum()), "members": 1,
                 "CRPS (K^2)": float(np.mean(single)), "CRPS (fair)": float(np.mean(single)),
                 "RMSE of mean": float(np.mean([np.sqrt(np.mean((seed[ok][:, i] - y) ** 2)) for i in range(seed.shape[1])]))})
df = pd.DataFrame(rows)
df.to_csv(f"{V6}/results/revisions_2026-11/is2_seed_ensemble.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(df.round(4).to_string(index=False))
