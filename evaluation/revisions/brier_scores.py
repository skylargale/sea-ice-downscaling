#!/usr/bin/env python3
"""
Threshold probabilities P(SIT >= t), t = 0.15 m and 0.5 m (2026-11-18): Brier score (BS) and Brier skill score relative to bilinear
(BSS = 1 - BS / BS_bilinear), perfect model (2021 test year), both datasets, final model, plus the ICESat-2 testbed.
Forecast probability: EngressNet = fraction of its 20 members >= t; seed ensemble = fraction of its 10 det-MSE U-Nets (seeds 0-9);
deterministic U-Net and bilinear = 0/1. Observed event: truth >= t (perfect model) / ICESat-2 pass-cell mean >= t (testbed).
Regions (perfect model): all ocean cells; coastal band (ocean cells within 5 px of land, fe.coastal_band_mask as the coastal metrics);
the five community cells of Table 4 (the run's nearest valid ocean cell, from the saved point series).
Testbed: the Table 5 pass-cells (QC as is2_crps_basis.py, 161,376 cells, 15 Oct 2018 - 30 Apr 2022), FOSI-trained models on PIOMAS.
Also writes reliability-diagram data: one row per distinct forecast probability (member count k of K, so no binning), with
the observed frequency and the number of cells (for the sharpness histogram).
Outputs: results/revisions_2026-11/brier_scores_<FINAL_SET>.csv, reliability_<FINAL_SET>.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "training"))
sys.path.insert(0, os.path.join(V6, "evaluation"))
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
sys.path.insert(0, os.path.join(V6, "observations", "processing"))
import functions_engressnet as fe  # noqa: E402
from final_set import FS, NAME as FINAL_SET  # noqa: E402

OUT = os.path.join(V6, "results", "revisions_2026-11")
THR = [0.15, 0.5]
POINTS = ["Point Hope", "Kivalina", "Kotzebue", "Shishmaref", "Nome"]
METHODS = ["Bilinear", "Deterministic U-Net", "EngressNet", "Seed/window ensemble (det.)"]


def run_dir(batch):
    m = [d for d in glob.glob(os.path.join(V6, "results", batch, "*_2021_*")) if os.path.isdir(d)]
    assert len(m) == 1, (batch, m)
    return m[0]


def probs_full(ds):
    """Full-field forecast probabilities and observed events for both thresholds, loading each file once:
    -> ({t: ({method: p (N, H, W) float32}, o (N, H, W) float32)}, {method: ensemble size})."""
    f = np.load(os.path.join(run_dir(FS[ds]["eng"]), "eval_data", "fields.npz"))
    truth, base = f["Y_test_phys"][:, 0], f["Y_base_phys"][:, 0]
    preds = f["preds_all_phys"]
    K_eng = preds.shape[1]
    out = {t: ({"Bilinear": (base >= t).astype(np.float32)}, (truth >= t).astype(np.float32)) for t in THR}
    for t in THR:
        pe = np.zeros(truth.shape, np.float32)
        for k in range(preds.shape[1]):
            pe += preds[:, k, 0] >= t
        out[t][0]["EngressNet"] = pe / preds.shape[1]
    del preds, f
    seeds = [run_dir(FS[ds]["det"])] + sorted(glob.glob(os.path.join(V6, "results", f"FINAL2_{ds}_detmse_seedens", "*_2021_*")))
    acc = {t: np.zeros(truth.shape, np.float32) for t in THR}
    for i, d in enumerate(seeds):
        x = np.load(os.path.join(d, "eval_data", "fields.npz"))["Y_pred_det_phys"][:, 0]
        for t in THR:
            acc[t] += x >= t
            if i == 0:
                out[t][0]["Deterministic U-Net"] = (x >= t).astype(np.float32)
    for t in THR:
        out[t][0]["Seed/window ensemble (det.)"] = acc[t] / len(seeds)
    return out, {"Bilinear": 1, "Deterministic U-Net": 1, "EngressNet": K_eng, "Seed/window ensemble (det.)": len(seeds)}


def score_rows(P, o, base):
    bs = {m: float(np.mean((P[m] - o) ** 2)) for m in METHODS}
    return [{**base, "method": m, "n": int(o.size), "event frequency": float(o.mean()), "BS": bs[m],
             "BSS vs bilinear": 1 - bs[m] / bs["Bilinear"] if bs["Bilinear"] > 0 else np.nan} for m in METHODS]


def reliability_rows(P, o, base, K):
    """One row per member count k (forecast probability k/K) that occurs; K = {method: ensemble size} (1 for the 0/1 forecasts)."""
    rows = []
    for m in METHODS:
        k = np.rint(P[m] * K[m]).astype(int)
        cnt = np.bincount(k.ravel(), minlength=K[m] + 1)
        hits = np.bincount(k.ravel(), weights=o.ravel(), minlength=K[m] + 1)
        for j in np.nonzero(cnt)[0]:
            rows.append({**base, "method": m, "K": K[m], "members": int(j), "forecast probability": j / K[m],
                         "observed frequency": hits[j] / cnt[j], "count": int(cnt[j])})
    return rows


def main():
    rows, rel = [], []
    for ds, label in (("MESA", "MESACLIP"), ("FOSI", "FOSI")):
        f = np.load(os.path.join(run_dir(FS[ds]["eng"]), "eval_data", "fields.npz"))
        mask = f["mask_test"][:, 0]
        ocean = mask <= 0.5
        coastal = fe.coastal_band_mask(torch.from_numpy(mask), coastal_width=5).numpy().astype(bool) & ocean
        pts = pd.read_csv(f"{V6}/evaluation/saved_figs/{FS[ds]['eng']}/_4window_aggregate/02_candidate_point_timeseries_data_4window.csv")
        regions = {"all ocean": ocean, "coastal band": coastal}
        for p in POINTS:
            iy, ix = pts.loc[pts.point == p, ["grid_iy", "grid_ix"]].iloc[0].astype(int)
            s = np.zeros_like(ocean); s[:, iy, ix] = True
            regions[p] = s & ocean
        del f
        full, K = probs_full(ds)
        for t in THR:
            for reg, sel in regions.items():
                P = {m: v[sel].astype(np.float64) for m, v in full[t][0].items()}
                o = full[t][1][sel].astype(np.float64)
                base = {"setting": "perfect model 2021", "dataset": label, "threshold (m)": t, "region": reg}
                rows += score_rows(P, o, base)
                if reg in ("all ocean", "coastal band"):
                    rel += reliability_rows(P, o, base, K)
                print(label, t, reg, {r["method"]: round(r["BSS vs bilinear"], 3) for r in rows[-4:]}, flush=True)
    # ICESat-2 testbed (FOSI-trained models on PIOMAS), Table 5 cells
    import calibration_options_compare as c
    pcd = c.PCD
    qc = ((pcd.extent_km >= 2.0) & (pcd.n_seg >= 20) & (pcd.unc_med <= 1.0)).values
    if "in_domain" in pcd:
        qc = qc & pcd.in_domain.values
    E = np.load(f"{c.EV2}/{FS['cells_eng']}_cells.npz"); D0 = np.load(f"{c.EV2}/{FS['cells_det']}_cells.npz")
    S9 = np.load(f"{c.EV2}/final2_seedens_piomas_cells.npz"); K9 = int(S9["K"])
    M = E["members"].astype(float)
    seed = np.column_stack([D0["members"][:, 0]] + [S9["members"][:, u * K9] for u in range(len(S9["windows"]))]).astype(float)
    B = E["bilinear"].astype(float)
    ok = qc & np.isfinite(M).all(1) & np.isfinite(seed).all(1) & np.isfinite(B) & np.isfinite(pcd.h_mean.values) & np.isfinite(pcd.h_median.values)
    y = pcd.h_mean.values[ok].astype(float)
    for t in THR:
        P = {"Bilinear": (B[ok] >= t).astype(float), "Deterministic U-Net": (seed[ok][:, 0] >= t).astype(float),
             "EngressNet": (M[ok] >= t).mean(1), "Seed/window ensemble (det.)": (seed[ok] >= t).mean(1)}
        o = (y >= t).astype(float)
        base = {"setting": "ICESat-2 testbed", "dataset": "FOSI on PIOMAS", "threshold (m)": t, "region": "Table 5 pass-cells"}
        rows += score_rows(P, o, base)
        rel += reliability_rows(P, o, base, {"Bilinear": 1, "Deterministic U-Net": 1, "EngressNet": M.shape[1], "Seed/window ensemble (det.)": seed.shape[1]})
        print("ICESat-2", t, {r["method"]: round(r["BSS vs bilinear"], 3) for r in rows[-4:]}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/brier_scores_{FINAL_SET}.csv", index=False)
    pd.DataFrame(rel).to_csv(f"{OUT}/reliability_{FINAL_SET}.csv", index=False)
    print("wrote", f"{OUT}/brier_scores_{FINAL_SET}.csv", f"{OUT}/reliability_{FINAL_SET}.csv")


if __name__ == "__main__":
    main()
