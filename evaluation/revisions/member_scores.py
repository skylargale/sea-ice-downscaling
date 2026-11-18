#!/usr/bin/env python3
"""
Per-member scores for the table charts (2026-11-17): every EngressNet member (K = 20) and every model of the seed-ensemble
baseline (10 det-MSE U-Nets, seeds 0-9 on the same 1,825 days) scored as a single forecast, final model, 2021 test year,
ocean cells, both datasets. Bars in the charts stay the ensemble-mean (or single-model) values; these are the dots.

Definitions are the paper's, imported from evaluation/revisions/spatial_prob_metrics.py (structure_metrics = the
recompute_ocean_metrics.py ocean-cell definitions; IIEE 1e-4 m, cos-lat), so the ensemble-mean row reproduces Tables 2/3 and the
spatial-metrics table exactly (checked and printed):
  RMSE, MAE, Pattern Corr, SSIM, Grad MAE, IIEE      per forecast, ocean cells
  FSS thr=0.5 m n=1                                   a single forecast is binary (exceedance 0/1); the ensemble row uses the
                                                      ensemble exceedance fraction, as the table
  ACC                                                 uncentred anomaly correlation vs the training-years day-of-year truth
                                                      climatology (31-day circular smoothing), as the table
Ensemble-only scores (CRPS, spread-skill, energy, variogram) have no per-member value and are not computed.

Writes results/revisions_2026-11/member_scores_<FINAL_SET>.csv (dataset, method, member, metrics) and prints member ranges.
Usage (project root): python evaluation/revisions/member_scores.py [MESA FOSI]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import uniform_filter

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
import spatial_prob_metrics as spm  # noqa: E402  (also puts training/ and evaluation/ on sys.path)
from final_set import FS, NAME as FINAL_SET  # noqa: E402

fe = spm.fe
OUT = os.path.join(V6, "results", "revisions_2026-11", f"member_scores_{FINAL_SET}.csv")
FSS_THR, FSS_N = 0.5, 1


def fss(fcst_frac, truth, ocean, thr=FSS_THR, n=FSS_N):
    """spatial_prob_metrics.score's FSS for one threshold/scale. fcst_frac (N, H, W) = exceedance fraction (0/1 for one forecast)."""
    ocf = ocean.astype(np.float64)
    P = fcst_frac.astype(np.float64) * ocf
    O = (truth > thr).astype(np.float64) * ocf
    den_oc = uniform_filter(ocf, size=n, mode="constant")
    Pf = uniform_filter(P, size=(1, n, n), mode="constant") / np.maximum(den_oc, 1e-12)
    Of = uniform_filter(O, size=(1, n, n), mode="constant") / np.maximum(den_oc, 1e-12)
    num = ((Pf - Of) ** 2)[:, ocean].sum()
    den = (Pf ** 2)[:, ocean].sum() + (Of ** 2)[:, ocean].sum()
    return 1 - num / den if den > 0 else np.nan


def acc(pred, truth, clim, ocean):
    fa, oa = (pred - clim)[:, ocean], (truth - clim)[:, ocean]
    den = np.sqrt((fa ** 2).sum(1) * (oa ** 2).sum(1))
    ok = den > 0
    return float(((fa * oa).sum(1)[ok] / den[ok]).mean())


def climatology(ds, win, tlat, tlon, shape):
    """Training-years DOY truth climatology, exactly as spatial_prob_metrics.main."""
    yrs = fe.parse_years(win)
    Yds = xr.open_dataset(spm.Y_FILES[ds])
    Yv = Yds[list(Yds.data_vars)[0]]
    ylat, ylon = Yds["lat"].values, Yds["lon"].values % 360
    li = [int(np.argmin(np.abs(ylat - v))) for v in tlat]
    lj = [int(np.argmin(np.abs(ylon - (v % 360)))) for v in tlon]
    tyears = np.array([t.year for t in Yds["time"].values])
    sel_t = np.flatnonzero(np.isin(tyears, yrs))
    clim_sum = np.zeros((365,) + shape); clim_n = np.zeros(365)
    doy_tr = spm.noleap_doy(Yds["time"].values[sel_t])
    for m in range(Yv.sizes["ensemble"]):
        arr = np.clip(Yv.isel(ensemble=m, time=sel_t, channel=0).values[:, li][:, :, lj], None, 6.0)
        np.add.at(clim_sum, doy_tr, arr); np.add.at(clim_n, doy_tr, 1)
    clim = clim_sum / np.maximum(clim_n, 1)[:, None, None]
    return np.mean([np.roll(clim, s, axis=0) for s in range(-15, 16)], axis=0).astype(np.float32)


def single(pred, truth, ocean, hlat, clim_t):
    """All per-forecast scores of one (N, H, W) forecast."""
    p = torch.from_numpy(np.ascontiguousarray(pred[:, None]))
    t = torch.from_numpy(truth[:, None])
    s = spm.structure_metrics(p, t, torch.from_numpy(ocean), hlat)
    s["MAE"] = float(np.abs(pred - truth)[:, ocean].astype(np.float64).mean())
    s[f"FSS thr={FSS_THR:g} m n={FSS_N}"] = fss(pred > FSS_THR, truth, ocean)
    s["ACC"] = acc(pred, truth, clim_t, ocean)
    return s


def main():
    import pickle
    rows = []
    for ds in (sys.argv[1:] or ["MESA", "FOSI"]):
        win = FS[ds]["windows"][0]
        eng = spm.run_dir(FS[ds]["eng"], win)
        f = np.load(os.path.join(eng, "eval_data", "fields.npz"))
        truth = f["Y_test_phys"][:, 0].astype(np.float32)
        ocean = f["mask_test"][0, 0] <= 0.5
        hlat = torch.from_numpy(f["hlat"][:truth.shape[1]] if f["hlat"].shape[0] != truth.shape[1] else f["hlat"])
        geo = pickle.load(open(os.path.join(eng, "eval_data", "tile_geometry.pkl"), "rb"))[0]
        times = pd.to_datetime(pd.read_csv(os.path.join(eng, "eval_data", "sample_times.csv"))["time"])
        clim_t = climatology(ds, win, np.asarray(geo["target_lat"]), np.asarray(geo["target_lon"]), ocean.shape)[spm.noleap_doy(times)]
        P = f["preds_all_phys"][:, :, 0].astype(np.float32)
        K = P.shape[1]
        print(f"{ds} {win}: N={truth.shape[0]}, K={K}, ocean {ocean.sum()}", flush=True)
        # ensemble mean (check against the tables) and members
        mu = P.mean(1)
        s = single(mu, truth, ocean, hlat, clim_t)
        s[f"FSS thr={FSS_THR:g} m n={FSS_N}"] = fss((P > FSS_THR).mean(1), truth, ocean)   # ensemble fraction, as the table
        rows.append({"dataset": ds, "method": "EngressNet", "member": "ensemble mean", **s})
        for k in range(K):
            rows.append({"dataset": ds, "method": "EngressNet", "member": k, **single(P[:, k], truth, ocean, hlat, clim_t)})
            print(f"  EngressNet member {k}: RMSE {rows[-1]['RMSE']:.4f}", flush=True)
        del P
        # seed ensemble: the det-MSE model of the final configuration (seed 0) + seeds 1-9
        seed_runs = [spm.run_dir(FS[ds]["det"], win)] + sorted(glob.glob(os.path.join(V6, "results", f"FINAL2_{ds}_detmse_seedens", "*_2021_*")))
        S = []
        for i, d in enumerate(seed_runs):
            fx = np.load(os.path.join(d, "eval_data", "fields.npz"))
            assert np.allclose(fx["Y_test_phys"][:, 0], truth, atol=1e-5), f"{d}: test set differs"
            x = fx["Y_pred_det_phys"][:, 0].astype(np.float32)
            S.append(x)
            rows.append({"dataset": ds, "method": "Seed ensemble", "member": i, "run": os.path.basename(d),
                         **single(x, truth, ocean, hlat, clim_t)})
            print(f"  seed model {i} ({os.path.basename(d)}): RMSE {rows[-1]['RMSE']:.4f}", flush=True)
        S = np.stack(S, 1)
        s = single(S.mean(1), truth, ocean, hlat, clim_t)
        s[f"FSS thr={FSS_THR:g} m n={FSS_N}"] = fss((S > FSS_THR).mean(1), truth, ocean)
        rows.append({"dataset": ds, "method": "Seed ensemble", "member": "ensemble mean", **s})
        del S
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    met = ["RMSE", "MAE", "Pattern Corr", "SSIM", "Grad MAE", "IIEE", f"FSS thr={FSS_THR:g} m n={FSS_N}", "ACC"]
    for (ds, m), g in df.groupby(["dataset", "method"], sort=False):
        mem = g[g["member"] != "ensemble mean"]
        print(f"\n{ds} {m}: {len(mem)} members")
        print(pd.DataFrame({"ensemble mean": g[g["member"] == "ensemble mean"][met].iloc[0], "member min": mem[met].min(),
                            "member max": mem[met].max(), "member mean": mem[met].mean()}).round(4).to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
