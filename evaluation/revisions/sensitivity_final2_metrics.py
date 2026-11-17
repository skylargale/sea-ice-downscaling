"""
Final round (2026-11): scores for the Appendix A sensitivity tests on the final design (training/submit_sensitivity_final2.sh).

Headline convention, as Tables 2/3 (evaluation/recompute_ocean_metrics.py, recompute_spread_skill.py): EngressNet ensemble
mean, ocean cells, 2021. Columns: MAE, RMSE, Bias, Grad MAE, Pattern Corr, SSIM, IIEE (1e-4 m and 0.15 m), Coastal MAE,
Coastal RMSE (5-cell band), CRPS (all members), Energy score (whole ocean field, beta = 1, / sqrt(n ocean cells)), Spread-skill
(sqrt((K+1)/K mean var) / RMSE). Formulas copied from those scripts so the appendix matches the main tables.

Runs: results/SENS2_<DS>_<config>/<run> plus results/FINAL2_<DS>_random1825 (final configuration, seed 0).
Like-for-like adjustments:
  season     scored on Mar-Jul 2021; the final configuration is also scored on Mar-Jul 2021 (rows "final (Mar-Jul)")
  bigdomain  cropped to the final configuration's domain before scoring
  land05     scored on its own ocean cells (threshold 0.5 drops ~1.2% of the final's ocean cells, all coastal); the final
             configuration is also scored on those cells (rows "final (land05 cells)")
  construth  scored against its own (conservative) truth
Writes results/revisions_2026-11/sensitivity_final2/<DS>_runs.csv (one row per run and scoring) and <DS>_summary.csv (mean and
min-max over seeds). Usage: python evaluation/revisions/sensitivity_final2_metrics.py [MESA FOSI]
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "training")); sys.path.insert(0, os.path.join(V6, "evaluation"))
import functions_engressnet as fe  # noqa: E402

OUT = f"{V6}/results/revisions_2026-11/sensitivity_final2"
CHUNK = 64


def load(run):
    f = np.load(os.path.join(run, "eval_data", "fields.npz"))
    import pickle
    geo = pickle.load(open(os.path.join(run, "eval_data", "tile_geometry.pkl"), "rb"))[0]
    times = pd.to_datetime(pd.read_csv(os.path.join(run, "eval_data", "sample_times.csv"))["time"]).values
    return {"truth": f["Y_test_phys"][:, 0], "ens": f["preds_all_phys"][:, :, 0], "mean": f["Y_pred_phys"][:, 0],
            "land": f["mask_test"][:, 0], "lat": np.asarray(geo["target_lat"]), "lon": np.asarray(geo["target_lon"]),
            "times": times}


def subset(d, rows=None, lat=None, lon=None):
    """rows: sample selection; lat/lon: crop to these target coordinates (nearest match)."""
    o = dict(d)
    if rows is not None:
        for k in ("truth", "ens", "mean", "land", "times"):
            o[k] = o[k][rows]
    if lat is not None:
        iy = np.array([np.argmin(np.abs(d["lat"] - v)) for v in lat])
        ix = np.array([np.argmin(np.abs((d["lon"] - v + 180) % 360 - 180)) for v in lon])
        assert np.allclose(d["lat"][iy], lat, atol=1e-3) and np.allclose(((d["lon"][ix] - lon + 180) % 360) - 180, 0, atol=1e-3)
        for k in ("truth", "mean", "land"):
            o[k] = o[k][:, iy][:, :, ix]
        o["ens"] = o["ens"][:, :, iy][:, :, :, ix]
        o["lat"], o["lon"] = d["lat"][iy], d["lon"][ix]
    return o


def score(d, ocean=None):
    truth, mean, ens = d["truth"], d["mean"], d["ens"]
    ocean = (d["land"] <= 0.5) if ocean is None else ocean
    err = mean - truth
    e_o = err[ocean].astype(np.float64)
    r = [np.corrcoef(mean[i][ocean[i]], truth[i][ocean[i]])[0, 1] for i in range(len(truth))
         if mean[i][ocean[i]].std() > 0 and truth[i][ocean[i]].std() > 0]
    gx_ok, gy_ok = ocean[:, :, 1:] & ocean[:, :, :-1], ocean[:, 1:, :] & ocean[:, :-1, :]
    gx = (np.diff(mean, axis=2) - np.diff(truth, axis=2))[gx_ok]
    gy = (np.diff(mean, axis=1) - np.diff(truth, axis=1))[gy_ok]
    pt, tt = torch.from_numpy(mean[:, None]), torch.from_numpy(truth[:, None])
    ssim_v = [float(fe.ssim(pt[i:i + 128], tt[i:i + 128])) for i in range(0, len(pt), 128)]
    ssim_w = [min(128, len(pt) - i) for i in range(0, len(pt), 128)]
    coastal = fe.coastal_band_mask(torch.from_numpy(1.0 - ocean.astype(np.float32)), coastal_width=5).numpy() & ocean
    e_c = err[coastal].astype(np.float64)
    lat_t = torch.from_numpy(d["lat"].astype(np.float32))
    oc_t = torch.from_numpy(ocean)
    iiee = {thr: float(fe.ice_edge_error(pt, tt, threshold=thr, mask_bool=oc_t, lat=lat_t)) for thr in (1e-4, 0.15)}
    # CRPS (K^2 estimator) and spread-skill, as recompute_spread_skill.py; energy score per sample, as spatial_prob_metrics.py
    K = ens.shape[1]
    w = 2 * np.arange(1, K + 1) - K - 1
    n = sv = sse = crps = 0.0
    es = []
    for i in range(0, len(truth), CHUNK):
        oc = ocean[i:i + CHUNK]
        E = np.moveaxis(ens[i:i + CHUNK], 1, -1)[oc].astype(np.float64)
        y = truth[i:i + CHUNK][oc].astype(np.float64)
        mu = E.mean(1)
        n += y.size; sv += E.var(1, ddof=1).sum(); sse += ((mu - y) ** 2).sum()
        crps += (np.abs(E - y[:, None]).mean(1) - (np.sort(E, 1) * w).sum(1) / K ** 2).sum()
        for j in range(oc.shape[0]):
            X = torch.from_numpy(ens[i + j][:, oc[j]].astype(np.float64))[None]
            yy = torch.from_numpy(truth[i + j][oc[j]].astype(np.float64))[None]
            t1 = torch.linalg.vector_norm(X - yy[:, None], dim=-1).mean(1)
            t2 = torch.cdist(X, X).sum((1, 2)) / (K * (K - 1))
            es.append(float((t1 - 0.5 * t2) / np.sqrt(oc[j].sum())))
    rmse_m = np.sqrt(sse / n)
    return {"MAE": np.abs(e_o).mean(), "RMSE": np.sqrt((e_o ** 2).mean()), "Bias": e_o.mean(),
            "Grad MAE": np.abs(np.concatenate([gx, gy])).mean(), "Pattern Corr": float(np.mean(r)),
            "SSIM": float(np.average(ssim_v, weights=ssim_w)), "IIEE (1e-4 m)": iiee[1e-4], "IIEE (0.15 m)": iiee[0.15],
            "Coastal MAE": np.abs(e_c).mean(), "Coastal RMSE": np.sqrt((e_c ** 2).mean()), "CRPS": crps / n,
            "Energy score": float(np.mean(es)), "Spread-skill": np.sqrt((K + 1) / K * sv / n) / rmse_m,
            "K": K, "N ocean cells": int(n)}


def main():
    os.makedirs(OUT, exist_ok=True)
    for ds in (sys.argv[1:] or ["MESA", "FOSI"]):
        final0 = glob.glob(f"{V6}/results/FINAL2_{ds}_random1825/*/eval_data/fields.npz")
        runs = [("final", os.path.dirname(os.path.dirname(final0[0])), 0)] if final0 else []
        for b in sorted(glob.glob(f"{V6}/results/SENS2_{ds}_*")):
            cfg_name = os.path.basename(b)[len(f"SENS2_{ds}_"):]
            for rd in sorted(glob.glob(f"{b}/*/")):  # runs that crashed have no fields.npz: say so instead of skipping silently
                if not os.path.exists(os.path.join(rd, "eval_data", "fields.npz")):
                    print(f"WARNING {ds} {cfg_name}: no eval_data/fields.npz (failed or unfinished run), skipped: {rd}")
            for r in sorted(glob.glob(f"{b}/*/eval_data/fields.npz")):
                rd = os.path.dirname(os.path.dirname(r))
                seed = json.load(open(os.path.join(rd, "run_config.json"))).get("seed", -1)
                runs.append((cfg_name, rd, seed))
        ref = load(runs[0][1])
        l05 = sorted(glob.glob(f"{V6}/results/SENS2_{ds}_land05/*/eval_data/fields.npz"))
        ocean05 = (np.load(l05[0])["mask_test"][:, 0] <= 0.5) if l05 else None  # land05's ocean cells (a subset of the final's)
        rows = []
        for cfg_name, rd, seed in runs:
            d = load(rd)
            jobs = []
            if cfg_name == "bigdomain":
                jobs.append(("bigdomain", subset(d, lat=ref["lat"], lon=ref["lon"]), None))
            elif cfg_name == "land05":
                jobs.append(("land05", d, None))
            else:
                jobs.append((cfg_name, d, None))
            if cfg_name == "final":
                mj = pd.DatetimeIndex(d["times"]).month
                jobs.append(("final (Mar-Jul)", subset(d, rows=np.where((mj >= 3) & (mj <= 7))[0]), None))
                if ocean05 is not None and ocean05.shape == d["land"].shape:
                    jobs.append(("final (land05 cells)", d, ocean05 & (d["land"] <= 0.5)))
            for label, dd, oc in jobs:
                s = score(dd, oc)
                rows.append({"dataset": ds, "config": label, "seed": seed, "run": os.path.basename(rd), **s})
                print(ds, label, seed, f"RMSE {s['RMSE']:.4f} CRPS {s['CRPS']:.4f} SSR {s['Spread-skill']:.3f}", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(f"{OUT}/{ds}_runs.csv", index=False)
        num = [c for c in df.columns if c not in ("dataset", "config", "seed", "run")]
        g = df.groupby("config", sort=False)[num]
        summ = g.mean()
        summ.insert(0, "seeds", df.groupby("config", sort=False)["seed"].nunique())
        for c in ("RMSE", "CRPS", "Spread-skill", "IIEE (1e-4 m)", "IIEE (0.15 m)", "Energy score"):
            gc = df.groupby("config", sort=False)[c]; summ[f"{c} min"], summ[f"{c} max"] = gc.min(), gc.max()
        summ.to_csv(f"{OUT}/{ds}_summary.csv")
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
        print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
