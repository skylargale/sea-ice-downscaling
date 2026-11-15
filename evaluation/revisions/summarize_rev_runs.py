"""
Revisions 2026-11: score every revision run (H, F, G, E) on the same basis as the paper (ocean cells, coarse-zero rule, original
truth after restore_raw_truth.py), and write one table per experiment with the mean and min-max over seeds.

Per run (from eval_data/fields.npz):
  recompute_ocean_metrics.window_metrics: RMSE, MAE, bias, pattern r, gradient MAE, SSIM, IIEE (1e-4 m), coastal RMSE of the
    ensemble mean (EngressNet-type runs) or the deterministic prediction (det-MSE runs)
  CRPS (K^2 estimator; = MAE for one member), corrected spread-skill overall and per thickness type (ensemble-mean bins),
  rank-histogram fraction outside the ensemble (rdeb.detection_limited ties, ideal 2/(K+1)), IIEE at 0.15 m
Experiments:
  H  MESACLIP 2015-2020: concat (paper seed 0 = MESA_final_ema + REV_H_MESA_concat seeds 1-2), mult, mult+bottleneck, add_feat
  F  random 2190 days from 2000-2020 vs consecutive 2015-2020, MESACLIP (contiguous = H concat) and FOSI
     (contiguous = FOSI_final_ema seed 0 + REV_F_FOSI_contiguous seeds 1-2)
  G  2000-2010-trained MESACLIP model tested on 2015..2021, with bilinear on the same years for reference
  E  per-run scores of E1 det-MSE seeds and E2 MC-dropout (the ensemble scores of the seed/window ensemble come from
     spatial_prob_metrics.py --seedwin)
Writes results/revisions_2026-11/rev_{H,F,G,E}_{per_run,summary}.csv and prints the summaries.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "training"))
sys.path.insert(0, os.path.join(V6, "evaluation"))
import recompute_ocean_metrics as rom  # noqa: E402
import run_daily_eval_batch as rdeb  # noqa: E402
import functions_engressnet as fe  # noqa: E402

OUT = f"{V6}/results/revisions_2026-11"
R = f"{V6}/results"
EDGES = [0.15, 0.5, 1.0]
REG = ["open", "thin", "moderate", "thick"]


def prob_stats(fp):
    f = np.load(fp)
    P, T, M = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"]
    N, K = P.shape[:2]
    w = 2 * np.arange(1, K + 1) - K - 1
    acc = {k: np.zeros(5) for k in ("n", "var", "sse")}
    crps = out = 0.0
    rng = np.random.default_rng(0)
    for i in range(0, N, 64):
        oc = M[i:i + 64, 0] <= 0.5
        e = np.moveaxis(P[i:i + 64, :, 0], 1, -1)[oc].astype(np.float64)
        y = T[i:i + 64, 0][oc].astype(np.float64)
        mu = e.mean(1); se = (mu - y) ** 2
        if K > 1:
            v = e.var(1, ddof=1)
            crps += (np.abs(e - y[:, None]).mean(1) - (np.sort(e, 1) * w).sum(1) / K ** 2).sum()
            er, tr = rdeb.detection_limited(e, y)
            below, ties = (er < tr[:, None]).sum(1), (er == tr[:, None]).sum(1)
            rk = below + rng.integers(0, ties + 1)
            out += ((rk == 0) | (rk == K)).sum()
        else:
            v = np.zeros_like(mu); crps += np.abs(mu - y).sum()
        b = np.digitize(mu, EDGES)
        for r in range(5):
            s = slice(None) if r == 4 else (b == r)
            acc["n"][r] += se[s].size; acc["var"][r] += v[s].sum(); acc["sse"][r] += se[s].sum()
    n = acc["n"]; rmse = np.sqrt(acc["sse"] / n)
    d = {"K": K, "CRPS (1 member: MAE)": crps / n[4]}
    if K > 1:
        ssr = np.sqrt((K + 1) / K * acc["var"] / n) / rmse
        d["spread-skill"] = ssr[4]
        d.update({f"SSR {r}": ssr[i] for i, r in enumerate(REG)})
        d["rank outside frac"] = out / n[4]
    # IIEE at 0.15 m for the ensemble mean / point value, same definition as Tables 2/3
    H = T.shape[-2]
    hlat = f["hlat"][:H] if f["hlat"].shape[0] != H else f["hlat"]
    pm = torch.from_numpy(f["Y_pred_phys"]); tt = torch.from_numpy(T); ocean = torch.from_numpy(M[:, 0] <= 0.5)
    d["IIEE 0.15 m"] = float(fe.ice_edge_error(pm, tt, threshold=0.15, mask_bool=ocean, lat=torch.from_numpy(hlat)))
    return d


def score(run_dir, det=False):
    fp = os.path.join(run_dir, "eval_data", "fields.npz")
    m = rom.window_metrics(fp)["Deterministic (zero-noise pass)" if det else "EngressNet ensemble mean"]
    m = {f"{k}": v for k, v in m.items()}
    m["IIEE 1e-4 m"] = m.pop("IIEE")
    return {**m, **prob_stats(fp)}


def runs(batch, window=None):
    ds = sorted(d for d in glob.glob(f"{R}/{batch}/*/") if os.path.exists(os.path.join(d, "eval_data", "fields.npz")))
    return [d for d in ds if window is None or f"_{window}_" in d]


def seed_of(d):
    m = re.search(r"_s(\d)_", os.path.basename(os.path.normpath(d)))
    return int(m.group(1)) if m else 0


def table(rows, keys, name):
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/rev_{name}_per_run.csv", index=False)
    num = [c for c in df.columns if c not in keys + ["run", "seed"] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys, sort=False)[num]
    summ = g.mean().round(4)
    rng_ = (g.min().round(4).astype(str) + "-" + g.max().round(4).astype(str)).add_suffix(" [min-max]")
    out = pd.concat([summ, rng_], axis=1)
    out.insert(0, "n seeds", g.size())
    out.to_csv(f"{OUT}/rev_{name}_summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"\n===== {name}"); print(summ.to_string())
    return df


exp = sys.argv[1:] or ["H", "F", "G", "E"]
if "H" in exp:
    rows = []
    for mode, batches in (("concat (paper)", ["MESA_final_ema", "REV_H_MESA_concat"]), ("mult", ["REV_H_MESA_mult"]),
                          ("mult + bottleneck", ["REV_H_MESA_multbott"]), ("add_feat", ["REV_H_MESA_addfeat"])):
        for b in batches:
            for d in runs(b, "2015-2020"):
                rows.append({"mode": mode, "seed": seed_of(d) if b != "MESA_final_ema" else 0, "run": os.path.basename(os.path.normpath(d)), **score(d)})
                print("H", mode, rows[-1]["run"], flush=True)
    table(rows, ["mode"], "H")
if "F" in exp:
    rows = []
    for ds, arms in (("MESACLIP", (("random days", ["REV_F_MESA_random"]), ("consecutive 2015-2020", ["MESA_final_ema", "REV_H_MESA_concat"]))),
                     ("FOSI", (("random days", ["REV_F_FOSI_random"]), ("consecutive 2015-2020", ["FOSI_final_ema", "REV_F_FOSI_contiguous"])))):
        for arm, batches in arms:
            for b in batches:
                for d in runs(b, None if "random" in b else "2015-2020"):
                    rows.append({"dataset": ds, "arm": arm, "seed": seed_of(d), "run": os.path.basename(os.path.normpath(d)), **score(d)})
                    print("F", ds, arm, rows[-1]["run"], flush=True)
    table(rows, ["dataset", "arm"], "F")
if "G" in exp:
    rows = []
    for d in runs("REV_G_MESA_extrap") + runs("REV_G_MESA_extrap_eval"):
        yr = int(re.search(r"2000-2010_(\d{4})", d).group(1))
        s = score(d)
        bl = rom.window_metrics(os.path.join(d, "eval_data", "fields.npz"))["Bilinear"]
        rows.append({"test year": yr, "gap (yr)": yr - 2010, "run": os.path.basename(os.path.normpath(d)), **s,
                     "bilinear RMSE": bl["RMSE"], "bilinear MAE (= CRPS)": bl["MAE"], "RMSE / bilinear": s["RMSE"] / bl["RMSE"],
                     "CRPS / bilinear MAE": s["CRPS (1 member: MAE)"] / bl["MAE"]})
        print("G", yr, flush=True)
    df = pd.DataFrame(rows).sort_values("test year")
    df.to_csv(f"{OUT}/rev_G_summary.csv", index=False)
    print("\n===== G"); print(df.drop(columns=["run"]).round(4).to_string(index=False))
if "E" in exp:
    rows = []
    for ds, pre in (("MESACLIP", "MESA"), ("FOSI", "FOSI")):
        for lab, batches, det in (("det-MSE seeds 0-2", [f"{pre}_detmse_ema", f"REV_E1_{pre}_detmse_seed1", f"REV_E1_{pre}_detmse_seed2"], True),
                                  ("MC-dropout p=0.1", [f"REV_E2_{pre}_mcdropout", f"REV_E2_{pre}_mcdropout_seed1", f"REV_E2_{pre}_mcdropout_seed2"], False)):
            for b in batches:
                for d in runs(b):
                    w = re.search(r"_(\d{4}-\d{4})_2021_", d).group(1)
                    sd = int(b[-1]) if b[-1].isdigit() else 0
                    rows.append({"dataset": ds, "method": lab, "window": w, "seed": sd, "run": os.path.basename(os.path.normpath(d)), **score(d, det=det)})
                    print("E", ds, lab, w, sd, flush=True)
    table(rows, ["dataset", "method"], "E")
