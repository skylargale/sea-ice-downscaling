"""
Revisions 2026-11, H.1 summary: for each ablation run under <dir>/ablate_<stage>/eval_data/fields.npz (written by
submit_noise_ablation.sh), ocean cells, 2021:
  ensemble-mean RMSE, CRPS, corrected spread-skill sqrt((K+1)/K mean s^2)/RMSE overall and per thickness type (regimes by
  ensemble-mean SIT, as in Fig. 7), mean ensemble variance, and the % of the un-ablated ensemble variance lost when that
  stage's noise is switched off.
Writes <dir>/noise_ablation_summary.csv. Usage: python summarize_noise_ablation.py <dir> [<dir> ...]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

EDGES = [0.15, 0.5, 1.0]
REGIMES = ["open (0-0.15)", "thin (0.15-0.5)", "moderate (0.5-1)", "thick (>1)"]
CHUNK = 64


def stats(fp):
    f = np.load(fp)
    P, T, M = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"]
    N, K = P.shape[:2]
    w = 2 * np.arange(1, K + 1) - K - 1
    acc = {k: np.zeros(5) for k in ("n", "var", "sse")}
    crps = 0.0
    for i in range(0, N, CHUNK):
        oc = M[i:i + CHUNK, 0] <= 0.5
        e = np.moveaxis(P[i:i + CHUNK, :, 0], 1, -1)[oc].astype(np.float64)
        y = T[i:i + CHUNK, 0][oc].astype(np.float64)
        mu = e.mean(1)
        v = e.var(1, ddof=1)
        se = (mu - y) ** 2
        crps += (np.abs(e - y[:, None]).mean(1) - (np.sort(e, 1) * w).sum(1) / K ** 2).sum()
        b = np.digitize(mu, EDGES)
        for r in range(5):
            s = slice(None) if r == 4 else (b == r)
            acc["n"][r] += se[s].size
            acc["var"][r] += v[s].sum()
            acc["sse"][r] += se[s].sum()
    n = acc["n"]
    rmse = np.sqrt(acc["sse"] / n)
    ssr = np.sqrt((K + 1) / K * acc["var"] / n) / rmse
    out = {"K": K, "RMSE": rmse[4], "CRPS": crps / n[4], "mean ens variance": acc["var"][4] / n[4], "spread-skill": ssr[4]}
    out.update({f"SSR {r}": ssr[i] for i, r in enumerate(REGIMES)})
    out.update({f"var {r}": acc["var"][i] / n[i] for i, r in enumerate(REGIMES)})
    return out


for d in sys.argv[1:]:
    rows = []
    for fp in sorted(glob.glob(os.path.join(d, "ablate_*", "eval_data", "fields.npz"))):
        stage = os.path.basename(os.path.dirname(os.path.dirname(fp))).replace("ablate_", "")
        print(d, stage, flush=True)
        rows.append({"ablated stage": stage, **stats(fp)})
    if not rows:
        print("no runs in", d)
        continue
    df = pd.DataFrame(rows).set_index("ablated stage")
    if "none" in df.index:
        base = df.loc["none"]
        df["% variance lost"] = 100 * (1 - df["mean ens variance"] / base["mean ens variance"])
        for r in REGIMES:
            df[f"% var lost {r}"] = 100 * (1 - df[f"var {r}"] / base[f"var {r}"])
        df["CRPS change %"] = 100 * (df["CRPS"] / base["CRPS"] - 1)
    order = [s for s in ("none", "d3", "d2", "d1", "final", "b") if s in df.index]
    df = df.loc[order]
    df.to_csv(os.path.join(d, "noise_ablation_summary.csv"))
    pd.set_option("display.width", 250)
    cols = ["RMSE", "CRPS", "CRPS change %", "spread-skill"] + [f"SSR {r}" for r in REGIMES] + ["% variance lost"] + \
           [f"% var lost {r}" for r in REGIMES]
    print(df[[c for c in cols if c in df.columns]].round(4).to_string())
