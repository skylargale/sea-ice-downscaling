"""
Revisions 2026-11, C.2 follow-up summary: for each window, <dir>/<window>_orig vs <dir>/<window>_cons (submit_u10_check.sh):
RMSE, CRPS, corrected spread-skill (ocean cells, 2021) of each, and the change in the ensemble-mean SIT field
(mean |cons - orig| over ocean cells and samples, 99th percentile, max). Writes <dir>/u10_check_summary.csv and appends nothing
else. Usage: python compare_u10_runs.py <dir>
"""
import os
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
rows = []
for w in ("2000-2005", "2005-2010", "2010-2015", "2015-2020"):
    f = {v: os.path.join(d, f"{w}_{v}", "eval_data", "fields.npz") for v in ("orig", "cons")}
    if not all(os.path.exists(p) for p in f.values()):
        print("missing", w); continue
    z = {v: np.load(p) for v, p in f.items()}
    oc = z["orig"]["mask_test"][:, 0] <= 0.5
    T = z["orig"]["Y_test_phys"][:, 0]
    r = {"window": w}
    for v in ("orig", "cons"):
        P = z[v]["preds_all_phys"][:, :, 0]
        K = P.shape[1]
        E = np.moveaxis(P, 1, -1)[oc].astype(np.float64); y = T[oc].astype(np.float64)
        mu = E.mean(1); rmse = np.sqrt(((mu - y) ** 2).mean())
        wts = 2 * np.arange(1, K + 1) - K - 1
        crps = (np.abs(E - y[:, None]).mean(1) - (np.sort(E, 1) * wts).sum(1) / K ** 2).mean()
        r[f"RMSE {v}"], r[f"CRPS {v}"] = rmse, crps
        r[f"SSR {v}"] = np.sqrt((K + 1) / K * E.var(1, ddof=1).mean()) / rmse
        r[f"mu {v}"] = mu
    dmu = np.abs(r.pop("mu cons") - r.pop("mu orig"))
    r.update({"ens-mean |change| mean (m)": dmu.mean(), "ens-mean |change| p99 (m)": np.percentile(dmu, 99),
              "ens-mean |change| max (m)": dmu.max()})
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(d, "u10_check_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(df.round(4).to_string(index=False))
