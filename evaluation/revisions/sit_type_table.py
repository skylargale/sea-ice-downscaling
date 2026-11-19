#!/usr/bin/env python3
"""
Calibration by SIT type for the text (2026-11-19): the five SIT types of sit_type_metrics.py, grouped by ENSEMBLE-MEAN SIT, 2021 test
year, EngressNet, for MESACLIP, FOSI and the FOSI coarse-zero-in-loss test run (results/P3_FOSI_czloss, zero rule in the loss as well
as at inference). Per type and for all ocean cells: share of cells, RMS spread sqrt((K+1)/K * mean s^2), ensemble-mean RMSE,
spread-skill ratio, % of truths outside the ensemble (rank 0 or K, as Fig. 7 / spatial_prob_metrics.py), and the rank-0 and rank-K
frequencies relative to chance 1/(K+1). Ranks: random tie-breaking, default_rng(0), rdeb.detection_limited (RANK_TOL, default 0), as
sit_type_metrics.py, so the MESACLIP numbers equal Fig. 8's.
Writes results/revisions_2026-11/sit_type_table_<FINAL_SET>.csv and prints it.
Run from the project root:  python evaluation/revisions/sit_type_table.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "evaluation"))
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
import run_daily_eval_batch as rdeb  # noqa: E402
from final_set import FS, NAME as FINAL_SET  # noqa: E402
from sit_type_metrics import TYPE_EDGES, TYPE_LABELS  # noqa: E402

CHUNK = 20


def one_dir(pattern):
    m = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    assert len(m) == 1, (pattern, m)
    return m[0]


def type_stats(run_dir, label):
    f = np.load(os.path.join(run_dir, "eval_data", "fields.npz"))
    preds, Y, mask = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"]
    K = preds.shape[1]
    rng = np.random.default_rng(0)
    nt = len(TYPE_LABELS)
    keys = ["n", "sum_var", "sum_sq", "out", "r0", "rK"]
    a = {k: np.zeros(nt + 1) for k in keys}
    for i in range(0, preds.shape[0], CHUNK):
        oc = mask[i:i + CHUNK, 0] <= 0.5
        ens = np.ascontiguousarray(np.moveaxis(preds[i:i + CHUNK, :, 0], 1, -1))[oc].astype(np.float32)
        y = Y[i:i + CHUNK, 0][oc]
        er, yr = rdeb.detection_limited(ens, y)
        ranks = (er < yr[:, None]).sum(1) + rng.integers(0, (er == yr[:, None]).sum(1) + 1)
        mu = ens.mean(1); var = ens.var(1, ddof=1)
        t = np.digitize(mu, TYPE_EDGES, right=True)
        for b in range(nt + 1):
            s = (t == b) if b < nt else slice(None)
            a["n"][b] += y[s].size; a["sum_var"][b] += float(var[s].sum()); a["sum_sq"][b] += float(((mu[s] - y[s]) ** 2).sum())
            a["r0"][b] += float((ranks[s] == 0).sum()); a["rK"][b] += float((ranks[s] == K).sum())
        print(f"  {label}: samples {i}-{i + oc.shape[0] - 1}", flush=True)
    a["out"] = a["r0"] + a["rK"]
    rows = []
    for b, name in enumerate(TYPE_LABELS + ["All ocean cells"]):
        n = a["n"][b]; spr = np.sqrt((K + 1) / K * a["sum_var"][b] / n); rmse = np.sqrt(a["sum_sq"][b] / n)
        rows.append({"run": label, "SIT type (m)": name, "share of cells (%)": 100 * n / a["n"][nt], "RMS spread (m)": spr,
                     "ens-mean RMSE (m)": rmse, "spread-skill": spr / rmse, "truths outside ensemble (%)": 100 * a["out"][b] / n,
                     "ideal outside (%)": 100 * 2 / (K + 1), "rank 0 / chance": a["r0"][b] / n * (K + 1),
                     f"rank {K} / chance": a["rK"][b] / n * (K + 1), "K": K, "N": int(n)})
    return rows


def main():
    runs = [("MESACLIP", one_dir(f"{V6}/results/{FS['MESA']['eng']}/*{FS['map_window']}*")),
            ("FOSI", one_dir(f"{V6}/results/{FS['FOSI']['eng']}/*{FS['map_window']}*")),
            ("FOSI, zero rule in the loss", one_dir(f"{V6}/results/P3_FOSI_czloss/*"))]
    rows = []
    for label, d in runs:
        print(label, d, flush=True)
        rows += type_stats(d, label)
    df = pd.DataFrame(rows)
    df.to_csv(f"{V6}/results/revisions_2026-11/sit_type_table_{FINAL_SET}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(df.drop(columns=["N"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
