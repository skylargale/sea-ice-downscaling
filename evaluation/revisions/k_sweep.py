#!/usr/bin/env python3
"""
K-sweep (2026-11-18): the final EngressNet seeds scored with K = 20, 50, 100, 200 members on the 2021 test year (perfect model),
from the 200-member re-inference in results/KSWEEP_{FOSI,MESA} (training/submit_ksweep.py; FOSI one K=200 run per seed, MESA two
K=100 runs per seed, concatenated). The first K members are used for each K. Truth = the raw-restored truth of the seed's original
run (restore_raw_truth.py), checked to match the re-inference's truth. Ocean cells.
Per seed and K: ensemble-mean RMSE, pattern r (per-sample spatial correlation of the ensemble mean, averaged; the tables' definition),
CRPS (K^2 estimator as the tables, and the fair estimator), spread-skill sqrt((K+1)/K * mean s^2) / RMSE, and the extrapolated
K -> inf ensemble-mean RMSE from MSE_K = MSE_inf + s^2/K (s^2 = mean unbiased member variance).
Writes results/revisions_2026-11/k_sweep_<FINAL_SET>.csv (one row per dataset, seed, K) and prints the seed ranges.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
from final_set import NAME as FINAL_SET  # noqa: E402

KS = [20, 50, 100, 200]
CHUNK = 8


def original_run(ds, seed):
    if seed == 0:
        return glob.glob(f"{V6}/results/FINAL2_{ds}_random1825/*/")[0]
    (d,) = [d for d in glob.glob(f"{V6}/results/SENS2_{ds}_final/*/") if json.load(open(d + "run_config.json"))["seed"] == seed]
    return d


def main():
    rows = []
    for ds in ("FOSI", "MESA"):
        runs = sorted(glob.glob(f"{V6}/results/KSWEEP_{ds}/*/eval_data/fields.npz"))
        by_seed = {}
        for r in runs:
            c = json.load(open(os.path.join(os.path.dirname(os.path.dirname(r)), "run_config.json")))
            by_seed.setdefault(c["seed"] % 1000, []).append((c["seed"], r))
        for seed, lst in sorted(by_seed.items()):
            lst.sort()
            parts = [np.load(r)["preds_all_phys"][:, :, 0] for _, r in lst]
            P = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
            del parts
            assert P.shape[1] >= max(KS), (ds, seed, P.shape)
            f = np.load(os.path.join(original_run(ds, seed), "eval_data", "fields.npz"))
            y, ocean = f["Y_test_phys"][:, 0], f["mask_test"][0, 0] <= 0.5
            y_new = np.load(lst[0][1])["Y_test_phys"][:, 0]
            assert np.allclose(y, y_new, atol=1e-5), f"{ds} seed {seed}: truth differs from the original run"
            acc = {K: dict(se=0.0, var=0.0, crps2=0.0, crpsf=0.0, n=0, r=[]) for K in KS}
            for i in range(0, P.shape[0], CHUNK):
                yy = y[i:i + CHUNK]
                for K in KS:
                    e = P[i:i + CHUNK, :K]                                      # (b, K, H, W)
                    mu = e.mean(1)
                    a = acc[K]
                    eo = np.moveaxis(e, 1, -1)[:, ocean].astype(np.float64)    # (b, n_ocean, K)
                    yo = yy[:, ocean].astype(np.float64)
                    mo = mu[:, ocean].astype(np.float64)
                    a["se"] += ((mo - yo) ** 2).sum(); a["n"] += yo.size
                    a["var"] += eo.var(-1, ddof=1).sum()
                    s = np.sort(eo, -1); w = 2 * np.arange(1, K + 1) - K - 1
                    mae = np.abs(eo - yo[..., None]).mean(-1); spr = (s * w).sum(-1)
                    a["crps2"] += (mae - spr / K ** 2).sum(); a["crpsf"] += (mae - spr / (K * (K - 1))).sum()
                    for b in range(yo.shape[0]):
                        if mo[b].std() > 0 and yo[b].std() > 0:
                            a["r"].append(np.corrcoef(mo[b], yo[b])[0, 1])
            for K in KS:
                a = acc[K]
                mse, s2 = a["se"] / a["n"], a["var"] / a["n"]
                rows.append({"dataset": "MESACLIP" if ds == "MESA" else "FOSI", "seed": seed, "K": K, "RMSE": np.sqrt(mse),
                             "Pattern Corr": float(np.mean(a["r"])), "CRPS (K^2)": a["crps2"] / a["n"], "CRPS (fair)": a["crpsf"] / a["n"],
                             "Spread-skill": np.sqrt((K + 1) / K * s2 / mse), "member variance": s2,
                             "RMSE K->inf (extrapolated)": np.sqrt(max(mse - s2 / K, 0))})
                print(ds, seed, K, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
            del P
    df = pd.DataFrame(rows)
    df.to_csv(f"{V6}/results/revisions_2026-11/k_sweep_{FINAL_SET}.csv", index=False)
    pd.set_option("display.width", 250)
    g = df.groupby(["dataset", "K"])
    summ = pd.concat({c: g[c].agg(["mean", "min", "max"]) for c in ["RMSE", "Pattern Corr", "CRPS (K^2)", "CRPS (fair)", "Spread-skill",
                                                                     "RMSE K->inf (extrapolated)"]}, axis=1)
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
