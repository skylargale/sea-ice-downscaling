"""
Revisions 2026-11, E: like-for-like comparison with the seed/window ensemble (12 det-MSE models over the 4 windows, which
pools training information from all of 2000-2020). Scores, on the 2021 test year (ocean cells, original truth, zero rule):
  EngressNet pooled over the 4 windows (4 x 20 = 80 members), EngressNet single window (mean of the 4), and the deterministic
  4-window mean (point value); CRPS (K^2 estimator), ensemble-mean RMSE, corrected spread-skill, rank outside fraction.
Writes results/revisions_2026-11/pooled_window_ensembles.csv.
"""
import glob, os, sys
import numpy as np, pandas as pd
V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "evaluation"))
import run_daily_eval_batch as rdeb  # noqa: E402
W = ("2000-2005", "2005-2010", "2010-2015", "2015-2020")
rows = []
for ds, eng, det in (("MESACLIP", "MESA_final_ema", "MESA_detmse_ema"), ("FOSI", "FOSI_final_ema", "FOSI_detmse_ema")):
    fe_ = [np.load(glob.glob(f"{V6}/results/{eng}/*_{w}_2021_*/eval_data/fields.npz")[0]) for w in W]
    T = fe_[0]["Y_test_phys"][:, 0]; oc = fe_[0]["mask_test"][:, 0] <= 0.5
    P = [f["preds_all_phys"][:, :, 0] for f in fe_]
    D = np.mean([np.load(glob.glob(f"{V6}/results/{det}/*_{w}_2021_*/eval_data/fields.npz")[0])["Y_pred_det_phys"][:, 0] for w in W], 0)
    acc = {"n": 0, "crps": 0.0, "var": 0.0, "sse": 0.0, "out": 0, "sse_det": 0.0, "ae_det": 0.0}
    rng = np.random.default_rng(0)
    for i in range(0, T.shape[0], 32):
        o = oc[i:i + 32]
        e = np.concatenate([np.moveaxis(p[i:i + 32], 1, -1)[o] for p in P], axis=1).astype(np.float64)
        y = T[i:i + 32][o].astype(np.float64); K = e.shape[1]; w = 2 * np.arange(1, K + 1) - K - 1
        mu = e.mean(1)
        acc["crps"] += (np.abs(e - y[:, None]).mean(1) - (np.sort(e, 1) * w).sum(1) / K ** 2).sum()
        acc["var"] += e.var(1, ddof=1).sum(); acc["sse"] += ((mu - y) ** 2).sum(); acc["n"] += y.size
        er, tr = rdeb.detection_limited(e, y); b, t = (er < tr[:, None]).sum(1), (er == tr[:, None]).sum(1)
        rk = b + rng.integers(0, t + 1); acc["out"] += ((rk == 0) | (rk == K)).sum()
        d = D[i:i + 32][o]; acc["sse_det"] += ((d - y) ** 2).sum(); acc["ae_det"] += np.abs(d - y).sum()
    n = acc["n"]; rmse = np.sqrt(acc["sse"] / n)
    rows.append({"dataset": ds, "estimate": "EngressNet pooled over 4 windows (80 members)", "K": 80, "CRPS": acc["crps"] / n,
                 "ens-mean RMSE": rmse, "spread-skill": np.sqrt(81 / 80 * acc["var"] / n) / rmse, "rank outside frac": acc["out"] / n})
    rows.append({"dataset": ds, "estimate": "Deterministic 4-window mean (point)", "K": 1, "CRPS": acc["ae_det"] / n,
                 "ens-mean RMSE": np.sqrt(acc["sse_det"] / n)})
    print(pd.DataFrame(rows[-2:]).round(4).to_string(index=False), flush=True)
pd.DataFrame(rows).to_csv(f"{V6}/results/revisions_2026-11/pooled_window_ensembles.csv", index=False)
