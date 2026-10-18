"""Why does the MESA final ensemble mean lose to the det-MSE U-Net on RMSE?
Per window: RMSE of mean of first k members (k=1..20) -> extrapolate K->inf; median RMSE;
3x3-smoothed mean; regime / month breakdown of SSE vs det-MSE; open-water clamp bias."""
import sys, glob, os, json
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

ROOT = "/glade/work/skygale/projects/SeaIceDownscaling/results"
OUT = sys.argv[1]
BATCHES = sys.argv[2:]
DET = sorted(glob.glob(f"{ROOT}/{os.environ.get('DET_BATCH', 'MESA_deterministic_mse_avg')}/*/eval_data/fields.npz"))
KS = [1, 2, 5, 10, 20]
BINS = [-1, 0.15, 0.5, 1.0, 99]
LAB = ["open<0.15", "thin0.15-0.5", "mod0.5-1", "thick>1"]
CH = 64
rows = []
for batch in BATCHES:
    for wi, fp in enumerate(sorted(glob.glob(f"{ROOT}/{batch}/*/eval_data/fields.npz"))):
        run = fp.split("/")[-3]
        print(run, flush=True)
        f = np.load(fp); g = np.load(DET[wi])
        P = f["preds_all_phys"]; Y = f["Y_test_phys"]; M = f["mask_test"]; Z = f["Y_pred_det_phys"]
        D = g["Y_pred_det_phys"]
        assert np.allclose(g["Y_test_phys"][:5], Y[:5], atol=1e-3), np.abs(g["Y_test_phys"][:5]-Y[:5]).max()
        times = pd.to_datetime(pd.read_csv(os.path.dirname(fp) + "/sample_times.csv")["time"]).dt.month.values
        N, K = P.shape[:2]
        acc = {}
        def add(k, v): acc[k] = acc.get(k, 0.0) + float(v)
        for i in range(0, N, CH):
            ens = P[i:i+CH, :, 0].astype(np.float64)  # n,K,H,W
            y = Y[i:i+CH, 0].astype(np.float64); oc = M[i:i+CH, 0] <= 0.5
            d = D[i:i+CH, 0].astype(np.float64); z = Z[i:i+CH, 0].astype(np.float64)
            mu = ens.mean(1)
            mus = uniform_filter(mu, size=(1, 3, 3), mode="nearest")
            ds = uniform_filter(d, size=(1, 3, 3), mode="nearest")
            med = np.median(ens, 1)
            var = ens.var(1, ddof=1)
            add("n", oc.sum()); add("sum_var", var[oc].sum())
            for k in KS:
                add(f"sse_k{k}", ((ens[:, :k].mean(1) - y)[oc] ** 2).sum())
            # unbiased infinite-ensemble MSE estimate, per cell: (mu-y)^2 - var/K
            add("sse_inf", (((mu - y) ** 2) - var / K)[oc].sum())
            add("sse_med", ((med - y)[oc] ** 2).sum())
            add("sse_mu_s3", ((mus - y)[oc] ** 2).sum())
            add("sse_det", ((d - y)[oc] ** 2).sum()); add("sse_det_s3", ((ds - y)[oc] ** 2).sum())
            add("sse_zero", ((z - y)[oc] ** 2).sum())
            add("bias_mu", (mu - y)[oc].sum()); add("bias_det", (d - y)[oc].sum())
            add("sxy", ((mu - mu[oc].mean()) * (y - y[oc].mean()))[oc].sum())
            reg = np.digitize(y, BINS) - 1
            for b, lab in enumerate(LAB):
                m = oc & (reg == b)
                add(f"n_{lab}", m.sum())
                add(f"sse_mu_{lab}", ((mu - y)[m] ** 2).sum()); add(f"sse_det_{lab}", ((d - y)[m] ** 2).sum())
                add(f"var_{lab}", var[m].sum()); add(f"bias_mu_{lab}", (mu - y)[m].sum()); add(f"bias_det_{lab}", (d - y)[m].sum())
            mon = times[i:i+CH][:, None, None] * np.ones_like(y, dtype=int)
            for mo in range(1, 13):
                m = oc & (mon == mo)
                add(f"sse_mu_m{mo:02d}", ((mu - y)[m] ** 2).sum()); add(f"sse_det_m{mo:02d}", ((d - y)[m] ** 2).sum()); add(f"n_m{mo:02d}", m.sum())
        n = acc["n"]
        r = {"batch": batch, "run": run, "K": K, "n": n,
             "ssr": np.sqrt((K + 1) / K * acc["sum_var"] / n) / np.sqrt(acc["sse_k20"] / n) if K == 20 else np.nan}
        for k in KS:
            if k <= K: r[f"rmse_k{k}"] = np.sqrt(acc[f"sse_k{k}"] / n)
        for key in ["inf", "med", "mu_s3", "det", "det_s3", "zero"]:
            r[f"rmse_{key}"] = np.sqrt(max(acc[f"sse_{key}"], 0) / n)
        r["bias_mu"] = acc["bias_mu"] / n; r["bias_det"] = acc["bias_det"] / n
        for lab in LAB:
            nb = acc[f"n_{lab}"]
            r[f"frac_{lab}"] = nb / n
            r[f"rmse_mu_{lab}"] = np.sqrt(acc[f"sse_mu_{lab}"] / nb); r[f"rmse_det_{lab}"] = np.sqrt(acc[f"sse_det_{lab}"] / nb)
            r[f"dsse_share_{lab}"] = (acc[f"sse_mu_{lab}"] - acc[f"sse_det_{lab}"])
            r[f"ssr_{lab}"] = np.sqrt((K + 1) / K * acc[f"var_{lab}"] / nb) / r[f"rmse_mu_{lab}"]
            r[f"bias_mu_{lab}"] = acc[f"bias_mu_{lab}"] / nb; r[f"bias_det_{lab}"] = acc[f"bias_det_{lab}"] / nb
        tot = acc["sse_k20"] - acc["sse_det"]
        for lab in LAB: r[f"dsse_share_{lab}"] /= tot
        for mo in range(1, 13):
            nm = acc[f"n_m{mo:02d}"]
            r[f"rmse_mu_m{mo:02d}"] = np.sqrt(acc[f"sse_mu_m{mo:02d}"] / nm); r[f"rmse_det_m{mo:02d}"] = np.sqrt(acc[f"sse_det_m{mo:02d}"] / nm)
        rows.append(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        del P, Y, M, Z, D
print("done", flush=True)
