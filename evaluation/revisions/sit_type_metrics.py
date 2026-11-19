#!/usr/bin/env python3
"""
Rank histogram, spread-skill and CRPS by sea ice thickness type (2026-11-19, advisor request): five types
[0, 0.15], (0.15, 0.70], (0.70, 1.20], (1.20, 3.00], > 3.00 m, replacing the four "regimes" (0-0.15, 0.15-0.5, 0.5-1.0, >1.0 m)
of evaluation/build_4window_aggregate_figures.py. Same definitions as that script, so the numbers are comparable:
  - ocean cells, the final EngressNet MESACLIP run (2021 test year), its K members;
  - rank histogram and spread-skill grouped by ENSEMBLE-MEAN SIT (conditioning on the truth distorts calibration
    diagnostics), CRPS grouped by TRUE SIT;
  - ranks with random tie-breaking, numpy default_rng(seed=0) (window index 0), RANK_TOL from the environment (default 0);
  - spread-skill = sqrt((K+1)/K * mean ensemble variance) / RMSE of the ensemble mean; CRPS = K^2 estimator.
Check: with the old four bins (left-closed, as np.digitize) the script reproduces the saved *_by_regime_data_4window.csv
exactly before writing the new files.
Writes figures/figures_v2/data_<FINAL_SET>/{rank_histogram_by_type,spread_skill_by_type,crps_by_type}.csv (read by
make_v2_figures.py, Fig. 8 and the supplementary CRPS-by-type figure).
Run from the project root:  python evaluation/revisions/sit_type_metrics.py
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

# right-closed types; the first includes 0 (open water)
TYPE_EDGES = [0.15, 0.70, 1.20, 3.00]
TYPE_LABELS = ["(0.00-0.15]", "(0.15-0.70]", "(0.70-1.20]", "(1.20-3.00]", ">3.00"]   # m
OLD_EDGES = [0.15, 0.5, 1.0]   # the previous regimes (np.digitize, left-closed), for the check only
CHUNK = 20                     # samples per chunk


def main():
    eng = FS["MESA"]["eng"]
    rd = sorted(d for d in glob.glob(f"{V6}/results/{eng}/*{FS['map_window']}*") if os.path.isdir(d))
    assert len(rd) == 1, rd
    f = np.load(os.path.join(rd[0], "eval_data", "fields.npz"))
    preds, Y, mask = f["preds_all_phys"], f["Y_test_phys"], f["mask_test"]
    K = preds.shape[1]
    rng = np.random.default_rng(0)
    ocean_all = mask[:, 0] <= 0.5
    nt, no = len(TYPE_LABELS), len(OLD_EDGES) + 1
    acc = {k: {"rank": np.zeros((n, K + 1), np.int64), "n": np.zeros(n + 1), "sum_truth": np.zeros(n + 1),
               "sum_var": np.zeros(n + 1), "sum_sq": np.zeros(n + 1), "sum_bias": np.zeros(n + 1),
               "crps_n": np.zeros(n), "crps_sum": np.zeros(n), "crps_mt": np.zeros(n)} for k, n in (("new", nt), ("old", no))}
    crps_all = [0, 0.0, 0.0]
    # the original draws the tie-breaking integers for all ocean cells at once, in sample-major order; drawing them chunk by
    # chunk in the same order gives the same stream only if each call consumes the same numbers, which rng.integers with an
    # array of highs does (one draw per element), so chunking preserves the exact ranks.
    for i in range(0, preds.shape[0], CHUNK):
        oc = ocean_all[i:i + CHUNK]
        ens = np.ascontiguousarray(np.moveaxis(preds[i:i + CHUNK, :, 0], 1, -1))[oc].astype(np.float32)   # (M, K)
        y = Y[i:i + CHUNK, 0][oc]
        er, yr = rdeb.detection_limited(ens, y)
        below = (er < yr[:, None]).sum(1); ties = (er == yr[:, None]).sum(1)
        ranks = below + rng.integers(0, ties + 1)
        mu = ens.mean(1); var = ens.std(1, ddof=1) ** 2; err = mu - y
        s = np.sort(ens, 1); w = 2 * np.arange(1, K + 1) - K - 1
        crps = np.abs(ens - y[:, None]).mean(1) - (s * w).sum(1) / K ** 2
        crps_all[0] += crps.size; crps_all[1] += float(crps.sum()); crps_all[2] += float(y.sum())
        for key, fc_idx, tr_idx, n in (("new", np.digitize(mu, TYPE_EDGES, right=True), np.digitize(y, TYPE_EDGES, right=True), nt),
                                       ("old", np.digitize(mu, OLD_EDGES), np.digitize(y, OLD_EDGES), no)):
            a = acc[key]
            for b in range(n + 1):
                sel = (fc_idx == b) if b < n else slice(None)
                a["n"][b] += y[sel].size; a["sum_truth"][b] += float(y[sel].sum()); a["sum_var"][b] += float(var[sel].sum())
                a["sum_sq"][b] += float((err[sel] ** 2).sum()); a["sum_bias"][b] += float(err[sel].sum())
                if b < n:
                    a["rank"][b] += np.bincount(ranks[sel], minlength=K + 1)
                    t = tr_idx == b
                    a["crps_n"][b] += t.sum(); a["crps_sum"][b] += float(crps[t].sum()); a["crps_mt"][b] += float(y[t].sum())
        print(f"  samples {i}-{i + oc.shape[0] - 1} done", flush=True)

    def tables(a, labels):
        exp = 1.0 / (K + 1)
        rk = pd.DataFrame([{"SIT type": lab, "rank": r, "count": int(a["rank"][b, r]),
                            "frequency": a["rank"][b, r] / max(a["rank"][b].sum(), 1),
                            "relative_frequency": a["rank"][b, r] / max(a["rank"][b].sum(), 1) / exp}
                           for b, lab in enumerate(labels) for r in range(K + 1)])
        ss = []
        for b, lab in enumerate(labels + ["All ocean cells"]):
            n = a["n"][b]; spr = np.sqrt((K + 1) / K * a["sum_var"][b] / n); rmse = np.sqrt(a["sum_sq"][b] / n)
            ss.append({"SIT type": lab, "N": int(n), "Mean true SIT (m)": round(a["sum_truth"][b] / n, 4),
                       "Ensemble-mean bias (m)": round(a["sum_bias"][b] / n, 4), "RMS spread, (K+1)/K-corrected (m)": round(spr, 4),
                       "RMSE (m)": round(rmse, 4), "Spread-skill ratio (1.0 = calibrated)": round(spr / rmse, 4)})
        cr = [{"SIT type": "All (domain-wide)", "N": crps_all[0], "Mean true SIT (m)": round(crps_all[2] / crps_all[0], 4),
               "CRPS (m)": round(crps_all[1] / crps_all[0], 4)}]
        cr += [{"SIT type": lab, "N": int(a["crps_n"][b]), "Mean true SIT (m)": round(a["crps_mt"][b] / a["crps_n"][b], 4),
                "CRPS (m)": round(a["crps_sum"][b] / a["crps_n"][b], 4)} for b, lab in enumerate(labels)]
        return rk, pd.DataFrame(ss), pd.DataFrame(cr)

    # check against the saved four-regime aggregate
    agg = f"{V6}/evaluation/saved_figs/{eng}/_4window_aggregate"
    rk_o, ss_o, cr_o = tables(acc["old"], ["r0", "r1", "r2", "r3"])
    ref_rk = pd.read_csv(f"{agg}/14b_rank_histogram_by_regime_data_4window.csv", index_col=0)
    ref_ss = pd.read_csv(f"{agg}/15_spread_skill_by_regime_data_4window.csv", index_col=0)
    ref_cr = pd.read_csv(f"{agg}/16_crps_by_regime_data_4window.csv", index_col=0)
    assert np.array_equal(rk_o["count"].values, ref_rk["count"].values), "rank counts differ from the saved aggregate"
    for c in ["N", "RMSE (m)", "RMS spread, (K+1)/K-corrected (m)", "Spread-skill ratio (1.0 = calibrated)"]:
        assert np.allclose(ss_o[c].values, ref_ss[c].values, atol=2e-4), c
    assert np.allclose(cr_o["CRPS (m)"].values, ref_cr["CRPS (m)"].values, atol=2e-4), "CRPS differs from the saved aggregate"
    print("check passed: the old four regimes reproduce the saved aggregate")

    out = f"{V6}/figures/figures_v2/data_{FINAL_SET}"
    rk, ss, cr = tables(acc["new"], TYPE_LABELS)
    rk.to_csv(f"{out}/rank_histogram_by_type.csv"); ss.to_csv(f"{out}/spread_skill_by_type.csv"); cr.to_csv(f"{out}/crps_by_type.csv")
    pd.set_option("display.width", 200)
    print(ss.to_string()); print(cr.to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
