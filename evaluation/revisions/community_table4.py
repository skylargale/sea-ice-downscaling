"""
Revisions 2026-11, task C.1: updated Table 4 (community nearest-cell values), test year 2021, final paper configuration
(EngressNet + EMA, K=20, coarse-zero rule, original truth; deterministic = det-MSE + EMA; seed 0; 4 training windows).

Cell per community: the grid_iy/grid_ix already stored in evaluation/saved_figs/<batch>/_4window_aggregate/
02_candidate_point_timeseries_data_4window.csv, i.e. the pipeline's nearest valid ocean cell
(rdeb.nearest_valid_index, excluding land | flat truth | flat PIOMAS; build_4window_aggregate_figures.py section 2).

Per window, per sample (MESACLIP: 6 members x 365 days; FOSI: 365 days) at that cell:
  mean truth, mean prediction, bias = mean(pred - truth), MAE, RMSE
then averaged over the 4 windows (bias and means are linear, so they equal the pooled-CSV values; MAE/RMSE are
per-window-then-mean, like Table 2). Ice season = Nov-May of 2021 (Jan-May + Nov-Dec), a fixed calendar window so the
subset doesn't condition on the truth; a truth > 0.15 m subset is written to the CSV as well.
Reads only; writes results/revisions_2026-11/table4_community.{csv,tex}.
"""
import glob
import os

import sys

import numpy as np
import pandas as pd

BASE = "/glade/work/skygale/projects/SeaIceDownscaling"
V6 = BASE
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
from final_set import FS, NAME as FINAL_SET  # noqa: E402
SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"
OUT = f"{BASE}/results/revisions_2026-11"
BATCHES = {"MESACLIP": (FS["MESA"]["eng"], FS["MESA"]["det"]), "FOSI": (FS["FOSI"]["eng"], FS["FOSI"]["det"])}
WINDOWS = FS["MESA"]["windows"]  # FINAL_SET (final_set.py); same windows for both datasets
POINTS = ["Point Hope", "Kivalina", "Kotzebue", "Shishmaref", "Nome"]
ICE_MONTHS = {11, 12, 1, 2, 3, 4, 5}


def run_dir(batch, window):
    m = [d for d in glob.glob(f"{BASE}/results/{batch}/*_{window}_2021_*") if os.path.isdir(d)]
    assert len(m) == 1, (batch, window, m)
    return m[0]


def point_values(npz_path, key, cells):
    a = np.load(npz_path, mmap_mode=None)[key][:, 0]  # (N, H, W)
    return {p: a[:, iy, ix].astype(np.float64) for p, (iy, ix) in cells.items()}


def stats(pred, truth, sel):
    p, t = pred[sel], truth[sel]
    e = p - t
    return {"n": int(sel.sum()), "truth": t.mean(), "pred": p.mean(), "bias": e.mean(),
            "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean())}


rows = []
for ds, (eng, det) in BATCHES.items():
    pts = pd.read_csv(f"{BASE}/evaluation/saved_figs/{eng}/_4window_aggregate/02_candidate_point_timeseries_data_4window.csv")
    cells = {p: tuple(pts.loc[pts.point == p, ["grid_iy", "grid_ix"]].iloc[0].astype(int)) for p in POINTS}
    dist = {p: float(pts.loc[pts.point == p, "dist_km"].iloc[0]) for p in POINTS}
    for w in WINDOWS:
        fe_, fd_ = (os.path.join(run_dir(b, w), "eval_data", "fields.npz") for b in (eng, det))
        months = pd.to_datetime(pd.read_csv(os.path.join(os.path.dirname(fe_), "sample_times.csv"))["time"]).dt.month.values
        truth = point_values(fe_, "Y_test_phys", cells)
        truth_d = point_values(fd_, "Y_test_phys", cells)
        for p in POINTS:
            assert np.allclose(truth[p], truth_d[p]), f"truth differs between {eng} and {det} at {p}, {w}"
        preds = {"EngressNet ens. mean": point_values(fe_, "Y_pred_phys", cells),
                 "Deterministic U-Net": point_values(fd_, "Y_pred_det_phys", cells),
                 "Bilinear": point_values(fe_, "Y_base_phys", cells)}
        for p in POINTS:
            t = truth[p]
            subsets = {"full year": np.ones_like(t, bool), "ice season (Nov-May)": np.isin(months, list(ICE_MONTHS)),
                       "truth > 0.15 m": t > 0.15}
            for method, pv in preds.items():
                for sub, sel in subsets.items():
                    rows.append({"dataset": ds, "community": p, "cell_iy": cells[p][0], "cell_ix": cells[p][1],
                                 "cell_dist_km": round(dist[p], 1), "window": w, "method": method, "subset": sub,
                                 **stats(pv[p], t, sel)})
        print("done", ds, w, flush=True)

long = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
long.to_csv(f"{OUT}/table4_community_per_window{SFX}.csv", index=False)
keys = ["dataset", "community", "cell_iy", "cell_ix", "cell_dist_km", "method", "subset"]
agg = long.groupby(keys, sort=False)[["n", "truth", "pred", "bias", "mae", "rmse"]].mean().reset_index()
agg["n"] = agg["n"].round().astype(int)
agg.to_csv(f"{OUT}/table4_community_long{SFX}.csv", index=False)

# Wide table: one row per (dataset, community); full-year and Nov-May means, plus nearest-cell MAE/RMSE/bias
wide_rows = []
for (ds, p), g in agg.groupby(["dataset", "community"], sort=False):
    def v(method, subset, col):
        return g[(g.method == method) & (g.subset == subset)][col].item()
    r = {"Dataset": ds, "Community": p, "Cell distance (km)": g.cell_dist_km.iloc[0]}
    for sub, tag in (("full year", "yr"), ("ice season (Nov-May)", "NovMay")):
        r[f"Truth mean {tag}"] = v("EngressNet ens. mean", sub, "truth")
        r[f"EngressNet mean {tag}"] = v("EngressNet ens. mean", sub, "pred")
        r[f"Det mean {tag}"] = v("Deterministic U-Net", sub, "pred")
    for m, tag in (("EngressNet ens. mean", "EngressNet"), ("Deterministic U-Net", "Det"), ("Bilinear", "Bilinear")):
        r[f"{tag} MAE yr"] = v(m, "full year", "mae")
        r[f"{tag} RMSE yr"] = v(m, "full year", "rmse")
        r[f"{tag} bias yr"] = v(m, "full year", "bias")
        r[f"{tag} bias NovMay"] = v(m, "ice season (Nov-May)", "bias")
    wide_rows.append(r)
wide = pd.DataFrame(wide_rows)
wide.round(4).to_csv(f"{OUT}/table4_community{SFX}.csv", index=False)

# LaTeX: means (full year / Nov-May) and EngressNet vs deterministic MAE, RMSE, bias
cols = [("Truth mean yr", "Truth NovMay"), ("EngressNet mean yr", "EngressNet mean NovMay"), ("Det mean yr", "Det mean NovMay")]
lines = [r"\begin{tabular}{llrrrrrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{Mean SIT, 2021 (m)} & \multicolumn{3}{c}{Mean SIT, Nov--May (m)} & "
         r"\multicolumn{2}{c}{MAE (m)} & \multicolumn{2}{c}{RMSE (m)} & \multicolumn{2}{c}{Bias, 2021 / Nov--May (m)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
         r"Dataset & Community & Truth & EngressNet & Det. & Truth & EngressNet & Det. & EngressNet & Det. & EngressNet & Det. & EngressNet & Det. \\",
         r"\midrule"]
for _, r in wide.iterrows():
    f = lambda x: f"{x:.3f}"
    lines.append(" & ".join([r["Dataset"], r["Community"],
                             f(r["Truth mean yr"]), f(r["EngressNet mean yr"]), f(r["Det mean yr"]),
                             f(r["Truth mean NovMay"]), f(r["EngressNet mean NovMay"]), f(r["Det mean NovMay"]),
                             f(r["EngressNet MAE yr"]), f(r["Det MAE yr"]), f(r["EngressNet RMSE yr"]), f(r["Det RMSE yr"]),
                             f"{r['EngressNet bias yr']:+.3f} / {r['EngressNet bias NovMay']:+.3f}",
                             f"{r['Det bias yr']:+.3f} / {r['Det bias NovMay']:+.3f}"]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table4_community{SFX}.tex", "w").write("\n".join(lines) + "\n")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(wide.round(3).to_string())
print("truth > 0.15 m subset (EngressNet / det bias):")
print(agg[agg.subset == "truth > 0.15 m"].pivot_table(index=["dataset", "community"], columns="method", values=["n", "bias"]).round(3))
