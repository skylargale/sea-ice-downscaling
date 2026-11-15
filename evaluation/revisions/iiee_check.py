"""
Revisions 2026-11, follow-up 2: why does bilinear have the lowest IIEE? Final configuration, 4 windows, 2021, ocean cells.

For bilinear (Y_base_phys, land zeroed), the deterministic U-Net (det-MSE + EMA), the EngressNet ensemble mean and the
EngressNet members (per member, then averaged), with the same definition as Tables 2/3 (fe.ice_edge_error: cos-lat weighted
fraction of ocean area where pred > thr and truth > thr disagree):
  IIEE at thr = 1e-4 m (paper) and 0.15 m (sensitivity), split into
    over  = pred ice, truth open water      under = pred open water, truth ice      (IIEE = over + under)
  mean truth SIT in the 'under' cells (are the missed cells trace ice?)
Zero-constraint check: fraction of coarse-zero ocean cells (Y_base_phys == 0) where each method is > 1e-4 m (should be 0).
Also the full Table 2/3 metric row for bilinear from the existing ocean-metric files.
Writes results/revisions_2026-11/iiee_check.csv and tables23_three_methods.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "training"))
sys.path.insert(0, os.path.join(V6, "evaluation"))
import functions_engressnet as fe  # noqa: E402
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
from final_set import FS, NAME as FINAL_SET  # noqa: E402
SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"

OUT = f"{V6}/results/revisions_2026-11"
THR = (1e-4, 0.15)


def run_dir(batch, w):
    m = [d for d in glob.glob(f"{V6}/results/{batch}/*_{w}_2021_*") if os.path.isdir(d)]
    assert len(m) == 1, (batch, w, m)
    return m[0]


def parts(pred, truth, ocean, wlat, thr):
    """pred, truth (N, H, W) numpy; returns IIEE, over, under (area fractions of ocean), mean truth in under cells."""
    p, t = pred > thr, truth > thr
    w = np.broadcast_to(wlat, pred.shape)[ocean]
    over = (p & ~t)[ocean]; under = (~p & t)[ocean]
    tot = w.sum()
    return (w * (over | under)).sum() / tot, (w * over).sum() / tot, (w * under).sum() / tot, \
        float(truth[ocean][under].mean()) if under.any() else np.nan


rows = []
for ds, eng, det in (("MESACLIP", FS["MESA"]["eng"], FS["MESA"]["det"]), ("FOSI", FS["FOSI"]["eng"], FS["FOSI"]["det"])):
    for w in FS["MESA"]["windows"]:
        fe_ = np.load(os.path.join(run_dir(eng, w), "eval_data", "fields.npz"))
        fd_ = np.load(os.path.join(run_dir(det, w), "eval_data", "fields.npz"))
        T = fe_["Y_test_phys"][:, 0]
        ocean = fe_["mask_test"][:, 0] <= 0.5
        H = T.shape[1]
        hlat = fe_["hlat"][:H] if fe_["hlat"].shape[0] != H else fe_["hlat"]
        wlat = np.cos(np.deg2rad(hlat))[None, :, None]
        base = fe_["Y_base_phys"][:, 0] * ocean
        P = fe_["preds_all_phys"][:, :, 0]
        meth = {"Bilinear": base, "Deterministic U-Net": fd_["Y_pred_det_phys"][:, 0], "EngressNet ensemble mean": fe_["Y_pred_phys"][:, 0]}
        cz = ocean & (base <= 0)
        for name, f in list(meth.items()) + [("EngressNet members (mean over members)", None)]:
            r = {"dataset": ds, "window": w, "method": name}
            if f is not None:
                r["zero rule: coarse-zero cells with pred > 1e-4 (%)"] = 100 * float((f[cz] > 1e-4).mean())
            else:
                r["zero rule: coarse-zero cells with pred > 1e-4 (%)"] = 100 * float((P[:, :, :, :][np.broadcast_to(cz[:, None], P.shape)] > 1e-4).mean())
            for thr in THR:
                if f is not None:
                    i, o, u, tu = parts(f, T, ocean, wlat, thr)
                else:
                    vals = np.array([parts(P[:, k], T, ocean, wlat, thr) for k in range(P.shape[1])])
                    i, o, u, tu = np.nanmean(vals, 0)
                r.update({f"IIEE thr={thr:g}": i, f"over thr={thr:g}": o, f"under thr={thr:g}": u,
                          f"mean truth in under cells thr={thr:g} (m)": tu})
            if f is not None:  # cross-check against the paper's function at the paper threshold
                r["IIEE fe.ice_edge_error (1e-4)"] = fe.ice_edge_error(torch.from_numpy(f[:, None]), torch.from_numpy(T[:, None]),
                                                                       threshold=1e-4, mask_bool=torch.from_numpy(ocean),
                                                                       lat=torch.from_numpy(hlat))
            rows.append(r)
        print("done", ds, w, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/iiee_check_per_window{SFX}.csv", index=False)
mean = df.groupby(["dataset", "method"], sort=False).mean(numeric_only=True).reset_index()
mean.to_csv(f"{OUT}/iiee_check{SFX}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(mean.round(4).to_string(index=False))

# full Table 2/3 rows incl. bilinear (existing ocean-metric files, 4-window mean; CRPS = MAE for single-valued methods)
C = f"{V6}/evaluation/corrected_calibration"
t = []
for ds, eng, det in (("MESACLIP", FS["MESA"]["eng"], FS["MESA"]["det"]), ("FOSI", FS["FOSI"]["eng"], FS["FOSI"]["det"])):
    oe, od = pd.read_csv(f"{C}/{eng}_ocean_metrics.csv"), pd.read_csv(f"{C}/{det}_ocean_metrics.csv")
    cal = pd.read_csv(f"{C}/{eng}.csv"); cal = cal[cal.run != "POOLED"]
    g = lambda d_, m: d_[(d_.run == "MEAN over windows") & (d_.method == m)].iloc[0]
    for lab, row, crps in (("EngressNet ensemble mean", g(oe, "EngressNet ensemble mean"), cal["CRPS (ocean)"].mean()),
                           ("Deterministic U-Net", g(od, "Deterministic (zero-noise pass)"), None),
                           ("Bilinear", g(oe, "Bilinear"), None)):
        rr = {"dataset": ds, "method": lab, **row.drop(["run", "method"]).to_dict()}
        rr["CRPS (single-valued: MAE)"] = crps if crps is not None else rr["MAE"]
        mm = mean[(mean.dataset == ds) & (mean.method == lab)]
        rr["IIEE thr=0.15"] = float(mm["IIEE thr=0.15"].iloc[0]) if len(mm) else np.nan
        t.append(rr)
t = pd.DataFrame(t)
t.to_csv(f"{OUT}/tables23_three_methods{SFX}.csv", index=False)
print(t.round(4).to_string(index=False))
