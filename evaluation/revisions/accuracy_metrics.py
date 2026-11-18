#!/usr/bin/env python3
"""
Accuracy metrics of the ensemble mean / single prediction for the Section 4.1 accuracy figure (2026-11-18): bilinear,
deterministic U-Net (det-MSE), EngressNet ensemble mean (20 members) and the seed-ensemble mean (10 det-MSE U-Nets, seeds 0-9 on
the same 1,825 days); final model, 2021 test year, ocean cells, both datasets.

Same definitions as the tables (functions imported from member_scores.py / spatial_prob_metrics.py): RMSE, MAE, pattern r,
gradient MAE, SSIM (recompute_ocean_metrics.py definitions), IIEE at 1e-4 m and 0.15 m (ocean-masked, cos-lat), ACC (uncentred,
vs the training-years day-of-year truth climatology). Bilinear is scored with land set to 0, as in the tables.
Writes results/revisions_2026-11/accuracy_metrics_<FINAL_SET>.csv.
Usage (project root): python evaluation/revisions/accuracy_metrics.py
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
import member_scores as ms  # noqa: E402
from final_set import FS, NAME as FINAL_SET  # noqa: E402

spm, fe = ms.spm, ms.fe
OUT = os.path.join(V6, "results", "revisions_2026-11", f"accuracy_metrics_{FINAL_SET}.csv")


def scores(pred, truth, ocean, hlat, clim_t):
    s = ms.single(pred, truth, ocean, hlat, clim_t)   # RMSE, Pattern Corr, Grad MAE, SSIM, IIEE (1e-4 m), MAE, FSS, ACC
    s["IIEE (1e-4 m)"] = s.pop("IIEE")
    oc = torch.from_numpy(ocean).expand(pred.shape[0], -1, -1)
    s["IIEE (0.15 m)"] = float(fe.ice_edge_error(torch.from_numpy(np.ascontiguousarray(pred[:, None])), torch.from_numpy(truth[:, None]),
                                                 threshold=0.15, mask_bool=oc, lat=hlat))
    s.pop(f"FSS thr={ms.FSS_THR:g} m n={ms.FSS_N}", None)
    return s


def main():
    rows = []
    for ds in ("MESA", "FOSI"):
        win = FS[ds]["windows"][0]
        eng = spm.run_dir(FS[ds]["eng"], win)
        f = np.load(os.path.join(eng, "eval_data", "fields.npz"))
        truth = f["Y_test_phys"][:, 0].astype(np.float32)
        ocean = f["mask_test"][0, 0] <= 0.5
        hlat = torch.from_numpy(f["hlat"][:truth.shape[1]] if f["hlat"].shape[0] != truth.shape[1] else f["hlat"])
        geo = pickle.load(open(os.path.join(eng, "eval_data", "tile_geometry.pkl"), "rb"))[0]
        times = pd.to_datetime(pd.read_csv(os.path.join(eng, "eval_data", "sample_times.csv"))["time"])
        clim_t = ms.climatology(ds, win, np.asarray(geo["target_lat"]), np.asarray(geo["target_lon"]), ocean.shape)[spm.noleap_doy(times)]
        preds = {"Bilinear": (f["Y_base_phys"][:, 0] * ocean).astype(np.float32),
                 "EngressNet": f["preds_all_phys"][:, :, 0].astype(np.float32).mean(1)}
        det_dir = spm.run_dir(FS[ds]["det"], win)
        seed_runs = [det_dir] + sorted(glob.glob(os.path.join(V6, "results", f"FINAL2_{ds}_detmse_seedens", "*_2021_*")))
        S = []
        for d in seed_runs:
            x = np.load(os.path.join(d, "eval_data", "fields.npz"))
            assert np.allclose(x["Y_test_phys"][:, 0], truth, atol=1e-5), f"{d}: test set differs"
            S.append(x["Y_pred_det_phys"][:, 0].astype(np.float32))
        preds["Deterministic U-Net"] = S[0]
        preds["Seed/window ensemble (det.)"] = np.mean(S, axis=0)
        assert len(S) == 10, len(S)
        for m, p in preds.items():
            rows.append({"dataset": "MESACLIP" if ds == "MESA" else "FOSI", "method": m, **scores(p, truth, ocean, hlat, clim_t)})
            print(ds, m, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
