"""
Final round (2026-11-16): cut the PIOMAS application output of the final model to the ICESat-2 period used in Table 5 / Fig. 9
(first and last QC'd in-domain pass-cell day: 2018-10-15 to 2022-04-30). No re-run: each saved day is kept unchanged, the
other days are dropped, so every value scored against ICESat-2 stays identical.

Batches: results/PIOMAS_2018-2022_FINAL2_FOSI (EngressNet, K = 20) and results/PIOMAS_2018-2022_FINAL2_FOSI_detmse.
Per run, in eval_data/: fields.npz (time-indexed arrays), sample_times.csv (sample_idx renumbered), candidate_point_timeseries.csv;
in the run folder: sit_timeseries.csv. The originals are copied to BACKUP first; eval_data/trimmed_to_is2_window.json marks a
trimmed run (skipped on a second call). metrics.csv and the run's PNGs still describe the full 2018-2022 run (noted in
description.txt). The batch folder names keep "2018-2022" because scripts refer to them.

Usage: python evaluation/revisions/trim_piomas_to_is2_window.py
"""
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
BATCHES = ["PIOMAS_2018-2022_FINAL2_FOSI", "PIOMAS_2018-2022_FINAL2_FOSI_detmse"]
START, END = pd.Timestamp("2018-10-15"), pd.Timestamp("2022-04-30")
BACKUP = "/glade/derecho/scratch/skygale/downscaling/outputs/backups/piomas_pre_is2_trim"
TIME_KEYS = ("X_test_sit_phys", "Y_base_phys", "Y_pred_det_phys", "preds_all_phys", "Y_pred_phys", "Y_test_phys", "mask_test",
             "test_tile_ids")

for b in BATCHES:
    for rd in sorted(glob.glob(f"{V6}/results/{b}/*/")):
        rd = rd.rstrip("/")
        ed = f"{rd}/eval_data"
        marker = f"{ed}/trimmed_to_is2_window.json"
        if os.path.exists(marker):
            print("skip (already trimmed):", rd)
            continue
        st = pd.read_csv(f"{ed}/sample_times.csv")
        t = pd.to_datetime(st["time"])
        keep = ((t >= START) & (t <= END)).to_numpy()
        n0 = len(t)
        bdir = f"{BACKUP}/{b}/{os.path.basename(rd)}"
        os.makedirs(bdir, exist_ok=True)
        for src in (f"{ed}/fields.npz", f"{ed}/sample_times.csv", f"{ed}/candidate_point_timeseries.csv", f"{rd}/sit_timeseries.csv",
                    f"{rd}/description.txt"):
            dst = f"{bdir}/{os.path.basename(src)}"
            if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
                shutil.copy2(src, dst)
        z = np.load(f"{ed}/fields.npz")
        d = {}
        for k in z.files:
            a = z[k]
            d[k] = a[keep] if k in TIME_KEYS else a
            if k in TIME_KEYS:
                assert a.shape[0] == n0, (k, a.shape)
        np.savez_compressed(f"{ed}/fields.tmp.npz", **d)
        os.replace(f"{ed}/fields.tmp.npz", f"{ed}/fields.npz")
        st2 = st[keep].reset_index(drop=True)
        st2["sample_idx"] = np.arange(len(st2))
        st2.to_csv(f"{ed}/sample_times.csv", index=False)
        for csv in (f"{ed}/candidate_point_timeseries.csv", f"{rd}/sit_timeseries.csv"):
            df = pd.read_csv(csv)
            tt = pd.to_datetime(df["time"])
            df[(tt >= START) & (tt <= END)].to_csv(csv, index=False)
        with open(f"{rd}/description.txt", "a") as f:
            f.write(f"\nTrimmed 2026-11-16 to the ICESat-2 period {START.date()} to {END.date()} ({keep.sum()} of {n0} days kept, values "
                    "unchanged) by evaluation/revisions/trim_piomas_to_is2_window.py; metrics.csv and the PNGs still describe the "
                    f"full run. Originals: {bdir}\n")
        json.dump({"start": str(START.date()), "end": str(END.date()), "days_before": n0, "days_after": int(keep.sum()),
                   "backup": bdir}, open(marker, "w"), indent=1)
        print(f"{b}/{os.path.basename(rd)}: kept {keep.sum()} of {n0} days ({st2.time.iloc[0]} .. {st2.time.iloc[-1]})", flush=True)
