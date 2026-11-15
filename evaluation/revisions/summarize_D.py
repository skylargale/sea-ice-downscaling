"""
Revisions 2026-11, D summary: 4-window mean (and min-max over windows) of every metric in
results/revisions_2026-11/D_metrics/<DS>_<window>.csv, per method. Writes D_metrics/<DS>_4window.csv and prints it.
Usage: python summarize_D.py [MESA FOSI]
"""
import glob
import os
import sys

import pandas as pd

D = "/glade/work/skygale/projects/SeaIceDownscaling/results/revisions_2026-11/D_metrics" + ("" if os.environ.get("FINAL_SET", "final2") == "final1" else "_" + os.environ.get("FINAL_SET", "final2"))
for ds in (sys.argv[1:] or ["MESA", "FOSI"]):
    files = sorted(f for f in glob.glob(f"{D}/{ds}_[0-9]*-[0-9]*.csv") if not f.endswith(("_psd.csv",)) and "_n" not in os.path.basename(f))
    files = [f for f in files if os.path.basename(f).count("_") == 1]
    if not files:
        print("no files for", ds); continue
    tabs = [pd.read_csv(f, index_col=0).apply(pd.to_numeric, errors="coerce") for f in files]
    allw = pd.concat(tabs, keys=[os.path.basename(f)[len(ds) + 1:-4] for f in files])
    mean = allw.groupby(level=1, sort=False).mean()
    lo, hi = allw.groupby(level=1, sort=False).min(), allw.groupby(level=1, sort=False).max()
    out = mean.copy()
    for c in mean.columns:
        out[f"{c} [min-max]"] = lo[c].round(4).astype(str) + "-" + hi[c].round(4).astype(str)
    out.to_csv(f"{D}/{ds}_4window.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(f"===== {ds}: mean over {len(files)} windows ({', '.join(k for k in allw.index.levels[0])})")
    print(mean.round(4).to_string())
