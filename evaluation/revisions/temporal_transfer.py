"""
Revisions 2026-11, G: 2021 test skill of each training window vs. the gap between window end and test year, from the
existing final runs (seed 0, K = 20, coarse-zero rule, original truth; files written 2026-10-24 in
evaluation/corrected_calibration/). EngressNet and det-MSE, both datasets.
Gap = 2021 - last training year (2000-2005 -> 16, 2005-2010 -> 11, 2010-2015 -> 6, 2015-2020 -> 1).
If the G extrapolation runs exist (results/REV_G_MESA_extrap*/), they are added as a separate block.
Writes results/revisions_2026-11/G_temporal_transfer.{csv,png}.
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
# method colours: same palette as figures/figures_v2/make_v2_figures.py (Setup cell)
METHOD_COLORS = {"EngressNet": "#2a78d6", "Deterministic U-Net": "#eb6834", "Bilinear": "#1baf7a"}

C = f"{V6}/evaluation/corrected_calibration"
OUT = f"{V6}/results/revisions_2026-11"
rows = []
for ds, eng, det in (("MESACLIP", "MESA_final_ema", "MESA_detmse_ema"), ("FOSI", "FOSI_final_ema", "FOSI_detmse_ema")):
    cal = pd.read_csv(f"{C}/{eng}.csv")
    cal = cal[cal.run != "POOLED"]
    om_e = pd.read_csv(f"{C}/{eng}_ocean_metrics.csv")
    om_d = pd.read_csv(f"{C}/{det}_ocean_metrics.csv")
    for _, r in cal.iterrows():
        w = re.search(r"(\d{4})-(\d{4})", r["run"]).group(0)
        end = int(w[-4:])
        e = om_e[(om_e.run == r["run"]) & (om_e.method == "EngressNet ensemble mean")].iloc[0]
        d = om_d[om_d.run.str.contains(w) & om_d.method.str.startswith("Deterministic")].iloc[0]
        rows.append({"dataset": ds, "window": w, "gap (yr)": 2021 - end, "method": "EngressNet", "RMSE": e["RMSE"],
                     "CRPS": r["CRPS (ocean)"], "spread-skill": r["Spread-skill ratio"], "coastal RMSE": e["Coastal RMSE"],
                     "IIEE": e["IIEE"]})
        rows.append({"dataset": ds, "window": w, "gap (yr)": 2021 - end, "method": "Deterministic U-Net", "RMSE": d["RMSE"],
                     "CRPS": d["MAE"], "spread-skill": np.nan, "coastal RMSE": d["Coastal RMSE"], "IIEE": d["IIEE"]})
df = pd.DataFrame(rows).sort_values(["dataset", "method", "gap (yr)"])
os.makedirs(OUT, exist_ok=True)
df.round(4).to_csv(f"{OUT}/G_temporal_transfer.csv", index=False)

fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, constrained_layout=True)
for i, ds in enumerate(("MESACLIP", "FOSI")):
    for j, m in enumerate(("RMSE", "CRPS", "coastal RMSE", "spread-skill")):
        ax = axs[i, j]
        for meth in ("EngressNet", "Deterministic U-Net"):
            s = df[(df.dataset == ds) & (df.method == meth)]
            if s[m].notna().any():
                ax.plot(s["gap (yr)"], s[m], "o-", color=METHOD_COLORS[meth], lw=2, ms=6, label=meth)
        if m == "spread-skill":
            ax.axhline(1.0, color="#8a8984", ls="--", lw=1)
        ax.set_title(f"{ds}: {m}" + (" (det: MAE)" if m == "CRPS" else ""), fontsize=11)
        ax.grid(alpha=0.2)
        if i == 1:
            ax.set_xlabel("Gap between training-window end and 2021 (years)")
axs[0, 0].legend(fontsize=9, frameon=False)
fig.savefig(f"{OUT}/G_temporal_transfer.png", dpi=200)
pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))
for ds in ("MESACLIP", "FOSI"):
    for meth in ("EngressNet", "Deterministic U-Net"):
        s = df[(df.dataset == ds) & (df.method == meth)]
        sl = np.polyfit(s["gap (yr)"], s["RMSE"], 1)[0]
        print(f"{ds} {meth}: RMSE slope {1000 * sl:+.2f} mm per year of gap; range {s.RMSE.min():.4f}-{s.RMSE.max():.4f}")
