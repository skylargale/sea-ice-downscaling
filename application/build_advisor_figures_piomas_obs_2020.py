"""
build_advisor_figures_piomas_obs_2020.py

Assembles a curated, presentation-ready set of figures in
results/PIOMAS_obs_2020/ summarizing the full Paragraph 7 (observational
application) investigation: the uncertainty map, the CryoSat-2 independent
validation, the coarsening check (rules out "just a resolution mismatch"),
the root-cause PIOMAS-vs-CryoSat bias map, and the new network-vs-bilinear
coastal comparison. Two figures (coarsening trend, network-vs-bilinear
coastal bars) are built fresh here since they only existed as CSVs before;
the other four are copied from their run-specific subdirectories with
clear numbered names so the whole story reads in order.
"""

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 13, "legend.fontsize": 11.5, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})

PALETTE = {"network": "#eb6834", "bilinear": "#2a78d6", "coastal": "#c0392b", "interior": "#2a78d6"}

BASE = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020"
RUN_DIR = f"{BASE}/FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs"
CRYOSAT_DIR = f"{RUN_DIR}/cryosat2_validation"
OSD_DIR = f"{RUN_DIR}/observing_system_design"
RAW_BIAS_DIR = f"{BASE}/piomas_vs_cryosat_raw_v2_landfixed"

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 10: "Oct", 11: "Nov", 12: "Dec"}

# ---------------------------------------------------------------------------
# 1. Copy the four already-built figures with clear numbered names
# ---------------------------------------------------------------------------

copies = [
    (f"{OSD_DIR}/uncertainty_map_2020.png", f"{BASE}/01_ensemble_uncertainty_map_2020.png"),
    (f"{CRYOSAT_DIR}/cryosat2_vs_network_202012.png", f"{BASE}/02_cryosat2_vs_network_dec2020.png"),
    (f"{CRYOSAT_DIR}/cryosat2_scatter_coastal_vs_interior.png", f"{BASE}/03_cryosat2_scatter_coastal_vs_interior.png"),
    (f"{RAW_BIAS_DIR}/piomas_minus_cryosat_bias_map.png", f"{BASE}/05_piomas_minus_cryosat_bias_map.png"),
]
for src, dst in copies:
    if os.path.exists(src):
        shutil.copy(src, dst)
        print("Copied:", dst)
    else:
        print("MISSING (skipped):", src)

# ---------------------------------------------------------------------------
# 2. New figure: coarsening check -- correlation vs. block size
# ---------------------------------------------------------------------------

coarsen_df = pd.read_csv(f"{CRYOSAT_DIR}/cryosat2_coarsening_check.csv")
per_month = coarsen_df[coarsen_df["month"] != "pooled_Jan-Apr"].copy()
per_month["month"] = per_month["month"].astype(int)
mean_within_month = per_month.groupby(["region", "block_size_deg"])["network_corr_vs_cryosat"].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 5.5))
for region, color in [("coastal", PALETTE["coastal"]), ("interior", PALETTE["interior"])]:
    sub = mean_within_month[mean_within_month["region"] == region].sort_values("block_size_deg")
    ax.plot(sub["block_size_deg"], sub["network_corr_vs_cryosat"], marker="o", linewidth=2.2,
            color=color, label=region.capitalize())
ax.axhline(0, color="#898781", linewidth=1, linestyle="--", zorder=0)
ax.axvline(0.6, color="#c9c8c2", linewidth=8, alpha=0.4, zorder=0, label="≈ CryoSat-2 footprint")
ax.set_xlabel("Block size (degrees) — coarsening the comparison")
ax.set_ylabel("Mean within-month correlation\n(network vs. CryoSat-2)")
ax.set_title("Coarsening doesn't rescue correlation\n(rules out \"just a resolution mismatch\")")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(f"{BASE}/04_coarsening_check.png")
print("Saved: 04_coarsening_check.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. New figure: network vs. bilinear RMSE against CryoSat-2, by region/month
# ---------------------------------------------------------------------------

nb_df = pd.read_csv(f"{RUN_DIR}/cryosat2_validation/network_vs_bilinear_coastal_bias.csv")
months_order = [1, 2, 3, 4, 10, 11, 12]
nb_df["month"] = pd.Categorical(nb_df["month"], categories=months_order, ordered=True)
nb_df = nb_df.sort_values("month")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, region in zip(axes, ["coastal", "interior"]):
    sub = nb_df[nb_df["region"] == region]
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w / 2, sub["network_rmse_vs_cryosat"], width=w, color=PALETTE["network"], label="Network")
    ax.bar(x + w / 2, sub["bilinear_rmse_vs_cryosat"], width=w, color=PALETTE["bilinear"], label="Bilinear")
    ax.set_xticks(x)
    ax.set_xticklabels([MONTH_NAMES[m] for m in sub["month"]])
    ax.set_title(f"{region.capitalize()} cells")
    ax.set_ylabel("RMSE vs. CryoSat-2 (m)")
    ax.legend()
fig.suptitle("Network vs. bilinear accuracy against independent CryoSat-2 truth\n"
             "(network makes the coastal PIOMAS bias worse, 6/7 months)", y=1.04)
fig.tight_layout()
fig.savefig(f"{BASE}/06_network_vs_bilinear_coastal_rmse.png", bbox_inches="tight")
print("Saved: 06_network_vs_bilinear_coastal_rmse.png")
plt.close(fig)

print("\nAll advisor figures assembled in:", BASE)
for f in sorted(os.listdir(BASE)):
    if f.endswith(".png"):
        print(" -", f)
