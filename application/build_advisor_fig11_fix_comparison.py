"""
build_advisor_fig11_fix_comparison.py

Advisor figure 11: summarizes all four fix attempts against the land-bleed
-fixed baseline, coastal-band RMSE vs. independent CryoSat-2 truth. All
numbers below are read directly from each test's own
network_vs_bilinear_coastal_bias.csv (land-fixed data throughout), averaged
over the same 7 months (Jan-Apr, Oct-Dec) used everywhere else in this
analysis.

The 5th bar (conservative regrid) is a data-pipeline fix, not a retrain --
PIOMAS's input was originally only ever bilinear-regridded onto the training
grid, unlike FOSI/MESACLIP's own "avg" (conservative) training convention.
Rebuilding it with a true conservative regridder (built from PSC's
grid.dat.pop U-point corners) turned out to beat every training-based fix
tried. See piomas_application.md Section 8 for the reasoning and Figure 14
for a visual (native vs. regridded PIOMAS vs. FOSI/MESACLIP texture).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/11_fix_attempts_comparison.png"

configs = ["Original\n(land-fixed,\nbilinear input)", "Bias-corrected\ninput",
           "coastal_boost=1.0\nretrain", "Domain-\nrandomization",
           "Conservative\nregrid (input)", "Domain-rand. +\nconservative regrid"]
network_rmse = [0.9695, 0.7057, 0.9430, 0.9370, 0.8978, 0.8855]
bilinear_rmse = [0.8296, 0.4598, 0.8296, 0.8296, 0.8086, 0.8086]
ratios = [n / b for n, b in zip(network_rmse, bilinear_rmse)]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(configs))
w = 0.35
colors_net = ["#eb6834"] * (len(configs) - 2) + ["#2ca02c", "#1a9e46"]
ax.bar(x - w / 2, network_rmse, width=w, color=colors_net, label="Network")
ax.bar(x + w / 2, bilinear_rmse, width=w, color="#2a78d6", label="Bilinear")
for i, r in enumerate(ratios):
    ax.annotate(f"{r:.2f}x", xy=(i, max(network_rmse[i], bilinear_rmse[i]) + 0.03),
                ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("Coastal RMSE vs. CryoSat-2 (m)")
ax.set_title("Five fix attempts, coastal band (land-bleed-fixed data)\n"
             "Label = network/bilinear ratio -- combining domain-randomization with the "
             "conservative-regridded input (dark green) gives the best RMSE ratio yet, "
             "though its bias-comparison is worse than the conservative regrid alone (see text)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print("Saved:", OUT_PATH)
