"""
build_advisor_fig12_recalibration.py

Advisor figure 12: the one fully successful fix. A single spread-scaling
factor (fit on Jan-Feb 2020, s=100x) brings ensemble coverage from ~2-3% to
~90% on completely held-out Mar-Apr 2020 -- confirmed out-of-sample, not
just fit to the calibration data. Numbers from
results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs/
cryosat2_validation/test1_recalibration_scale_factors.csv.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/12_recalibration_success.png"

regions = ["Coastal", "Interior"]
raw_coverage = [3.47, 2.42]
scaled_coverage = [89.30, 90.55]

fig, ax = plt.subplots(figsize=(7, 6))
x = np.arange(len(regions))
w = 0.35
ax.bar(x - w / 2, raw_coverage, width=w, color="#c0392b", label="Raw ensemble (unscaled)")
ax.bar(x + w / 2, scaled_coverage, width=w, color="#27ae60", label="Rescaled (s=100x, fit on Jan-Feb)")
ax.axhline(90, color="#333", linestyle="--", linewidth=1.2, label="Ideal (90%)")
ax.set_xticks(x)
ax.set_xticklabels(regions)
ax.set_ylabel("% of cells where CryoSat-2 truth falls\ninside the 5-95% ensemble band")
ax.set_ylim(0, 100)
ax.set_title("Post-hoc uncertainty recalibration -- held-out validation (Mar-Apr 2020)\n"
             "Scale factor fit on Jan-Feb, evaluated on genuinely unseen months")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print("Saved:", OUT_PATH)
