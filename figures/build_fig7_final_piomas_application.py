"""
Paper Figure 7 (observational application), 2026-10-08: the previous all_figures_by_run.ipynb Figure 7 cell, unchanged in layout,
colours and labels, now driven by the final EngressNet (T2 amplitude maps + 10 px energy tiles + noise lr x10; window 2015-2020,
PIOMAS 2021, results/PIOMAS_2010-2023_cdrv6_finalFOSI_4win) instead of PIOMAS_2021avg_scratchFOSI_4win. The only addition: the
month's ICESat-2 tracks (QC'd 0.1 deg pass-cell means, same colour scale) overlaid on the maps. ICESat-2 has no summer data, so the
July map has no tracks. Point time series are unchanged (too few ICESat-2 passes near the coast to add).
Output: figures/final_figures/07_piomas_application.png.
"""
import glob
import os
import pickle
import sys

import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nicopal as ncp
import numpy as np
import pandas as pd

BASE = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(BASE, "training"))
sys.path.insert(0, os.path.join(BASE, "evaluation"))
import run_daily_eval_batch as rdeb  # noqa: E402
import functions_engressnet as fe  # noqa: E402

IS2C = "/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells"
YEAR = 2021

piomas_app_matches = [
    d for d in sorted(glob.glob(os.path.join(BASE, "results", "PIOMAS_2010-2023_cdrv6_finalFOSI_4win", f"*_2015-2020_{YEAR}_*")))
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "eval_data", "fields.npz"))
]
assert len(piomas_app_matches) == 1, piomas_app_matches
RUN_DIR_FIG8 = piomas_app_matches[0]
EVAL_DIR_FIG8 = os.path.join(RUN_DIR_FIG8, "eval_data")

fields8 = np.load(os.path.join(EVAL_DIR_FIG8, "fields.npz"))
Y_pred_phys8 = fields8["Y_pred_phys"][:, 0]
preds_all_phys8 = fields8["preds_all_phys"][:, :, 0]
mask_test8 = fields8["mask_test"]
with open(os.path.join(EVAL_DIR_FIG8, "tile_geometry.pkl"), "rb") as f:
    tile_geometry8 = pickle.load(f)
target_lat8 = np.asarray(tile_geometry8[0]["target_lat"])
target_lon8 = np.asarray(tile_geometry8[0]["target_lon"])
sample_times8 = pd.read_csv(os.path.join(EVAL_DIR_FIG8, "sample_times.csv"))
sample_times8["time"] = pd.to_datetime(sample_times8["time"])
sample_month8 = sample_times8["time"].dt.month.values
ocean_hw8 = mask_test8[0, 0] <= 0.5
point_df8 = pd.read_csv(os.path.join(EVAL_DIR_FIG8, "candidate_point_timeseries.csv"))

# ICESat-2 pass-cells (same QC as the evaluation notebooks: extent >= 2 km, >= 20 segments, median segment uncertainty <= 1 m)
is2 = pd.read_parquet(f"{IS2C}/pass_cells.parquet").query("in_domain")
is2 = is2[(is2.extent_km >= 2.0) & (is2.n_seg >= 20) & (is2.unc_med <= 1.0) & (is2.date.dt.year == YEAR)]

MONTHS = [1, 4, 7, 10]
MONTH_NAMES = {1: "January", 4: "April", 7: "July", 10: "October"}
plot_bbox = {"lon_min": -182, "lon_max": -151, "lat_min": 60, "lat_max": 75}
proj8, boundary_path8, central_lon8 = fe.make_polar_proj(plot_bbox)
lon2d, lat2d = np.meshgrid(target_lon8, target_lat8)

fig = plt.figure(figsize=(16, 12))
axes = []
for i, month in enumerate(MONTHS):
    ax = fig.add_subplot(2, 4, i + 1, projection=proj8)
    axes.append(ax)
    fe.style_polar_ax(ax, proj8, boundary_path8, plot_bbox)

    point_hope_lon = ((rdeb.POINT_HOPE_LON_360 + 180) % 360) - 180
    ax.plot(point_hope_lon, rdeb.POINT_HOPE_LAT, marker="*", color="red", markersize=10,
            transform=ccrs.PlateCarree())
    ax.text(point_hope_lon + 1, rdeb.POINT_HOPE_LAT + 0.35, "Point Hope", color="red", fontsize=7,
            transform=ccrs.PlateCarree())

    day_idx = np.where(sample_month8 == month)[0]
    month_mean = np.where(ocean_hw8, Y_pred_phys8[day_idx].mean(axis=0), np.nan)
    pc = ax.pcolormesh(lon2d, lat2d, month_mean, transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=2.5, shading="auto")
    # new: this month's ICESat-2 tracks, coloured on the same scale (thin dark outline so light ice stays visible)
    m = is2[is2.date.dt.month == month]
    if len(m):
        ax.scatter(((m.lon.values + 180) % 360) - 180, m.lat.values, c=m.h_mean.values, cmap="Blues", vmin=0, vmax=2.5,
                   s=2.5, linewidths=0.15, edgecolors="#3a3a3a", transform=ccrs.PlateCarree(), zorder=5, rasterized=True)
    ax.set_title(f"{MONTH_NAMES[month]} {YEAR}", fontsize=14)

cbar = fig.colorbar(pc, ax=axes, orientation="horizontal", aspect=30, shrink=0.9, pad=0.02)
cbar.set_label("Ensemble Mean SIT (m)", fontsize=16)
cbar.ax.tick_params(labelsize=14)

sample = ncp.pal_sample("Sodium", 8)

for j, point_name in enumerate(["Kivalina", "Point Hope"]):
    ax = fig.add_subplot(2, 2, 3 + j)
    row = point_df8[(point_df8.point == point_name) & (point_df8.method == "stochastic_unet_mean")]
    iy, ix = int(row["grid_iy"].iloc[0]), int(row["grid_ix"].iloc[0])
    daily_mean = Y_pred_phys8[:, iy, ix]
    daily_spread = preds_all_phys8[:, :, iy, ix].std(axis=1)
    dates = sample_times8["time"].values
    ax.fill_between(dates, daily_mean - daily_spread, daily_mean + daily_spread, alpha=0.3, color=sample[1], label="±1 Ensemble Std")
    ax.plot(dates, daily_mean, color=sample[0], linewidth=2, label="Ensemble Mean")
    ax.set_title(f"{point_name} Daily Downscaled PIOMAS SIT ({YEAR})", fontsize=16)
    ax.set_ylabel("SIT (m)", fontsize=14)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=14)
    ax.tick_params(axis="x", rotation=30)

out = os.path.join(BASE, "figures", "final_figures", "07_piomas_application.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out, "| run", os.path.basename(RUN_DIR_FIG8), "| ICESat-2 cells per month",
      {MONTH_NAMES[m]: int((is2.date.dt.month == m).sum()) for m in MONTHS})
