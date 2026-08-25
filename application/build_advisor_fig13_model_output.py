"""
build_advisor_fig13_model_output.py

Advisor figure 13: the actual Paragraph 7 deliverable, on its own -- the
network's downscaled, daily, probabilistic coastal SIT product for 2020,
with no CryoSat/bilinear comparison or diagnostic framing. Every other
figure in this set is a validation/diagnostic; this is just "here is the
product."

Top row: ensemble-mean SIT maps for four representative months spanning the
seasonal cycle (Jan, Apr, Jul, Oct) -- Jul has no CryoSat-2 coverage at all,
included deliberately to show the product covers the full year even where
no independent check is possible.
Bottom: daily time series with the K=20 ensemble spread shown as a shaded
band, for two candidate coastal communities (Kivalina, Point Hope) across
all of 2020 -- the "daily, probabilistic" part of the deliverable made
concrete.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_engressnet as fe

RUN_DIR = ("/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/"
           "FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs")
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
OUT_PATH = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/13_model_output_2020.png"

fields = np.load(os.path.join(EVAL_DIR, "fields.npz"))
Y_pred_phys = fields["Y_pred_phys"][:, 0]        # (365, H, W) ensemble mean
preds_all_phys = fields["preds_all_phys"][:, :, 0]  # (365, K, H, W)
mask_test = fields["mask_test"]

with open(os.path.join(EVAL_DIR, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
target_lat = np.asarray(tile_geometry[0]["target_lat"])
target_lon = np.asarray(tile_geometry[0]["target_lon"])

sample_times = pd.read_csv(os.path.join(EVAL_DIR, "sample_times.csv"))
sample_times["time"] = pd.to_datetime(sample_times["time"])
sample_month = sample_times["time"].dt.month.values

ocean_hw = mask_test[0, 0] <= 0.5

point_df = pd.read_csv(os.path.join(EVAL_DIR, "candidate_point_timeseries.csv"))

# ---------------------------------------------------------------------------
# Top row: seasonal snapshots (monthly mean of the ensemble mean)
# ---------------------------------------------------------------------------

MONTHS = [1, 4, 7, 10]
MONTH_NAMES = {1: "January", 4: "April", 7: "July", 10: "October"}
plot_bbox = {"lon_min": -182, "lon_max": -151, "lat_min": 60, "lat_max": 75}
proj, boundary_path, central_lon = fe.make_polar_proj(plot_bbox)
lon2d, lat2d = np.meshgrid(target_lon, target_lat)

fig = plt.figure(figsize=(18, 13))
for i, month in enumerate(MONTHS):
    ax = fig.add_subplot(2, 4, i + 1, projection=proj)
    fe.style_polar_ax(ax, proj, boundary_path, plot_bbox)
    day_idx = np.where(sample_month == month)[0]
    month_mean = np.where(ocean_hw, Y_pred_phys[day_idx].mean(axis=0), np.nan)
    pc = ax.pcolormesh(lon2d, lat2d, month_mean, transform=ccrs.PlateCarree(),
                        cmap="viridis", vmin=0, vmax=2.5, shading="auto")
    no_cryosat = " (no CryoSat-2 this month)" if month == 7 else ""
    ax.set_title(f"{MONTH_NAMES[month]} 2020{no_cryosat}", fontsize=12)
cax = fig.add_axes([0.25, 0.53, 0.5, 0.018])
fig.colorbar(pc, cax=cax, orientation="horizontal", label="Ensemble-mean SIT (m)")

# ---------------------------------------------------------------------------
# Bottom row: daily time series + ensemble spread for two communities
# ---------------------------------------------------------------------------

for j, point_name in enumerate(["Kivalina", "Point Hope"]):
    ax = fig.add_subplot(2, 2, 3 + j)
    row = point_df[(point_df.point == point_name) & (point_df.method == "stochastic_unet_mean")]
    iy, ix = int(row["grid_iy"].iloc[0]), int(row["grid_ix"].iloc[0])
    daily_mean = Y_pred_phys[:, iy, ix]
    daily_spread = preds_all_phys[:, :, iy, ix].std(axis=1)
    dates = sample_times["time"].values
    ax.fill_between(dates, daily_mean - daily_spread, daily_mean + daily_spread,
                     alpha=0.3, color="#eb6834", label="±1 ensemble std")
    ax.plot(dates, daily_mean, color="#eb6834", linewidth=1.2, label="Ensemble mean")
    ax.set_title(f"{point_name}: daily downscaled SIT, 2020", fontsize=12)
    ax.set_ylabel("SIT (m)")
    ax.legend(loc="upper left", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Paragraph 7 deliverable: downscaled, daily, probabilistic coastal SIT, 2020\n"
             "(PIOMAS-driven, FOSI-trained network -- no CryoSat-2/bilinear comparison, just the product)",
             y=0.98, fontsize=14)
fig.subplots_adjust(hspace=0.45, wspace=0.3)
fig.savefig(OUT_PATH, dpi=170, bbox_inches="tight")
print("Saved:", OUT_PATH)
