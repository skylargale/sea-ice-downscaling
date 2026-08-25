"""
observing_cryosat2_validation_2020.py

Real independent-truth validation of Paragraph 7's PIOMAS-driven downscaled
SIT product, using CryoSat-2 RDEFT4 (NASA GSFC) -- a genuine satellite
altimetry retrieval, not a model/assimilation product like PIOMAS -- as an
actual accuracy check instead of the earlier PIOMAS-vs-PIOMAS
self-consistency diagnostic (see SELF_CONSISTENCY_NOTE.txt in the
PIOMAS_obs_2020 run directories).

Design choice, and why this is a post-hoc comparison rather than a fresh
`train_engressnet.py --test-y-path=...` pipeline run: RDEFT4 only has
winter-month coverage (no CryoSat retrieval through the summer melt
season -- 2020 has real data for Jan-Apr, Oct-Dec only, 7 of 12 months).
run_pipeline()'s --months flag applies the SAME month filter to both the
train split (used only to reproduce the checkpoint's original
normalization statistics, via x_path/y_path/train_years) and the test
split -- restricting it to winter months would recompute X_mean/X_std/
Y_mean/Y_std from a winter-only slice of FOSI 2015-2019, which would NOT
match the actual checkpoint's training-time normalization (computed from
all 12 months) and could subtly bias the comparison. Since predictions
don't depend on Y at all (this is pure --num-epochs 0 inference), the
already-completed true-daily run's predictions
(results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_daily_*/eval_data/)
are reused as-is -- byte-identical to what a fresh run would produce --
and only the *comparison* against CryoSat-2 truth is new here.

Method: regrid each of the 7 available 2020 RDEFT4 monthly sea_ice_thickness
fields (masking the -9999 fill value) directly onto the same cropped
0.1-degree medium-domain grid (150x310, target_lat/target_lon from
tile_geometry.pkl) the network's own predictions already live on -- no
extra crop-index bookkeeping needed. Compare each month's CryoSat field
against that same month's *monthly-mean* of the network's daily
predictions (and the bilinear baseline), at matching cadence.
"""

import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_engressnet as fe

YEAR = 2020
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

RUN_DIR = ("/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/"
           "FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs")
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
OUT_DIR = os.path.join(RUN_DIR, "cryosat2_validation")
os.makedirs(OUT_DIR, exist_ok=True)
print("Using run:", RUN_DIR)

# ---------------------------------------------------------------------------
# Load the network's already-completed daily predictions
# ---------------------------------------------------------------------------

fields = np.load(os.path.join(EVAL_DIR, "fields.npz"))
Y_pred_phys = fields["Y_pred_phys"][:, 0]     # (365, H, W) -- stochastic ensemble mean
Y_base_phys = fields["Y_base_phys"][:, 0]     # (365, H, W) -- bilinear baseline
mask_test = fields["mask_test"][:, 0]         # (365, H, W) -- 1=land, 0=ocean (static)

with open(os.path.join(EVAL_DIR, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
target_lat = np.asarray(tile_geometry[0]["target_lat"])   # (150,)
target_lon = np.asarray(tile_geometry[0]["target_lon"])   # (310,)

sample_times = pd.read_csv(os.path.join(EVAL_DIR, "sample_times.csv"))
sample_times["time"] = pd.to_datetime(sample_times["time"])
sample_month = sample_times["time"].dt.month.values
print(f"Network predictions: {Y_pred_phys.shape[0]} days, grid {Y_pred_phys.shape[1]}x{Y_pred_phys.shape[2]}")

ocean_hw = mask_test[0] <= 0.5

# ---------------------------------------------------------------------------
# Regrid each available CryoSat-2 month onto the exact same grid
# ---------------------------------------------------------------------------

dst_grid = xr.Dataset({"lat": ("lat", target_lat), "lon": ("lon", target_lon)})

cryosat_months = sorted(int(f.split("_")[1][4:6]) for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}"))
print("CryoSat-2 months available for", YEAR, ":", cryosat_months)

regridder = None
rows = []
month_maps = {}

for month in cryosat_months:
    fpath = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")][0]
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, fpath))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)  # mask -9999 fill
    src_grid = xr.Dataset({
        "lat": (["y", "x"], ds["lat"].values),
        "lon": (["y", "x"], ds["lon"].values % 360),
    })
    sit_da = sit.assign_coords(lat=(["y", "x"], src_grid["lat"].values), lon=(["y", "x"], src_grid["lon"].values))

    if regridder is None:
        regridder = xe.Regridder(
            src_grid, dst_grid, method="bilinear", periodic=False,
            filename=f"{WEIGHTED_GRIDS_DIR}/cryosat2_rdeft4_to_medium_domain.nc", reuse_weights=False,
        )
        print(" >> Built regridder_cryosat_to_medium_domain.")

    cryo_regridded = regridder(sit_da, skipna=True).values   # (150, 310), NaN where no data/land
    month_maps[month] = cryo_regridded

    day_idx = np.where(sample_month == month)[0]
    net_month_mean = Y_pred_phys[day_idx].mean(axis=0)
    bilin_month_mean = Y_base_phys[day_idx].mean(axis=0)

    valid = ocean_hw & ~np.isnan(cryo_regridded)
    n_valid = int(valid.sum())
    if n_valid == 0:
        print(f"Month {month}: no valid overlapping CryoSat cells in-domain, skipping.")
        continue

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))

    def _bias(a, b):
        return float(np.mean(a[valid] - b[valid]))

    def _corr(a, b):
        return float(np.corrcoef(a[valid], b[valid])[0, 1])

    rows.append({
        "month": month, "n_valid_cells": n_valid,
        "cryosat_mean_m": float(np.nanmean(cryo_regridded[valid])),
        "network_mean_m": float(net_month_mean[valid].mean()),
        "bilinear_mean_m": float(bilin_month_mean[valid].mean()),
        "network_rmse_vs_cryosat": _rmse(net_month_mean, cryo_regridded),
        "bilinear_rmse_vs_cryosat": _rmse(bilin_month_mean, cryo_regridded),
        "network_bias_vs_cryosat": _bias(net_month_mean, cryo_regridded),
        "bilinear_bias_vs_cryosat": _bias(bilin_month_mean, cryo_regridded),
        "network_corr_vs_cryosat": _corr(net_month_mean, cryo_regridded),
        "bilinear_corr_vs_cryosat": _corr(bilin_month_mean, cryo_regridded),
    })
    ds.close()

result_df = pd.DataFrame(rows)
result_df.to_csv(os.path.join(OUT_DIR, "cryosat2_independent_validation_2020.csv"), index=False)
print("\n=== Independent-truth validation vs. CryoSat-2 RDEFT4, 2020 (winter months only) ===")
print(result_df.to_string(index=False))

overall = {
    "network_rmse_vs_cryosat_mean_over_months": float(result_df["network_rmse_vs_cryosat"].mean()),
    "bilinear_rmse_vs_cryosat_mean_over_months": float(result_df["bilinear_rmse_vs_cryosat"].mean()),
    "network_bias_vs_cryosat_mean_over_months": float(result_df["network_bias_vs_cryosat"].mean()),
    "bilinear_bias_vs_cryosat_mean_over_months": float(result_df["bilinear_bias_vs_cryosat"].mean()),
    "network_corr_vs_cryosat_mean_over_months": float(result_df["network_corr_vs_cryosat"].mean()),
    "bilinear_corr_vs_cryosat_mean_over_months": float(result_df["bilinear_corr_vs_cryosat"].mean()),
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(overall, f, indent=2)
print("\nOverall (mean across the", len(result_df), "available winter months):")
for k, v in overall.items():
    print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------------------------
# Figure: one representative month (most recent with data), 3-panel
# CryoSat-2 truth / network / bilinear
# ---------------------------------------------------------------------------

plot_month = cryosat_months[-1]
cryo_map = month_maps[plot_month]
day_idx = np.where(sample_month == plot_month)[0]
net_map = Y_pred_phys[day_idx].mean(axis=0)
bilin_map = Y_base_phys[day_idx].mean(axis=0)

plot_bbox = {"lon_min": -182, "lon_max": -151, "lat_min": 60, "lat_max": 75}
proj, boundary_path, central_lon = fe.make_polar_proj(plot_bbox)
lon2d, lat2d = np.meshgrid(target_lon, target_lat)

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), subplot_kw={"projection": proj})
panels = [("CryoSat-2 RDEFT4 (truth)", cryo_map), ("Network (Stochastic UNet Mean)", net_map),
          ("Bilinear baseline", bilin_map)]
for ax, (title, field) in zip(axes, panels):
    fe.style_polar_ax(ax, proj, boundary_path, plot_bbox)
    field_masked = np.where(ocean_hw, field, np.nan)
    pc = ax.pcolormesh(lon2d, lat2d, field_masked, transform=ccrs.PlateCarree(),
                        cmap="viridis", vmin=0, vmax=2.5, shading="auto")
    ax.set_title(title, fontsize=11)
fig.colorbar(pc, ax=axes, orientation="horizontal", pad=0.05, shrink=0.5, label="SIT (m)")
fig.suptitle(f"2020-{plot_month:02d}: independent CryoSat-2 truth vs. PIOMAS-driven downscaled SIT", y=1.02)

fig_path = os.path.join(OUT_DIR, f"cryosat2_vs_network_{YEAR}{plot_month:02d}.png")
fig.savefig(fig_path, dpi=180, bbox_inches="tight")
print("\nSaved figure:", fig_path)
print("Done. Outputs in:", OUT_DIR)
