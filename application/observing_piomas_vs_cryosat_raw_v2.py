"""
observing_piomas_vs_cryosat_raw_v2_landfixed.py

Chases the PIOMAS-vs-CryoSat-2 disagreement to its source: does it already
exist in PIOMAS's own raw, coarse (1-degree) field, before any downscaling
network or bilinear upsampling touches it? Both
observing_cryosat2_correlation_diagnosis.py (network vs. bilinear show
nearly identical weak correlation) and observing_cryosat2_coarsening_check.py
(disagreement doesn't improve, gets slightly worse, when coarsened toward
CryoSat's own footprint) point at PIOMAS itself as the source, not the
downscaling step -- this confirms that directly by comparing PIOMAS's raw
1-degree hi_d channel (the actual X_PIOMAS_obs_2020_daily_v2_interp.nc input,
untouched by the network) against CryoSat-2, and mapping the spatial bias
pattern to see if it's spatially coherent (a real regional PIOMAS bias) or
scattered/random (noise).
"""

import os
import warnings

import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_engressnet as fe

YEAR = 2020
MONTHS = [1, 2, 3, 4, 10, 11, 12]
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"
X_PATH = "/glade/derecho/scratch/skygale/Downscaling_Data/X_PIOMAS_obs_2020_daily_v2_interp.nc"

OUT_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/piomas_vs_cryosat_raw_v2_landfixed"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load PIOMAS's raw 1-degree hi_d (the actual network input, untouched)
# ---------------------------------------------------------------------------

X_ds = xr.open_dataset(X_PATH)
hi_d = X_ds["X"].sel(channel="hi_d").isel(ensemble=0)   # (time=365, lat=21, lon=51)
piomas_lat = X_ds.lat.values
piomas_lon = X_ds.lon.values
print("PIOMAS raw 1deg grid:", piomas_lat.shape, piomas_lon.shape)

daily_time = X_ds.time.values
sample_month = np.array([t.month for t in daily_time])

# ---------------------------------------------------------------------------
# Regrid CryoSat-2 onto PIOMAS's own raw 1-degree grid directly (not the
# 0.1-degree common grid used before) -- comparing at PIOMAS's native scale.
# ---------------------------------------------------------------------------

dst_grid = xr.Dataset({"lat": ("lat", piomas_lat), "lon": ("lon", piomas_lon)})
regridder = None
diff_maps = {}
rows = []

for month in MONTHS:
    matches = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")]
    if not matches:
        continue
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, matches[0]))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)
    sit_da = sit.assign_coords(lat=(["y", "x"], ds["lat"].values), lon=(["y", "x"], ds["lon"].values % 360))

    if regridder is None:
        regridder = xe.Regridder(
            xr.Dataset({"lat": (["y", "x"], ds["lat"].values), "lon": (["y", "x"], ds["lon"].values % 360)}),
            dst_grid, method="bilinear", periodic=False,
            filename=f"{WEIGHTED_GRIDS_DIR}/cryosat2_rdeft4_to_piomas_1deg.nc", reuse_weights=False,
        )
        print(" >> Built regridder_cryosat_to_piomas_1deg.")

    cryo_1deg = regridder(sit_da, skipna=True).values   # (21, 51)
    ds.close()

    day_idx = np.where(sample_month == month)[0]
    piomas_month = hi_d.isel(time=day_idx).mean(dim="time").values   # (21, 51)

    valid = np.isfinite(cryo_1deg) & (cryo_1deg > 0)
    n = int(valid.sum())
    if n < 5:
        print(f"Month {month}: only {n} valid cells at 1deg, skipping.")
        continue

    diff = piomas_month - cryo_1deg
    diff_maps[month] = diff
    corr = float(np.corrcoef(piomas_month[valid], cryo_1deg[valid])[0, 1])
    bias = float(np.nanmean(diff[valid]))
    rmse = float(np.sqrt(np.nanmean(diff[valid] ** 2)))
    rows.append({
        "month": month, "n_valid_1deg_cells": n,
        "piomas_mean_m": float(np.nanmean(piomas_month[valid])),
        "cryosat_mean_m": float(np.nanmean(cryo_1deg[valid])),
        "bias_piomas_minus_cryosat": bias,
        "rmse": rmse, "corr": corr,
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "piomas_raw_vs_cryosat2_by_month.csv"), index=False)
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Spatial coherence check: is the bias pattern consistent month to month
# (a real regional PIOMAS bias), or does it flip around (noise)?
# Correlate each month's diff map against every other month's, at valid
# cells common to both.
# ---------------------------------------------------------------------------

months_with_data = sorted(diff_maps.keys())
print("\n=== Month-to-month correlation of the PIOMAS-minus-CryoSat bias PATTERN ===")
print("(high positive values = same cells are biased the same direction every month -> real regional bias)")
cross_rows = []
for i, m1 in enumerate(months_with_data):
    for m2 in months_with_data[i + 1:]:
        d1, d2 = diff_maps[m1], diff_maps[m2]
        valid = np.isfinite(d1) & np.isfinite(d2)
        if valid.sum() < 5:
            continue
        r = float(np.corrcoef(d1[valid], d2[valid])[0, 1])
        cross_rows.append({"month_1": m1, "month_2": m2, "bias_pattern_corr": r, "n": int(valid.sum())})
cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(os.path.join(OUT_DIR, "bias_pattern_month_to_month_corr.csv"), index=False)
print(cross_df.to_string(index=False))
print(f"\nMean month-to-month bias-pattern correlation: {cross_df['bias_pattern_corr'].mean():.3f}")

# ---------------------------------------------------------------------------
# Figure: mean bias map across all available months
# ---------------------------------------------------------------------------

mean_diff = np.nanmean(np.stack([diff_maps[m] for m in months_with_data]), axis=0)

plot_bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
proj, boundary_path, central_lon = fe.make_polar_proj(plot_bbox)
lon2d, lat2d = np.meshgrid(piomas_lon, piomas_lat)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj})
fe.style_polar_ax(ax, proj, boundary_path, plot_bbox)
vmax = np.nanmax(np.abs(mean_diff))
pc = ax.pcolormesh(lon2d, lat2d, mean_diff, transform=ccrs.PlateCarree(),
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85)
cb.set_label("PIOMAS minus CryoSat-2 (m), mean across %d available 2020 months" % len(months_with_data))
ax.set_title("Where does PIOMAS's raw 1-deg input disagree with CryoSat-2?")

fig_path = os.path.join(OUT_DIR, "piomas_minus_cryosat_bias_map.png")
fig.savefig(fig_path, dpi=180, bbox_inches="tight")
print("\nSaved figure:", fig_path)
print("Done. Outputs in:", OUT_DIR)
