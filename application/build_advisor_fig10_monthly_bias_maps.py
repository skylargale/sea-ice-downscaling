"""
build_advisor_fig10_monthly_bias_maps.py

Advisor figure 10: PIOMAS-minus-CryoSat-2 bias, shown as individual monthly
maps for Jan-Apr 2020 only -- the "good" months with full, stable CryoSat-2
coverage (see 09_cryosat2_annual_coverage.png; Oct-Dec are small,
partial-coverage samples and shouldn't be read with the same confidence).
Unlike 05_piomas_minus_cryosat_bias_map.png (the mean across all 7 available
months, mixing reliable and unreliable samples), this shows each reliable
month on its own panel so the month-to-month CONSISTENCY of the coastal
bias band is visible directly -- the evidence that this is a real regional
bias, not noise (month-to-month bias-pattern correlation was 0.77-0.90
among these four months, see
results/PIOMAS_obs_2020/piomas_vs_cryosat_raw/bias_pattern_month_to_month_corr.csv).
"""

import os
import warnings

import numpy as np
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
MONTHS = [1, 2, 3, 4]
MONTH_NAMES = {1: "January", 2: "February", 3: "March", 4: "April"}
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"
X_PATH = "/glade/derecho/scratch/skygale/Downscaling_Data/X_PIOMAS_obs_2020_daily_v2_interp.nc"
OUT_PATH = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/10_piomas_vs_cryosat_bias_by_month.png"

X_ds = xr.open_dataset(X_PATH)
hi_d = X_ds["X"].sel(channel="hi_d").isel(ensemble=0)
piomas_lat = X_ds.lat.values
piomas_lon = X_ds.lon.values
daily_time = X_ds.time.values
sample_month = np.array([t.month for t in daily_time])

dst_grid = xr.Dataset({"lat": ("lat", piomas_lat), "lon": ("lon", piomas_lon)})
regridder = None
piomas_maps = {}
cryo_maps = {}
diff_maps = {}

for month in MONTHS:
    matches = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")]
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, matches[0]))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)
    sit_da = sit.assign_coords(lat=(["y", "x"], ds["lat"].values), lon=(["y", "x"], ds["lon"].values % 360))

    if regridder is None:
        regridder = xe.Regridder(
            xr.Dataset({"lat": (["y", "x"], ds["lat"].values), "lon": (["y", "x"], ds["lon"].values % 360)}),
            dst_grid, method="bilinear", periodic=False,
            filename=f"{WEIGHTED_GRIDS_DIR}/cryosat2_rdeft4_to_piomas_1deg.nc", reuse_weights=True,
        )
    cryo_1deg = regridder(sit_da, skipna=True).values
    ds.close()

    day_idx = np.where(sample_month == month)[0]
    piomas_month = hi_d.isel(time=day_idx).mean(dim="time").values
    valid = np.isfinite(cryo_1deg) & (cryo_1deg > 0)
    piomas_maps[month] = np.where(valid, piomas_month, np.nan)
    cryo_maps[month] = np.where(valid, cryo_1deg, np.nan)
    diff_maps[month] = np.where(valid, piomas_month - cryo_1deg, np.nan)

X_ds.close()

sit_vmax = np.nanmax([np.nanmax(list(piomas_maps.values())), np.nanmax(list(cryo_maps.values()))])
diff_vmax = np.nanmax([np.nanmax(np.abs(d)) for d in diff_maps.values()])

plot_bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
proj, boundary_path, central_lon = fe.make_polar_proj(plot_bbox)
lon2d, lat2d = np.meshgrid(piomas_lon, piomas_lat)

fig, axes = plt.subplots(3, 4, figsize=(20, 16), subplot_kw={"projection": proj})

row_specs = [
    ("PIOMAS SIT (m)", piomas_maps, "viridis", 0, sit_vmax),
    ("CryoSat-2 SIT (m)", cryo_maps, "viridis", 0, sit_vmax),
    ("PIOMAS minus CryoSat-2 (m)", diff_maps, "RdBu_r", -diff_vmax, diff_vmax),
]

for row, (row_label, data, cmap, vmin, vmax) in enumerate(row_specs):
    for col, month in enumerate(MONTHS):
        ax = axes[row, col]
        fe.style_polar_ax(ax, proj, boundary_path, plot_bbox)
        pc = ax.pcolormesh(lon2d, lat2d, data[month], transform=ccrs.PlateCarree(),
                            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        if row == 0:
            ax.set_title(f"2020-{month:02d} ({MONTH_NAMES[month]})", fontsize=12)
    cb = fig.colorbar(pc, ax=axes[row, :].tolist(), orientation="vertical", pad=0.02, shrink=0.85)
    cb.set_label(row_label, fontsize=11)

fig.suptitle("PIOMAS vs. CryoSat-2 sea ice thickness, month by month (Jan-Apr 2020 -- full CryoSat-2 coverage)\n"
             "Row 1: PIOMAS input   Row 2: CryoSat-2 truth   Row 3: difference "
             "(same coastal thin-bias band persists every month, not random noise)", y=0.995, fontsize=14)

fig.savefig(OUT_PATH, dpi=170, bbox_inches="tight")
print("Saved:", OUT_PATH)
