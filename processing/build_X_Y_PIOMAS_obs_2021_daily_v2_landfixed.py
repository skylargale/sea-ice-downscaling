"""
build_X_Y_PIOMAS_obs_2021_daily_v2_landfixed.py

2021 counterpart of Version5/processing/build_X_Y_PIOMAS_obs_2020_daily_v2_landfixed.py.
Fixes the same land-bleed regridding bug documented there: PIOMAS's raw hi field has no land
flag (land and zero-ice ocean are both literal 0.0, no NaN), so plain skipna=True regridding
lets land's zero values bleed into coastal ocean cells' bilinear interpolation stencils. This
masks confirmed-land cells (via PSC's official io.dat_360_120.output land/ocean mask -- static,
same file every year, no update needed) to NaN before regridding, so skipna=True actually
excludes only land while preserving genuine zero-ice open water as real 0.0.

Reuses aice_d/u_10/v_10 unchanged from build_X_Y_PIOMAS_obs_2021_daily.py's output (only hi_d
is affected by the land-bleed bug, since it's the only channel sourced from PIOMAS itself).

Run build_X_Y_PIOMAS_obs_2021_daily.py FIRST -- this script requires its output as OLD_X_PATH.

Output: X_PIOMAS_obs_2021_daily_v2_interp.nc / Y_PIOMAS_obs_2021_daily_v2.nc in
/glade/derecho/scratch/skygale/downscaling -- this is the version that should actually be
used as --test-x-path/--test-y-path for model inference, matching the 2020 precedent
(X_PIOMAS_obs_2020_daily_v2_interp.nc was the file actually used for the paper's Fig 8 / Section
4.6 results, NOT any of the separate, unused "biascorrected" proof-of-concept variants).
"""

import os
import warnings

import numpy as np
import xarray as xr
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

YEAR = int(os.environ.get("PIOMAS_YEAR", "2021"))  # 2026-09-24: year overridable; default unchanged
IO_MASK_PATH = "/glade/derecho/scratch/skygale/observations/PIOMAS_daily/utilities/io.dat_360_120.output"
PIOMAS_DAILY_PATH = f"/glade/derecho/scratch/skygale/observations/PIOMAS_daily/PIOMAS_hiday_{YEAR}.nc"
FILE_TAG = os.environ.get("FILE_TAG", "")  # 2026-09-29: e.g. "_cdrv6" so the 2010-2023 rebuild doesn't overwrite existing years
OLD_X_PATH = f"/glade/derecho/scratch/skygale/downscaling/X_PIOMAS_obs_{YEAR}{FILE_TAG}_daily_interp.nc"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/projects/SeaIceDownscaling/weighted_grids"
OUT_DIR = "/glade/derecho/scratch/skygale/downscaling"

bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360
bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360

dst_1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 1, 1.0)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 1, 1.0)),
})
dst_0p1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 0.1, 0.1)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 0.1, 0.1)),
})

# ---------------------------------------------------------------------------
# 1. Load PSC's official land/ocean mask (static -- identical file used for 2020)
# ---------------------------------------------------------------------------

rows = []
with open(IO_MASK_PATH) as f:
    for line in f:
        line = line.rstrip("\n")
        rows.append([int(line[i:i + 2]) for i in range(0, len(line), 2)])
io_mask = np.array(rows)  # (nlat=120, nlon=360), 0 = land
land_native = io_mask == 0
print(f"Native land mask: {land_native.sum()}/{land_native.size} cells are land "
      f"({100 * land_native.mean():.1f}%).")

# ---------------------------------------------------------------------------
# 2. Load true-daily PIOMAS, mask confirmed-land cells to NaN
# ---------------------------------------------------------------------------

import xesmf as xe  # noqa: E402 (kept local to mirror the 2020 script's import ordering)

ds_daily = xr.open_dataset(PIOMAS_DAILY_PATH)
piomas_lat = ds_daily["lat"].values
piomas_lon = ds_daily["lon"].values % 360
if land_native.shape != piomas_lat.shape:
    raise RuntimeError(f"io_mask shape {land_native.shape} != PIOMAS grid shape {piomas_lat.shape}")

hi_da = ds_daily["hi"].where(~xr.DataArray(land_native, dims=["nlat", "nlon"]))
hi_da = hi_da.assign_coords(lat=(["nlat", "nlon"], piomas_lat), lon=(["nlat", "nlon"], piomas_lon))

n_before = int((ds_daily["hi"].isel(time=0).values == 0).sum())
n_after_nan = int(np.isnan(hi_da.isel(time=0).values).sum())
print(f"Day 1: {n_before} cells were exactly 0 before masking; {n_after_nan} now NaN (land-masked).")

grid_piomas = xr.Dataset({"lat": (["nlat", "nlon"], piomas_lat), "lon": (["nlat", "nlon"], piomas_lon)})

regridder_piomas_to_1deg = xe.Regridder(
    grid_piomas, dst_1deg, method="bilinear", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/piomas_to_1deg_interp.nc", reuse_weights=True,
)
regridder_piomas_to_0p1deg = xe.Regridder(
    grid_piomas, dst_0p1deg, method="bilinear", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/piomas_to_0p1deg_interp.nc", reuse_weights=True,
)
print(" >> Loaded both regridders (reused weights -- grid geometry unchanged).")

daily_time = ds_daily["time"].values
print(f"Daily time axis: {daily_time.size} days ({daily_time[0]} .. {daily_time[-1]}).")

print("Regridding land-masked PIOMAS hi -> 1deg (365 days, for X hi_d)...")
hi_1deg_daily = regridder_piomas_to_1deg(hi_da, skipna=True)
hi_1deg_daily = hi_1deg_daily.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
hi_1deg_daily = hi_1deg_daily.fillna(0).astype(np.float32).reset_coords(drop=True)
print("hi_d daily (land-fixed):", hi_1deg_daily.shape)

print("Regridding land-masked PIOMAS hi -> 0.1deg (365 days, for Y)...")
hi_0p1deg_daily = regridder_piomas_to_0p1deg(hi_da, skipna=True)
hi_0p1deg_daily = hi_0p1deg_daily.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
hi_0p1deg_daily = hi_0p1deg_daily.fillna(0).astype(np.float32).reset_coords(drop=True)
ds_daily.close()

# ---------------------------------------------------------------------------
# 3. Assemble X: reuse aice_d/u_10/v_10 unchanged from the base 2021 daily build.
# ---------------------------------------------------------------------------

X_old = xr.open_dataset(OLD_X_PATH).X
channel_order = list(X_old.channel.values)
if channel_order != ["hi_d", "aice_d", "u_10", "v_10"]:
    raise RuntimeError(f"Unexpected channel order: {channel_order}")

if not (np.array_equal(X_old.lat.values, hi_1deg_daily.lat.values) and
        np.array_equal(X_old.lon.values, hi_1deg_daily.lon.values) and
        np.array_equal(X_old.time.values, hi_1deg_daily.time.values)):
    raise RuntimeError("Grid/time mismatch between base 2021 X and new land-fixed hi_d.")

X_new = X_old.copy(deep=True)
hi_idx = channel_order.index("hi_d")
X_new.values[:, :, hi_idx, :, :] = hi_1deg_daily.transpose("time", "lat", "lon").values[None, :, :, :]

old_hi = X_old.sel(channel="hi_d").values
new_hi = X_new.sel(channel="hi_d").values
print(f"\nhi_d change from land-bleed fix: mean={float(np.mean(new_hi - old_hi)):.4f} m, "
      f"max abs={float(np.max(np.abs(new_hi - old_hi))):.4f} m")

X_new.attrs["description"] = (
    "Land-bleed-fixed version of X_PIOMAS_obs_2021_daily_interp.nc: hi_d regridded with "
    "PSC's official land/ocean mask (io.dat_360_120.output) applied before interpolation. "
    "aice_d/u_10/v_10 unchanged. 2021 counterpart of X_PIOMAS_obs_2020_daily_v2_interp.nc, "
    "the file actually used for the paper's PIOMAS-driven observational-application results."
)
X_new.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

save_path_X = f"{OUT_DIR}/X_PIOMAS_obs_{YEAR}{FILE_TAG}_daily_v2_interp.nc"
X_new.to_netcdf(save_path_X)
print("Saved X to:", save_path_X)

# ---------------------------------------------------------------------------
# 4. Assemble Y (land-fixed)
# ---------------------------------------------------------------------------

Y_da = hi_0p1deg_daily.reset_coords(drop=True)
Y_ds = Y_da.expand_dims({"channel": [0]})
Y_ds.name = "Y"
Y_ds = Y_ds.expand_dims({"ensemble": [0]})
Y_ds = Y_ds.transpose("ensemble", "time", "channel", "lat", "lon")
Y_ds.attrs["description"] = (
    "Land-bleed-fixed version of Y_PIOMAS_obs_2021_daily.nc -- see "
    "X_PIOMAS_obs_2021_daily_v2_interp.nc's attrs / this script's docstring."
)
Y_ds.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

save_path_Y = f"{OUT_DIR}/Y_PIOMAS_obs_{YEAR}{FILE_TAG}_daily_v2.nc"
Y_ds.to_netcdf(save_path_Y)
print("Saved Y to:", save_path_Y)

# ---------------------------------------------------------------------------
# 5. Checks
# ---------------------------------------------------------------------------

X_check = xr.open_dataset(save_path_X).X
Y_check = xr.open_dataset(save_path_Y).Y
X_fosi = xr.open_dataset(f"{OUT_DIR}/X_FOSI_HR_JRA55_daily_interp.nc").X
Y_fosi = xr.open_dataset(f"{OUT_DIR}/Y_FOSI_HR_JRA55_daily.nc").Y
lat_ok = np.allclose(X_check.lat.values, X_fosi.lat.values) and np.allclose(X_check.lon.values, X_fosi.lon.values)
hlat_ok = np.allclose(Y_check.lat.values, Y_fosi.lat.values) and np.allclose(Y_check.lon.values, Y_fosi.lon.values)
print("\nX low-res grid matches FOSI's:", lat_ok)
print("Y high-res grid matches FOSI's:", hlat_ok)
if not (lat_ok and hlat_ok):
    raise RuntimeError("Grid mismatch against X_FOSI_HR_JRA55_daily_*.nc.")
print("Done.")
