"""
build_X_Y_PIOMAS_obs_2021_daily.py

2021 counterpart of Version5/processing/build_X_Y_PIOMAS_obs_2020_daily.py, adapted only for
the year (PIOMAS_hiday_2021.nc already exists on glade; the NSIDC/CDR file used for 2020
already covers through 2021-12-31 with no changes needed; the JRA55 wind files' filename
date-suffix changed from ...210504.nc (2020) to ...220505.nc (2021) -- confirmed by listing
/glade/p/cesmdata/cseg/inputdata/ocn/jra55/v1.5_noleap/ directly, not assumed).

This is the *base* daily build (hi_d/Y regridded with plain skipna=True, no land-bleed
correction yet) -- mirrors the 2020 script's own two-step structure so the land-bleed fix
lives in a separate, auditable second script
(build_X_Y_PIOMAS_obs_2021_daily_v2_landfixed.py), exactly like the 2020 precedent.

Output: X_PIOMAS_obs_2021_daily_interp.nc / Y_PIOMAS_obs_2021_daily.nc in
/glade/derecho/scratch/skygale/downscaling.
"""

import glob
import os
import warnings
import numpy as np
import xesmf as xe
import xarray as xr
from datetime import datetime, timezone

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")
warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS")

YEAR = int(os.environ.get("PIOMAS_YEAR", "2021"))  # 2026-09-24: year overridable; default unchanged
FILE_TAG = os.environ.get("FILE_TAG", "")  # 2026-09-29: e.g. "_cdrv6" so the 2010-2023 rebuild doesn't overwrite existing years
run_name = f"PIOMAS_obs_{YEAR}{FILE_TAG}_daily"
JRA55_FILENAME_SUFFIX = os.path.basename(glob.glob(f"/glade/p/cesmdata/cseg/inputdata/ocn/jra55/v1.5_noleap/JRA.v1.5.u_10.TL319.{YEAR}.*.nc")[0]).split(".")[-2]  # per-year date suffix (e.g. 220505 for 2021)

PIOMAS_DAILY_PATH = f"/glade/derecho/scratch/skygale/observations/PIOMAS_daily/PIOMAS_hiday_{YEAR}.nc"
# 2026-09-29: CDR source overridable (the 2010-2023 build uses a v6 file converted to this layout by
# observations/processing/convert_cdr_v6_to_noleap.py); defaults unchanged.
CDR_PATH = os.environ.get("CDR_PATH", "/glade/campaign/cesm/development/pcwg/ssmi/CDR/cdr_seaice_conc_daily_nh.cdr.noleap.19790101-20211231.nc")
CDR_START = os.environ.get("CDR_START", "1979-01-01")
CDR_END = os.environ.get("CDR_END", "2021-12-31")
JRA55_DIR = "/glade/p/cesmdata/cseg/inputdata/ocn/jra55/v1.5_noleap"

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

print("Destination grids built.")

# ---------------------------------------------------------------------------
# 1. True daily PIOMAS -> 1deg (hi_d) and 0.1deg (Y), reusing the same cached
#    regridder weights as the 2020 build (identical source PIOMAS grid).
# ---------------------------------------------------------------------------

ds_daily = xr.open_dataset(PIOMAS_DAILY_PATH)
piomas_lat = ds_daily["lat"].values
piomas_lon = ds_daily["lon"].values % 360
grid_piomas = xr.Dataset({"lat": (["nlat", "nlon"], piomas_lat), "lon": (["nlat", "nlon"], piomas_lon)})

regridder_piomas_to_1deg = xe.Regridder(
    grid_piomas, dst_1deg, method="bilinear", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/piomas_to_1deg_interp.nc", reuse_weights=True,
)
print(" >> Loaded regridder_piomas_to_1deg (reused weights).")

regridder_piomas_to_0p1deg = xe.Regridder(
    grid_piomas, dst_0p1deg, method="bilinear", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/piomas_to_0p1deg_interp.nc", reuse_weights=True,
)
print(" >> Loaded regridder_piomas_to_0p1deg (reused weights).")

daily_time = ds_daily["time"].values  # noleap cftime, 365 days, matches FOSI's own calendar
print(f"Daily time axis: {daily_time.size} days ({daily_time[0]} .. {daily_time[-1]}).")

hi_da = ds_daily["hi"].assign_coords(lat=(["nlat", "nlon"], piomas_lat), lon=(["nlat", "nlon"], piomas_lon))

print("Regridding true-daily PIOMAS hi -> 1deg (365 days, for X hi_d)...")
hi_1deg_daily = regridder_piomas_to_1deg(hi_da, skipna=True)
hi_1deg_daily = hi_1deg_daily.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
hi_1deg_daily = hi_1deg_daily.fillna(0).astype(np.float32).reset_coords(drop=True)
print("hi_d daily (true daily, not broadcast):", hi_1deg_daily.shape)

print("Regridding true-daily PIOMAS hi -> 0.1deg (365 days, for Y)...")
hi_0p1deg_daily = regridder_piomas_to_0p1deg(hi_da, skipna=True)
hi_0p1deg_daily = hi_0p1deg_daily.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
hi_0p1deg_daily = hi_0p1deg_daily.fillna(0).astype(np.float32).reset_coords(drop=True)
ds_daily.close()

# ---------------------------------------------------------------------------
# 2. CDR concentration -> 1deg, daily (same source file as 2020 -- it already
#    spans through 2021-12-31, so only the sel(time=daily_time) window changes).
# ---------------------------------------------------------------------------

ds_cdr = xr.open_dataset(CDR_PATH, chunks={"time": 365})
cdr_dates = xr.date_range(start=CDR_START, end=CDR_END, freq="D", calendar="noleap", use_cftime=True)
ds_cdr = ds_cdr.assign_coords(time=cdr_dates)
sic_cdr = ds_cdr["cdr_seaice_conc_daily"].sel(time=daily_time)

grid_cdr = xr.Dataset({
    "lat": (["nlat", "nlon"], ds_cdr["latitude"].values),
    "lon": (["nlat", "nlon"], ds_cdr["longitude"].values % 360),
})
sic_cdr = sic_cdr.rename({"jdim": "nlat", "idim": "nlon"}).assign_coords(
    lat=(["nlat", "nlon"], grid_cdr["lat"].values), lon=(["nlat", "nlon"], grid_cdr["lon"].values)
)

regridder_cdr_to_1deg = xe.Regridder(
    grid_cdr, dst_1deg, method="bilinear", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/cdr_to_1deg_interp.nc", reuse_weights=True,
)
print(" >> Loaded regridder_cdr_to_1deg (reused weights).")

print("Regridding CDR concentration -> 1deg (365 days)...")
aice_1deg_daily = regridder_cdr_to_1deg(sic_cdr, skipna=True).compute()
aice_1deg_daily = aice_1deg_daily.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
aice_1deg_daily = (aice_1deg_daily / 100.0).fillna(0).astype(np.float32).reset_coords(drop=True)
ds_cdr.close()
print("aice_d daily:", aice_1deg_daily.shape)

# ---------------------------------------------------------------------------
# 3. JRA55 winds -> 1deg, daily-mean (same as 2020 except the filename
#    date-suffix: 220505 for 2021 vs. 210504 for 2020 -- confirmed by listing
#    the source directory, not assumed).
# ---------------------------------------------------------------------------

_ds_jra = xr.open_dataset(f"{JRA55_DIR}/JRA.v1.5.u_10.TL319.{YEAR}.{JRA55_FILENAME_SUFFIX}.nc")
jra_lat_full = _ds_jra["latitude"].values
jra_lon_full = _ds_jra["longitude"].values
_ds_jra.close()

jlat_idx = np.where((jra_lat_full >= bbox_regrid["lat_min"]) & (jra_lat_full <= bbox_regrid["lat_max"]))[0]
jlon_idx = np.where((jra_lon_full >= lon_min_regrid) & (jra_lon_full <= lon_max_regrid))[0]
jr0, jr1 = jlat_idx.min(), jlat_idx.max()
jc0, jc1 = jlon_idx.min(), jlon_idx.max()

grid_jra = xr.Dataset({
    "lat": ("latitude", jra_lat_full[jr0:jr1 + 1]),
    "lon": ("longitude", jra_lon_full[jc0:jc1 + 1]),
})

regridder_jra_to_1deg = xe.Regridder(
    grid_jra, dst_1deg, method="bilinear", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/jra55_to_1deg_interp_obs2020.nc", reuse_weights=True,
)
print(" >> Loaded regridder_jra_to_1deg (reused weights -- same JRA55 source grid every year).")


def process_wind(var):
    ds = xr.open_dataset(f"{JRA55_DIR}/JRA.v1.5.{var}.TL319.{YEAR}.{JRA55_FILENAME_SUFFIX}.nc")
    da = ds[var].isel(latitude=slice(jr0, jr1 + 1), longitude=slice(jc0, jc1 + 1))
    da = da.resample(time="1D").mean()
    da_reg = regridder_jra_to_1deg(da, skipna=True)
    da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    da_reg = da_reg.fillna(0).astype(np.float32)
    ds.close()
    return da_reg


print("Processing u_10...")
u_10_daily = process_wind("u_10").sel(time=daily_time).reset_coords(drop=True)
print("Processing v_10...")
v_10_daily = process_wind("v_10").sel(time=daily_time).reset_coords(drop=True)
print("u_10/v_10 daily:", u_10_daily.shape, v_10_daily.shape)

# ---------------------------------------------------------------------------
# 4. Assemble X / Y
# ---------------------------------------------------------------------------

channel_order = ["hi_d", "aice_d", "u_10", "v_10"]
channels = {"hi_d": hi_1deg_daily, "aice_d": aice_1deg_daily, "u_10": u_10_daily, "v_10": v_10_daily}
stacked = [channels[c] for c in channel_order]
X_ds = xr.concat(stacked, dim="channel")
X_ds.name = "X"
X_ds = X_ds.assign_coords(channel=channel_order)
X_ds = X_ds.expand_dims({"ensemble": [0]})

X_ds.attrs["description"] = (
    "True-daily observational-application predictors for the FOSI-trained (JRA55-forced) "
    "recommended EngressNet config: genuine daily PIOMAS v2.1 hi + NSIDC/CDR concentration + "
    "JRA55 forcing winds, all regridded bilinear to the 1-degree regional grid. 2021 "
    "counterpart of X_PIOMAS_obs_2020_daily_interp.nc; NOT land-bleed-fixed yet -- see "
    "build_X_Y_PIOMAS_obs_2021_daily_v2_landfixed.py for the corrected version actually "
    "used for model input."
)
X_ds.attrs["source_run"] = run_name
X_ds.attrs["regrid_method"] = "interp (bilinear)"
X_ds.attrs["created_by"] = "Sky Gale"
X_ds.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
X_ds.attrs["variables"] = (
    "hi_d: PIOMAS v2.1 daily sea ice thickness (m), genuine daily (not monthly-broadcast); "
    "aice_d: NSIDC/NOAA CDR sea ice concentration (fraction), daily; "
    "u_10/v_10: JRA55 10m wind (m s-1), daily mean"
)
X_ds = X_ds.transpose("ensemble", "time", "channel", "lat", "lon")

save_path_X = f"{OUT_DIR}/X_{run_name}_interp.nc"
X_ds.to_netcdf(save_path_X)
print("Saved X to:", save_path_X)

Y_ds = hi_0p1deg_daily.expand_dims({"channel": [0]})
Y_ds.name = "Y"
Y_ds = Y_ds.expand_dims({"ensemble": [0]})
Y_ds = Y_ds.transpose("ensemble", "time", "channel", "lat", "lon")
Y_ds.attrs["description"] = (
    "Genuine daily PIOMAS v2.1 hi regridded to the fine 0.1-degree regional grid -- a "
    "self-consistency reference (PIOMAS compared against itself downscaled), not an "
    "independent SIT validation truth. 2021 counterpart of Y_PIOMAS_obs_2020_daily.nc."
)
Y_ds.attrs["source_run"] = run_name
Y_ds.attrs["created_by"] = "Sky Gale"
Y_ds.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
Y_ds.attrs["variables"] = "hi_d: PIOMAS v2.1 daily sea ice thickness (m)"

save_path_Y = f"{OUT_DIR}/Y_{run_name}.nc"
Y_ds.to_netcdf(save_path_Y)
print("Saved Y to:", save_path_Y)

# ---------------------------------------------------------------------------
# 5. Checks
# ---------------------------------------------------------------------------

X_check = xr.open_dataset(save_path_X).X
Y_check = xr.open_dataset(save_path_Y).Y
print("X:", X_check.shape, X_check.channel.values)
print("Y:", Y_check.shape)

X_fosi = xr.open_dataset(f"{OUT_DIR}/X_FOSI_HR_JRA55_daily_interp.nc").X
Y_fosi = xr.open_dataset(f"{OUT_DIR}/Y_FOSI_HR_JRA55_daily.nc").Y
lat_ok = np.allclose(X_check.lat.values, X_fosi.lat.values) and np.allclose(X_check.lon.values, X_fosi.lon.values)
hlat_ok = np.allclose(Y_check.lat.values, Y_fosi.lat.values) and np.allclose(Y_check.lon.values, Y_fosi.lon.values)
print("X low-res grid matches FOSI's:", lat_ok)
print("Y high-res grid matches FOSI's:", hlat_ok)
if not (lat_ok and hlat_ok):
    raise RuntimeError("Grid mismatch against X_FOSI_HR_JRA55_daily_*.nc.")

hi_d_std_within_month = X_check.sel(channel="hi_d").isel(ensemble=0).groupby("time.month").std("time").mean()
print("Mean within-month std of hi_d (should be > 0, unlike a monthly-broadcast version):",
      float(hi_d_std_within_month))
print("Done.")
