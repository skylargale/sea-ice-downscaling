"""Standalone script: conservative-regridded daily FOSI Y, for the 2015-2021 coastal-bias
cause-1 experiment (train 2015-2020, test 2021 -- matches FOSI_el0_dmed_es0_2015-2020_2021,
the baseline used for the coastal_ch/classif/attn_end comparison).

Root-cause investigation (2026-07-27, see Version4 memory) found the production Y regrid
(`build_X_Y_from_FOSI-HR_daily.py`'s `regridder_hr_to_0p1deg`, bilinear) produces noisy truth
right at the coast -- a destination cell with a tiny true ocean footprint still gets a full
thickness value, and bilinear averaging over a mostly-land stencil is unstable. A conservative
regridder (exact source-cell-area weighting, using the t13 grid's true U-point corners --
`grid_ice_hr_conserv`, already built in the production script for X's 1-degree `avg` variant)
cuts that coastal noise ~2.4x when tested standalone on one year. That test never fed into an
actual training run, though -- re-evaluating an *already-trained* model (trained on the noisier
bilinear truth) against conservative truth without retraining just measures against a moved
target. This script produces a real conservative-regridded Y so a model can actually be trained
against it, to test cause 1 for real.

Only processes hi_d for 2015-2021 (not the full 1958-2022 record) -- that's all the baseline
run needs (train 2015-2020, test 2021), and keeps this a cheap, targeted experiment rather than
a full production data rebuild. Grid-construction code (native t13 grid, `grid_ice_hr_conserv`
corner arrays, `bbox`/`bbox_regrid`) is copied verbatim from `build_X_Y_from_FOSI-HR_daily.py`
so results are directly comparable; only the destination-grid bounds (`dst_0p1deg_b`, new here)
and the regridder method (`conservative` instead of `bilinear`) differ.

Run this on Casper/Derecho with /glade/campaign mounted:
    python build_Y_FOSI-HR_daily_conservative.py
"""

import os
import warnings
import numpy as np
import xesmf as xe
import xarray as xr
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")
warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS")

YEARS = range(2015, 2022)  # train 2015-2020 + test 2021, inclusive

# ---------- 1. Collect files (hi_d only -- Y's only source variable) ----------

RUN_DIR = Path(
    "/glade/campaign/cgd/oce/projects/FOSI_BGC/HR/g.e22.TL319_t13.G1850ECOIAF_JRA_HR.4p2z.001"
)
TSERIES_DIR = RUN_DIR / "ice" / "proc" / "tseries" / "day_1"

run_name = "FOSI_HR_JRA55_daily"
target_var = "hi_d"

all_files = sorted(TSERIES_DIR.glob(f"*.cice.h1.{target_var}.*.nc"))
run_files = [f for f in all_files if int(f.name.split(".")[-2][:4]) in YEARS]
found_years = sorted({int(f.name.split(".")[-2][:4]) for f in run_files})
print(f"{target_var}: {len(run_files)} files covering years {found_years}")
missing_years = sorted(set(YEARS) - set(found_years))
if missing_years:
    raise RuntimeError(f"Missing hi_d files for years: {missing_years}")

# ---------- 2. Native grid + conservative source grid (identical to the production script) ----------

bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360

bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360

ds_grid = xr.open_dataset(run_files[0])
ds_grid = ds_grid.rename({"nj": "nlat", "ni": "nlon"})

for v in ["TLON", "TLAT", "ULON", "ULAT"]:
    if v not in ds_grid:
        raise RuntimeError(f"Expected {v} in the HR history file, not found.")

tlon_full = ds_grid["TLON"].values % 360
tlat_full = ds_grid["TLAT"].values
ulon_full = ds_grid["ULON"].values % 360
ulat_full = ds_grid["ULAT"].values
ny_full = tlon_full.shape[0]

mask_ice_hr = np.any(
    (tlat_full >= bbox_regrid["lat_min"])
    & (tlat_full <= bbox_regrid["lat_max"])
    & (tlon_full >= lon_min_regrid)
    & (tlon_full <= lon_max_regrid),
    axis=1,
)

grid_ice_hr = xr.Dataset({
    "lat": (["nlat", "nlon"], tlat_full[mask_ice_hr]),
    "lon": (["nlat", "nlon"], tlon_full[mask_ice_hr]),
})

print("Native t13 grid prepared.")

rows = np.where(mask_ice_hr)[0]
r0, r1 = rows.min(), rows.max()

if not np.array_equal(rows, np.arange(r0, r1 + 1)):
    raise RuntimeError(
        "mask_ice_hr rows are not contiguous in nlat -- the corner "
        "construction below assumes a contiguous latitude band."
    )

if r0 < 1 or r1 + 1 > ny_full - 1:
    raise RuntimeError(
        "Masked region touches the native grid's j-edge/pole fold -- "
        "corner construction near the true boundary isn't handled here."
    )

ulon_i = np.pad(ulon_full, ((0, 0), (1, 0)), mode="wrap")
ulat_i = np.pad(ulat_full, ((0, 0), (1, 0)), mode="wrap")
lon_b_ice = ulon_i[r0 - 1:r1 + 1, :]
lat_b_ice = ulat_i[r0 - 1:r1 + 1, :]

grid_ice_hr_conserv = xr.Dataset({
    "lat": (["nlat", "nlon"], tlat_full[r0:r1 + 1, :]),
    "lon": (["nlat", "nlon"], tlon_full[r0:r1 + 1, :]),
    "lat_b": (["nlat_b", "nlon_b"], lat_b_ice),
    "lon_b": (["nlat_b", "nlon_b"], lon_b_ice),
})

print("Ice conservative source grid built (exact t13 U-point corners).")

# ---------- 3. Destination grid, WITH bounds (new -- production script only builds the
# bounds-less version, since it only ever uses bilinear for Y) ----------

dst_0p1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 0.1, 0.1)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 0.1, 0.1)),
})

dst_lat_c, dst_lon_c = dst_0p1deg.lat.values, dst_0p1deg.lon.values
dst_lat_b = np.concatenate([[dst_lat_c[0] - 0.05], dst_lat_c + 0.05])
dst_lon_b = np.concatenate([[dst_lon_c[0] - 0.05], dst_lon_c + 0.05])

dst_0p1deg_b = xr.Dataset({
    "lat": ("lat", dst_lat_c),
    "lon": ("lon", dst_lon_c),
    "lat_b": ("lat_b", dst_lat_b),
    "lon_b": ("lon_b", dst_lon_b),
})

# ---------- 4. Conservative regridder for Y ----------

WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

print("Building conservative Y regridder (this is the slow step)...")
regridder_hr_to_0p1deg_cons = xe.Regridder(
    grid_ice_hr_conserv, dst_0p1deg_b, method="conservative", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/hr_t13_to_0p1deg_cons.nc", reuse_weights=False,
)
print(" >> Built regridder_hr_to_0p1deg_cons.")


def process_scalar(file, var, regridder):
    """Identical to the production script's process_scalar -- skipna=True so a coastal
    destination cell's regrid weight is renormalized over only its ocean-side source
    contributions, instead of a single land-side NaN poisoning the whole average."""
    ds = xr.open_dataset(file)
    da = ds[var].rename({"nj": "nlat", "ni": "nlon"})
    da = da.isel(nlat=mask_ice_hr)

    da_reg = regridder(da, skipna=True)
    da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    da_reg = da_reg.fillna(0).astype(np.float32)
    ds.close()
    return da_reg


# ---------- 5. Build Y ----------

print(f"=== Building conservative Y, target_var={target_var}, years={found_years} ===")

y_parts = [process_scalar(f, target_var, regridder_hr_to_0p1deg_cons) for f in run_files]
Y_da = xr.concat(y_parts, dim="time")
Y_da.name = "Y"

Y_ds = Y_da.expand_dims({"channel": [0]})
Y_ds = Y_ds.expand_dims({"ensemble": [0]})
Y_ds = Y_ds.transpose("ensemble", "time", "channel", "lat", "lon")

# ---------- 6. Splice into the full-length production Y so the output has the SAME
# (ensemble, time, channel, lat, lon) shape as X_{run_name}_*.nc -- train_engressnet's
# split_train_test indexes X and Y positionally by a boolean year-mask built from X's time
# axis (Y_train_raw = Y[:, train_mask_t]), so a Y file scoped to only 2015-2021 (shorter
# than X's full 1958-2022 record) raises an IndexError at training time. Splicing into a
# copy of the production Y avoids re-regridding the other ~58 years, which we don't need
# anyway since TRAIN_YEARS/TEST_YEARS never select outside 2015-2021 for this experiment.
# (Originally this script just saved the bare 2015-2021 Y_ds directly -- that shape
# mismatch crashed the first FOSI_consregrid training attempt; patched here to match how
# build_Y_MESA-HR_daily_conservative.py does it, after finding the same bug there too.)

print("Splicing into a copy of the production Y (preserves years outside 2015-2021 unchanged)...")
PROD_Y_PATH = f"/glade/derecho/scratch/skygale/Downscaling_Data/Y_{run_name}.nc"
prod_Y = xr.open_dataset(PROD_Y_PATH)

patched = prod_Y.copy(deep=True)
# Y_ds is still a bare (named) DataArray here, not a Dataset -- Y_ds["Y"] would look up a
# coordinate named "Y" (KeyError) rather than the array's own values; use .values directly.
patched["Y"].loc[dict(time=Y_ds.time.values)] = Y_ds.values

patched.attrs["description"] = (
    "Prescribed-atmosphere (JRA55-forced) t13 hindcast, hi_d regridded to the fine "
    "0.1-degree regional grid -- IDENTICAL to the production Y_FOSI_HR_JRA55_daily.nc "
    "except 2015-01-02 through 2021-12-31 (or 2022-01-01, whichever the source files cover), "
    "which uses CONSERVATIVE (exact source-area-weighted) regridding instead of bilinear, "
    "for the coastal-bias cause-1 experiment (train 2015-2020, test 2021). Years outside "
    "that window are untouched bilinear data (irrelevant here since TRAIN_YEARS/TEST_YEARS "
    "never select them)."
)
patched.attrs["source_run"] = run_name
patched.attrs["regrid_method"] = "conservative (2015-2021 slice only; bilinear elsewhere)"
patched.attrs["notes"] = (
    "NaNs filled with zero. Built by build_Y_FOSI-HR_daily_conservative.py: conservative-"
    "regridded 2015-2021 slice computed with the exact t13 U-point corner grid + "
    "skipna=True (same convention as the production script), then spliced into a copy of "
    "Y_FOSI_HR_JRA55_daily.nc by exact date."
)
patched.attrs["created_by"] = "Sky Gale"
patched.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
patched.attrs["variables"] = "hi_d: sea ice thickness (m)"

save_path = f"/glade/derecho/scratch/skygale/Downscaling_Data/Y_{run_name}_conservative_2015_2021.nc"
patched.to_netcdf(save_path + ".tmp")
prod_Y.close()
patched.close()
os.replace(save_path + ".tmp", save_path)
print("Saved to:", save_path)

verify = xr.open_dataset(save_path)
print("Shape:", verify.Y.shape, " time range:", verify.time.values.min(), "to", verify.time.values.max())
