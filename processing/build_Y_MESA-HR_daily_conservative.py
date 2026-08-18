"""Standalone script: conservative-regridded daily MESA Y, MESA twin of
build_Y_FOSI-HR_daily_conservative.py -- see that script for the full coastal-bias cause-1
rationale (production Y regrid is bilinear, noisy right at the coast; testing whether
training from scratch against conservative-regridded truth actually helps, since the earlier
re-evaluation-only test on FOSI was inconclusive).

Scoped to the 2015-2021 baseline (train 2015-2020, test 2021 -- matches
MESA_el0_dmed_es0_2015-2020_2021, the baseline for MESA's coastal RMSE comparison). That
window falls entirely within the RCP8.5 portion of the record (2006 onward), so this reuses
build_X_Y_from_MESA-HR_daily_rcp85.py's file collection/grid/regridder machinery directly and
does NOT need the HIST+RCP8.5 stitching step -- the stitched Y_MESA_HR_daily.nc is just HIST
(pre-2006) concatenated with RCP8.5 (2006+) in time, so this script's own direct RCP8.5 output
for 2015-2021 is equivalent to slicing that same period out of the stitched file.

Keeps the same 6-member ensemble structure as the production Y (MESA training treats each
ensemble member as extra samples, not a single realization -- see submit_engressnet_daily_mesa.sh).
Only hi_d is processed (Y's only source variable) -- no X, no wind/atm channels, no HIST.

Output is spliced into a full-length copy of the production Y_MESA_HR_daily.nc (same shape as
X_MESA_HR_daily_*.nc), not a standalone 2015-2021-only file -- train_engressnet's
split_train_test indexes X/Y positionally using a boolean mask built from X's time axis, so a
shorter Y file raises an IndexError at training time. Also trims each RCP8.5 file to
2015-2021 BEFORE regridding (not after) -- the first version of this script regridded each
member's full ~20-year multi-file span before trimming, which is almost certainly why that
attempt got OOM-killed at 128GB.

Run this on Casper/Derecho with /glade/campaign and pop_tools' grid data mounted:
    python build_Y_MESA-HR_daily_conservative.py
"""

import glob
import warnings
import numpy as np
import pop_tools
import xesmf as xe
import xarray as xr
from datetime import datetime, timezone

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")
warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS")

# ---------- 1. Collect files (hi_d only, RCP8.5 members, 2006-2021 -- see production
# script's docstring for why these 6 members and why end_year=2021) ----------

EXCLUDED_MEMBER_TAGS = ["sehires38", ".30-2006-2100.002", ".31-2006-2100.003", "-2006-2100.009"]

comps = {"hi_d": "ice"}


def collect_files(dirs, target_var, start_year, end_year=None):
    out = []
    for d in dirs:
        pattern = f"{d}/{comps[target_var]}/proc/tseries/day_1/*.{target_var}.*.nc"
        files = sorted(glob.glob(pattern))
        filtered = [f for f in files if start_year <= int(f.split(".")[-2][:4]) <= (end_year or 9999)]
        out.append(filtered)
    return out


high_res_dirs = sorted(glob.glob("/glade/campaign/collections/gdex/data/d651009/b.e13.*"))
high_res_dirs = [d for d in high_res_dirs if not any(tag in d for tag in EXCLUDED_MEMBER_TAGS)]
print(f"Usable RCP8.5 high-res members ({len(high_res_dirs)}):")
for d in high_res_dirs:
    print(" ", d)

target_var = "hi_d"
high_res_files = collect_files(high_res_dirs, target_var, start_year=2006, end_year=2021)
for d, files in zip(high_res_dirs, high_res_files):
    if not files:
        raise RuntimeError(f"No {target_var} files found for member {d}")
print("File counts per member:", [len(f) for f in high_res_files])

# ---------- 2. Native grid + conservative source grid (identical to the production RCP8.5
# script -- pop_tools POP_tx0.1v2, exact U-point corners) ----------

bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360

bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360

nat_ice_hr = pop_tools.get_grid("POP_tx0.1v2")
ice_lon = nat_ice_hr.TLONG % 360

mask_ice_hr = np.any(
    ((nat_ice_hr.TLAT >= bbox_regrid["lat_min"])
     & (nat_ice_hr.TLAT <= bbox_regrid["lat_max"])
     & (ice_lon >= lon_min_regrid)
     & (ice_lon <= lon_max_regrid)),
    axis=1,
)

print("Native grid prepared.")

has_upoints = hasattr(nat_ice_hr, "ULONG") and hasattr(nat_ice_hr, "ULAT")
if not has_upoints:
    raise RuntimeError("Expected ULONG/ULAT on the POP grid object for exact ice cell corners.")

tlon_full = nat_ice_hr.TLONG.values % 360
tlat_full = nat_ice_hr.TLAT.values
ulon_full = nat_ice_hr.ULONG.values % 360
ulat_full = nat_ice_hr.ULAT.values
ny_full = tlon_full.shape[0]

rows = np.where(mask_ice_hr)[0]
r0, r1 = rows.min(), rows.max()
if not np.array_equal(rows, np.arange(r0, r1 + 1)):
    raise RuntimeError("mask_ice_hr rows are not contiguous in nlat.")
if r0 < 1 or r1 + 1 > ny_full - 1:
    raise RuntimeError("Masked ice region touches the native grid's j-edge/pole fold.")

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

print("Ice conservative source grid built (exact POP U-point corners).")

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
regridder_ice_to_0p1deg_cons = xe.Regridder(
    grid_ice_hr_conserv, dst_0p1deg_b, method="conservative", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/ice_hr_to_0p1deg_cons_mesa.nc", reuse_weights=False,
)
print(" >> Built regridder_ice_to_0p1deg_cons.")


def process_file_hi(file):
    """Identical to the production script's process_file_hi, plus one addition: trims to
    the 2015-2021 window BEFORE regridding (not after) -- each RCP8.5 ice file is a 10-year
    chunk, so regridding the full chunk (as the first version of this script did, trimming
    only after concatenation) meant conservative-regridding ~3x more daily timesteps per
    member than needed, which is almost certainly why the first attempt at this job got
    OOM-killed at 128GB. skipna=True so a coastal destination cell's regrid weight is
    renormalized over only its ocean-side source contributions, instead of a single
    land-side NaN poisoning the whole average."""
    ds = xr.open_dataset(file)
    da = ds[target_var].rename({"nj": "nlat", "ni": "nlon"})
    da = da.isel(nlat=mask_ice_hr)
    da = da.sel(time=slice("2015-01-01", "2021-12-31"))
    if da.sizes["time"] == 0:
        ds.close()
        return None

    da_reg = regridder_ice_to_0p1deg_cons(da, skipna=True)
    da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    da_reg = da_reg.fillna(0).astype(np.float32)
    ds.close()
    return da_reg


# ---------- 5. Build Y (all 6 members) ----------

print(f"=== Building conservative Y, target_var={target_var}, {len(high_res_dirs)} members ===")

Y_list = []
for i, files in enumerate(high_res_files):
    print(f"Processing Ensemble #{i + 1}/{len(high_res_files)}...")
    parts = [p for p in (process_file_hi(f) for f in files) if p is not None]
    if not parts:
        raise RuntimeError(f"No 2015-2021 timesteps found for member #{i + 1} ({high_res_dirs[i]})")
    var_da = xr.concat(parts, dim="time")
    var_da.name = "Y"
    member_da = var_da.expand_dims({"ensemble": [i], "channel": [0]})
    member_da = member_da.transpose("ensemble", "time", "channel", "lat", "lon")
    Y_list.append(member_da)

Y_ds = xr.concat(Y_list, dim="ensemble")

if "channel" in Y_ds.dims and "channel" not in Y_ds.coords:
    Y_ds = Y_ds.assign_coords(channel=np.arange(Y_ds.sizes["channel"]))

# ---------- 6. Splice into the full-length production Y so the output has the SAME
# (ensemble, time, channel, lat, lon) shape as X_MESA_HR_daily_*.nc -- train_engressnet's
# split_train_test indexes X and Y positionally by a boolean year-mask built from X's time
# axis (Y_train_raw = Y[:, train_mask_t]), so a Y file scoped to only 2015-2021 (shorter
# than X's full 1999-2025 record) raises an IndexError at training time instead of just
# training on less data. Splicing into a copy of the production Y (same trick as the FOSI
# version of this script) avoids re-regridding the other ~20 years, which we don't need
# anyway since TRAIN_YEARS/TEST_YEARS never select outside 2015-2021 for this experiment.

print("Splicing into a copy of the production Y (preserves years outside 2015-2021 unchanged)...")
PROD_Y_PATH = "/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_daily.nc"
prod_Y = xr.open_dataset(PROD_Y_PATH)

if prod_Y.sizes["ensemble"] != Y_ds.sizes["ensemble"]:
    raise RuntimeError(
        f"Ensemble size mismatch: production Y has {prod_Y.sizes['ensemble']} members, "
        f"this script built {Y_ds.sizes['ensemble']} -- check EXCLUDED_MEMBER_TAGS is still "
        "in sync with the production build script."
    )

patched = prod_Y.copy(deep=True)
# Y_ds is still a bare DataArray at this point (xr.concat of DataArrays never got wrapped
# into a Dataset) -- Y_ds["Y"] looks up a coordinate named "Y" that doesn't exist, not the
# array's own values. Use Y_ds.values directly instead.
patched["Y"].loc[dict(time=Y_ds.time.values)] = Y_ds.values

patched.attrs["description"] = (
    "High-resolution daily MESACLIP hi_d regridded to the regional 0.1-degree grid -- "
    "IDENTICAL to the production Y_MESA_HR_daily.nc except 2015-01-01 through 2021-12-31, "
    "which uses CONSERVATIVE (exact source-area-weighted) regridding instead of bilinear, "
    "for the coastal-bias cause-1 experiment (train 2015-2020, test 2021). Same 6-member "
    "ensemble as the production Y file; years outside 2015-2021 are untouched bilinear data "
    "(irrelevant here since TRAIN_YEARS/TEST_YEARS never select them)."
)
patched.attrs["notes"] = (
    "Built by build_Y_MESA-HR_daily_conservative.py: conservative-regridded 2015-2021 slice "
    "computed with the exact POP U-point corner grid + skipna=True (same convention as the "
    "production script), then spliced into a copy of Y_MESA_HR_daily.nc by exact date."
)
patched.attrs["created_by"] = "Sky Gale"
patched.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
patched.attrs["ensemble_members"] = ", ".join(high_res_dirs)

save_path = "/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_daily_conservative_2015_2021.nc"
patched.to_netcdf(save_path + ".tmp")
prod_Y.close()
patched.close()
import os
os.replace(save_path + ".tmp", save_path)
print("Saved to:", save_path)

verify = xr.open_dataset(save_path)
print("Shape:", verify.Y.shape, " time range:", verify.time.values.min(), "to", verify.time.values.max())
