"""Standalone script: conservative-regridded daily MESA Y, full HIST+RCP8.5 pipeline,
2000-2020 -- the harder MESA counterpart to build_Y_FOSI-HR_daily_conservative_2000_2020.py.

Unlike the existing build_Y_MESA-HR_daily_conservative.py (2015-2021 only, entirely within
the RCP8.5 era so it could reuse that era's file-collection/grid/regridder machinery
directly with no stitching), the standard 4-window grid's earliest training window
(2000-2004) starts before the 2006 HIST/RCP8.5 boundary. This script replicates the
production stitching pipeline (build_X_Y_from_MESA-HR_daily.py + rcp85 +
stitch_MESA_HR_daily_hist_rcp85.py) but with CONSERVATIVE regridding instead of bilinear,
so a model can be trained end-to-end against conservative truth across the full standard
grid (2000-2004/2005-2009/2010-2014/2015-2019, test 2020), not just the one 2015-2021 split
already covered.

Member alignment -- copied verbatim from stitch_MESA_HR_daily_hist_rcp85.py, NOT
re-derived, per this project's established convention (same reasoning as
member_metrics.py's formula-for-formula copying): HIST has 9 members (d651007, .002-.010,
sorted glob order -> index 0-8); RCP8.5 only has 6 of those 9 with daily ice+wind data
(d651009, .004/.005/.006/.007/.008/.010, sorted glob order -> index 0-5). Verified directly
against the real filesystem (2026-08, this script's own dry-run check) that all 6 target
members have the needed hi_d chunk files on both sides:
    RCP idx 0 (.004) -> HIST idx 2      HIST file: *.20000102-20051231.nc
    RCP idx 1 (.005) -> HIST idx 3      (same chunk, all 6 HIST members)
    RCP idx 2 (.006) -> HIST idx 4
    RCP idx 3 (.007) -> HIST idx 5
    RCP idx 4 (.008) -> HIST idx 6
    RCP idx 5 (.010) -> HIST idx 8      RCP files: *.20060101-20160101.nc +
                                                    *.20160102-20260101.nc (both needed --
                                                    2020 falls in the second chunk)

Grid construction is IDENTICAL for HIST and RCP8.5 (verified directly by comparing both
production scripts: both call `pop_tools.get_grid("POP_tx0.1v2")` with the same
bbox/bbox_regrid) -- built once here and reused for both eras, unlike the FOSI conservative
script which doesn't have this HIST/RCP8.5 split at all.

NaN handling at the 2006 join: stitch_MESA_HR_daily_hist_rcp85.py found two real boundary
artifacts (HIST X's last timestep entirely NaN; RCP8.5 X's first timestep partially NaN in
2 of 3 channels) but explicitly noted "Y's HIST file doesn't have this problem" for the
first one -- X-specific issues, not Y's. This script still applies the same generic
any-NaN-timestep check to both HIST-era and RCP8.5-era Y independently before concatenating
(same function, same bar, applied on principle rather than assuming Y is clean just because
the production stitch found it was last time), but does not expect it to actually drop
anything for hi_d specifically.

Per-file trimming BEFORE regridding (not after) for both eras -- required lesson from the
existing 2015-2021 MESA conservative script's docstring ("the first version of this script
regridded each member's full ~20-year multi-file span before trimming, which is almost
certainly why that attempt got OOM-killed at 128GB").

Output is spliced into a full-length copy of the production Y_MESA_HR_daily.nc (same shape
as X_MESA_HR_daily_*.nc), matching every other conservative-regrid script in this project.

Run this on Casper/Derecho with /glade/campaign and pop_tools' grid data mounted:
    python build_Y_MESA-HR_daily_conservative_2000_2020.py
"""

import os
import glob
import warnings
import numpy as np
import pop_tools
import xesmf as xe
import xarray as xr
from datetime import datetime, timezone

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")
warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS")

TARGET_VAR = "hi_d"
HIST_TO_RCP_ENSEMBLE_IDX = [2, 3, 4, 5, 6, 8]  # copied verbatim from stitch_MESA_HR_daily_hist_rcp85.py
RCP_EXCLUDED_MEMBER_TAGS = ["sehires38", ".30-2006-2100.002", ".31-2006-2100.003", "-2006-2100.009"]
HIST_YEAR_START, HIST_YEAR_END = 2000, 2005
RCP_YEAR_START, RCP_YEAR_END = 2006, 2020

comps = {"hi_d": "ice"}

# ---------- 1. Collect files ----------

hist_dirs_all = sorted(glob.glob("/glade/campaign/collections/gdex/data/d651007/b.e13.*"))
hist_dirs_all = [d for d in hist_dirs_all if "sehires38" not in d]
if len(hist_dirs_all) != 9:
    raise RuntimeError(f"Expected 9 HIST members, found {len(hist_dirs_all)} -- check EXCLUDED tags are still in sync.")
hist_dirs = [hist_dirs_all[i] for i in HIST_TO_RCP_ENSEMBLE_IDX]

rcp_dirs = sorted(glob.glob("/glade/campaign/collections/gdex/data/d651009/b.e13.*"))
rcp_dirs = [d for d in rcp_dirs if not any(tag in d for tag in RCP_EXCLUDED_MEMBER_TAGS)]
if len(rcp_dirs) != 6:
    raise RuntimeError(f"Expected 6 RCP8.5 members, found {len(rcp_dirs)} -- check EXCLUDED_MEMBER_TAGS is still in sync.")

print("HIST members (6-member subset, HIST index order):")
for i, d in zip(HIST_TO_RCP_ENSEMBLE_IDX, hist_dirs):
    print(f"  HIST idx {i}: {d}")
print("RCP8.5 members:")
for i, d in enumerate(rcp_dirs):
    print(f"  RCP idx {i}: {d}")


def collect_hi_d_files(d, start_year, end_year):
    pattern = f"{d}/{comps[TARGET_VAR]}/proc/tseries/day_1/*.{TARGET_VAR}.*.nc"
    files = sorted(glob.glob(pattern))
    return [f for f in files if start_year <= int(f.split(".")[-2][:4]) <= end_year]


hist_files = [collect_hi_d_files(d, HIST_YEAR_START, HIST_YEAR_END) for d in hist_dirs]
rcp_files = [collect_hi_d_files(d, RCP_YEAR_START, RCP_YEAR_END) for d in rcp_dirs]
for i, (d, files) in enumerate(zip(hist_dirs, hist_files)):
    if not files:
        raise RuntimeError(f"No HIST hi_d files found for member {d} in {HIST_YEAR_START}-{HIST_YEAR_END}")
    print(f"HIST member {i}: {len(files)} file(s) -- {[os.path.basename(f) for f in files]}")
for i, (d, files) in enumerate(zip(rcp_dirs, rcp_files)):
    if not files:
        raise RuntimeError(f"No RCP8.5 hi_d files found for member {d} in {RCP_YEAR_START}-{RCP_YEAR_END}")
    print(f"RCP8.5 member {i}: {len(files)} file(s) -- {[os.path.basename(f) for f in files]}")

# ---------- 2. Native grid + conservative source grid (identical for HIST and RCP8.5 --
# same pop_tools.get_grid("POP_tx0.1v2") call as both production scripts) ----------

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
print("Ice conservative source grid built (exact POP U-point corners) -- shared by HIST and RCP8.5.")

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

WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

print("Building conservative Y regridder (reusing cached weights from the 2015-2021 MESA script -- same grids)...")
regridder_cons = xe.Regridder(
    grid_ice_hr_conserv, dst_0p1deg_b, method="conservative", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/ice_hr_to_0p1deg_cons_mesa.nc", reuse_weights=True,
)
print(" >> Regridder ready.")


def process_file(file, time_start, time_end):
    """Trims to [time_start, time_end] BEFORE regridding (not after) -- see module
    docstring for why (OOM lesson from the earlier 2015-2021 MESA script). skipna=True so a
    coastal destination cell's regrid weight is renormalized over only its ocean-side
    source contributions."""
    ds = xr.open_dataset(file)
    da = ds[TARGET_VAR].rename({"nj": "nlat", "ni": "nlon"})
    da = da.isel(nlat=mask_ice_hr)
    da = da.sel(time=slice(time_start, time_end))
    if da.sizes["time"] == 0:
        ds.close()
        return None

    da_reg = regridder_cons(da, skipna=True)
    da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    da_reg = da_reg.fillna(0).astype(np.float32)
    ds.close()
    return da_reg


def drop_any_nan_time(da, label):
    """Copied verbatim (same any-NaN bar, same generic non-time-dims approach) from
    stitch_MESA_HR_daily_hist_rcp85.py's drop_fully_nan_time -- applied here to Y
    specifically even though that script found Y's HIST side clean, on the same
    "generic on purpose" principle."""
    non_time_dims = [d for d in da.dims if d != "time"]
    bad_time = da.isnull().any(dim=non_time_dims).compute()
    n_bad = int(bad_time.sum())
    if n_bad > 0:
        bad_times = da["time"].values[bad_time.values]
        print(f"  Dropping {n_bad} NaN-containing {label} timestep(s): {bad_times}")
        da = da.sel(time=~bad_time)
    return da


# ---------- 3. Build HIST-era Y (6-member subset, 2000-2005) ----------

print(f"=== Building HIST-era conservative Y, {HIST_YEAR_START}-{HIST_YEAR_END} ===")
hist_member_das = []
for i, files in enumerate(hist_files):
    print(f"  HIST member {i} (HIST idx {HIST_TO_RCP_ENSEMBLE_IDX[i]})...")
    parts = [p for p in (process_file(f, f"{HIST_YEAR_START}-01-01", f"{HIST_YEAR_END}-12-31") for f in files) if p is not None]
    if not parts:
        raise RuntimeError(f"No {HIST_YEAR_START}-{HIST_YEAR_END} timesteps found for HIST member {i}")
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    da = da.expand_dims({"ensemble": [i]})
    hist_member_das.append(da)
hist_da = xr.concat(hist_member_das, dim="ensemble")
hist_da.name = "Y"
hist_da = drop_any_nan_time(hist_da, "HIST Y")
print(f"HIST-era shape: {hist_da.shape}, time {hist_da.time.values[0]} to {hist_da.time.values[-1]}")

# ---------- 4. Build RCP8.5-era Y (6 members, 2006-2020) ----------

print(f"=== Building RCP8.5-era conservative Y, {RCP_YEAR_START}-{RCP_YEAR_END} ===")
rcp_member_das = []
for i, files in enumerate(rcp_files):
    print(f"  RCP8.5 member {i}...")
    parts = [p for p in (process_file(f, f"{RCP_YEAR_START}-01-01", f"{RCP_YEAR_END}-12-31") for f in files) if p is not None]
    if not parts:
        raise RuntimeError(f"No {RCP_YEAR_START}-{RCP_YEAR_END} timesteps found for RCP8.5 member {i}")
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    da = da.expand_dims({"ensemble": [i]})
    rcp_member_das.append(da)
rcp_da = xr.concat(rcp_member_das, dim="ensemble")
rcp_da.name = "Y"
rcp_da = drop_any_nan_time(rcp_da, "RCP8.5 Y")
print(f"RCP8.5-era shape: {rcp_da.shape}, time {rcp_da.time.values[0]} to {rcp_da.time.values[-1]}")

# ---------- 5. Stitch at the 2006 boundary (ensemble coordinates already aligned 0-5 on
# both sides by construction) ----------

rcp_first_time = rcp_da["time"].values[0]
hist_da = hist_da.sel(time=hist_da["time"] < rcp_first_time)  # guard against any overlap, mirrors stitch_one

combined = xr.concat([hist_da, rcp_da], dim="time")
combined.name = "Y"
combined = combined.expand_dims({"channel": [0]})
combined = combined.transpose("ensemble", "time", "channel", "lat", "lon")
print(f"Stitched shape: {combined.shape}, time {combined.time.values[0]} to {combined.time.values[-1]}")

# ---------- 6. Splice into the full-length production Y (same pattern as every other
# conservative-regrid script in this project) ----------

print("Splicing into a copy of the production Y (preserves years outside 2000-2020 unchanged)...")
PROD_Y_PATH = "/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_daily.nc"
prod_Y = xr.open_dataset(PROD_Y_PATH)

if prod_Y.sizes["ensemble"] != combined.sizes["ensemble"]:
    raise RuntimeError(
        f"Ensemble size mismatch: production Y has {prod_Y.sizes['ensemble']} members, "
        f"this script built {combined.sizes['ensemble']}."
    )

patched = prod_Y.copy(deep=True)
patched["Y"].loc[dict(time=combined.time.values)] = combined.values

patched.attrs["description"] = (
    "High-resolution daily MESACLIP hi_d regridded to the regional 0.1-degree grid -- "
    "IDENTICAL to the production Y_MESA_HR_daily.nc except 2000-01-01 through "
    "2020-12-31 (or nearest available), which uses CONSERVATIVE (exact source-area-"
    "weighted) regridding instead of bilinear, built from scratch across the HIST/RCP8.5 "
    "boundary (2000-2005 HIST + 2006-2020 RCP8.5, stitched with the same 6-member "
    "alignment as the production Y_MESA_HR_daily.nc) for a full-standard-grid test of "
    "coastal-bias cause 1 (2000-2004/2005-2009/2010-2014/2015-2019, test 2020). Years "
    "outside 2000-2020 are untouched bilinear data."
)
patched.attrs["notes"] = (
    "Built by build_Y_MESA-HR_daily_conservative_2000_2020.py: conservative-regridded "
    "2000-2020 slice, HIST+RCP8.5 stitched at the 2006 boundary using the exact member "
    "alignment from stitch_MESA_HR_daily_hist_rcp85.py, then spliced into a copy of "
    "Y_MESA_HR_daily.nc by exact date. Sibling of the earlier "
    "build_Y_MESA-HR_daily_conservative.py (2015-2021 only, RCP8.5-only) -- that script's "
    "output is untouched by this one."
)
patched.attrs["created_by"] = "Sky Gale"
patched.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
patched.attrs["ensemble_members_rcp85"] = ", ".join(rcp_dirs)
patched.attrs["ensemble_members_hist"] = ", ".join(hist_dirs)

save_path = "/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_daily_conservative_2000_2020.nc"
patched.to_netcdf(save_path + ".tmp")
prod_Y.close()
patched.close()
os.replace(save_path + ".tmp", save_path)
print("Saved to:", save_path)

verify = xr.open_dataset(save_path)
print("Shape:", verify.Y.shape, " time range:", verify.time.values.min(), "to", verify.time.values.max())
