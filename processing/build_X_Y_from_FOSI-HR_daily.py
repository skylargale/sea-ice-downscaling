"""Auto-generated from build_X_Y_from_FOSI-HR_daily.ipynb. Do not hand-edit; edit the notebook and regenerate."""

# # Build perfect-model X/Y from FOSI_BGC HR (prescribed-atmosphere t13 run), daily
#
# Same convention as `build_X_Y_from_MESA-HR_daily.ipynb`, applied to the single
# JRA55-forced HR hindcast instead of the CESM LE ensemble:
#
# - **X, coarsened both ways** — the t13 native ice fields regridded down to the 1-degree
#   regional grid via both `interp` (bilinear) and `avg` (conservative-average), saved as
#   two separate files.
# - **Y, high-res target** — `hi` regridded to the fine 0.1-degree regional grid: the
#   "truth" half of a perfect-model pair, since X and Y both come from the same underlying
#   HR simulation.
#
# Differences from the existing pipeline, because of what this run actually is:
# - **Single realization, not a 10-member ensemble** — `ensemble` dim is length 1 on both
#   X and Y (kept on Y too, size 1, so `run_pipeline()`'s `(N, time, C, H, W)` shape
#   convention still holds).
# - **Wind (`u_10`/`v_10`) comes from the JRA55 atmospheric forcing files, not the ice
#   component.** The daily ice history (`day_1`/`h1`) does not carry `uatm`/`vatm` at
#   all — those diagnostics only exist in the *monthly* ice stream (`month_1`/`h`), which
#   is what the non-daily FOSI notebook (`build_X_Y_from_FOSI-HR.ipynb`) reads. At daily
#   frequency the true atmosphere state has to come from the forcing itself: one file per
#   calendar year on the TL319 Gaussian lat/lon grid, 3-hourly, resampled to a daily mean
#   aligned to hi/aice's day-end time labeling (see `process_wind`) before regridding with
#   its own `interp`/`avg` regridder pair built from the TL319 grid. Unlike the ice
#   component's grid-relative `uatm`/`vatm`, JRA55's `u_10`/`v_10` are already true
#   east/north — no rotation needed.
# - **Native ice grid pulled directly from the HR files**, not `pop_tools` — t13 isn't a
#   registered grid there, but every tseries file carries the full static grid.
# - **`run_name = "FOSI_HR_JRA55_daily"`**, distinct from the monthly notebook's
#   `"FOSI_HR_JRA55"`, so this saves to its own `X_FOSI_HR_JRA55_daily_*.nc` /
#   `Y_FOSI_HR_JRA55_daily.nc` instead of overwriting the monthly files that
#   `DEFAULT_X_PATH`/`DEFAULT_Y_PATH` in `functions_engressnet.py` currently point at.
#
# **Coastal-accuracy additions (carried over from the monthly notebook):**
# - **`skipna=True` on every regrid call** (`process_scalar`/`process_wind`) — the native
#   `hi`/`aice` fields are `NaN` over land, and without `skipna`, xESMF's regridding lets a
#   single land-side `NaN` poison the whole weighted average for any destination cell whose
#   interpolation stencil touches land -- i.e. every coastal ocean cell. The subsequent
#   `fillna(0)` was then silently zeroing out real near-coast ice thickness/concentration in
#   both X and Y, as if it were open water. `skipna=True` re-normalizes weights over only
#   the valid (ocean) source cells instead, so only genuinely land-locked destination cells
#   still come back `NaN` -> `0`.
# - **A 5th `X` channel, `ocean_frac`** — a static (time-invariant) ocean-fraction field,
#   regridded the same way as `hi`/`aice`/etc. from the native grid's own land/ocean split.
#   Lets the UNet's encoder see coastal geometry from its very first conv layer, in addition
#   to the separate high-res land mask already concatenated at the model's output.
#
# Run this on Casper/Derecho with `/glade/campaign` and `/glade/p/cesmdata` mounted.

import glob
import warnings
import numpy as np
import xesmf as xe
import xarray as xr
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")
warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS")

# ### 1. Collect files

RUN_DIR = Path(
    "/glade/campaign/cgd/oce/projects/FOSI_BGC/HR/g.e22.TL319_t13.G1850ECOIAF_JRA_HR.4p2z.001"
)
TSERIES_DIR = RUN_DIR / "ice" / "proc" / "tseries" / "day_1"
JRA55_DIR = Path("/glade/p/cesmdata/cseg/inputdata/ocn/jra55/v1.5_noleap")

# Distinct from the existing monthly run_name ("FOSI_HR_JRA55") so this daily build
# saves to its own files (X_FOSI_HR_JRA55_daily_*.nc / Y_FOSI_HR_JRA55_daily.nc) instead
# of silently overwriting the monthly data that DEFAULT_X_PATH/DEFAULT_Y_PATH in
# functions_engressnet.py currently point at.
run_name = "FOSI_HR_JRA55_daily"

# The *daily* ice stream (day_1/h1) carries these under their on-disk variable names
# with a "_d" suffix (hi_d, aice_d) -- unlike the *monthly* stream (month_1/h) the other
# FOSI notebook reads, where the same fields are stored as plain "hi"/"aice". Confirmed
# directly against a sample file; ds["hi"] raises KeyError on these daily files.
ice_vars = ["hi_d", "aice_d"]  # X predictors sourced from the ice component's own daily history
wind_vars = ["u_10", "v_10"]   # X predictors sourced from the JRA55 atmospheric forcing
target_var = "hi_d"            # Y predictand

# ---------- Ice component files (hi_d, aice_d) ----------
run_files = {}
for v in ice_vars:
    files = sorted(TSERIES_DIR.glob(f"*.cice.h1.{v}.*.nc"))
    run_files[v] = files
    print(f"{v}: {len(files)} files")

# hi_d/aice_d should cover the same record, or channels get silently truncated to the
# shortest one later -- check now rather than assume
lengths = {v: len(run_files[v]) for v in ice_vars}
if len(set(lengths.values())) > 1:
    print("WARNING: mismatched file counts across ice predictor variables:", lengths)
else:
    print("File counts match across ice predictor variables:", lengths)

# ---------- Wind forcing files (u_10, v_10) ----------
# The daily ice history (day_1/h1) does NOT carry uatm/vatm at all -- those diagnostics
# only exist in the *monthly* ice stream (month_1/h), which is what the non-daily FOSI
# notebook reads. So at daily frequency the true atmosphere state has to come from the
# JRA55 forcing itself: one file per calendar year, TL319 Gaussian lat/lon grid, 3-hourly.
wind_files = {v: {} for v in wind_vars}
for v in wind_vars:
    for f in sorted(JRA55_DIR.glob(f"*.{v}.TL319.*.nc")):
        year = int(f.name.split(".")[-3])
        wind_files[v][year] = f
    yrs = sorted(wind_files[v])
    print(f"{v}: {len(yrs)} yearly files, {yrs[0]}-{yrs[-1]}")

# Ice record's calendar years (from each hi_d file's start date), used to look up the
# matching JRA55 yearly file for each ice year below.
ice_years = [int(f.name.split(".")[-2][:4]) for f in run_files["hi_d"]]

missing = [y for y in ice_years if y not in wind_files["u_10"] or y not in wind_files["v_10"]]
if missing:
    raise RuntimeError(f"JRA55 forcing missing for ice years: {missing}")
print("JRA55 forcing covers every ice year:", ice_years[0], "-", ice_years[-1])

# ### 2. Native grid + regridders (t13)
#
# Same region as the existing pipeline (Kivalina domain) — keep `bbox_regrid`/`bbox` in
# sync with `3d_data_process.ipynb` if that domain ever changes.
#
# Three regridders, matching the three used in the original notebook for the high-res
# CESM LE member:
# - `regridder_hr_to_1deg_interp` — bilinear, for the X `interp` method
# - `regridder_hr_to_1deg_cons` — conservative-average, for the X `avg` method (needs exact
#   T-cell corners, built from U-points same as the original)
# - `regridder_hr_to_0p1deg` — bilinear, for the Y high-res target

# ---------- Region select (keep in sync with 3d_data_process.ipynb) ----------

bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360

bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360

# ---------- Native grid, pulled straight from an HR file ----------

ds_grid = xr.open_dataset(run_files["hi_d"][0])
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

# ---------- Conservative source grid (exact T-cell corners from U-points) ----------

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

# ---------- Destination grids ----------

dst_1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 1, 1.0)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 1, 1.0)),
})

dst_lat_c, dst_lon_c = dst_1deg.lat.values, dst_1deg.lon.values
dst_lat_b = np.concatenate([[dst_lat_c[0] - 0.5], dst_lat_c + 0.5])
dst_lon_b = np.concatenate([[dst_lon_c[0] - 0.5], dst_lon_c + 0.5])

dst_1deg_b = xr.Dataset({
    "lat": ("lat", dst_lat_c),
    "lon": ("lon", dst_lon_c),
    "lat_b": ("lat_b", dst_lat_b),
    "lon_b": ("lon_b", dst_lon_b),
})

dst_0p1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 0.1, 0.1)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 0.1, 0.1)),
})

# ---------- Regridders ----------

print("Building/locating regridders...")

WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

regridder_hr_to_1deg_interp = xe.Regridder(
    grid_ice_hr, dst_1deg, method="bilinear", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/hr_t13_to_1deg_interp.nc", reuse_weights=False,
)
print(" >> Built regridder_hr_to_1deg_interp.")

regridder_hr_to_1deg_cons = xe.Regridder(
    grid_ice_hr_conserv, dst_1deg_b, method="conservative", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/hr_t13_to_1deg_cons.nc", reuse_weights=False,
)
print(" >> Built regridder_hr_to_1deg_cons.")

regridder_hr_to_0p1deg = xe.Regridder(
    grid_ice_hr, dst_0p1deg, method="bilinear", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/hr_t13_to_0p1deg.nc", reuse_weights=False,
)
print(" >> Built regridder_hr_to_0p1deg.")

# ---------- JRA55 forcing grid + regridders (u_10/v_10 wind) ----------
# TL319 is a global Gaussian lat/lon grid (regular in longitude, non-uniform spacing in
# latitude), so unlike the ice t13 grid this is already "rectilinear" -- no curvilinear
# corner construction needed, just cell-bound midpoints for the conservative method.

def bounds_1d(centers):
    """Cell-edge bounds for a monotonic 1D array of cell-center coordinates."""
    centers = np.asarray(centers)
    mids = (centers[:-1] + centers[1:]) / 2
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])

_ds_jra = xr.open_dataset(next(iter(wind_files["u_10"].values())))
jra_lat_full = _ds_jra["latitude"].values
jra_lon_full = _ds_jra["longitude"].values
_ds_jra.close()

jra_lat_b_full = bounds_1d(jra_lat_full)
jra_lon_b_full = bounds_1d(jra_lon_full)

jlat_idx = np.where((jra_lat_full >= bbox_regrid["lat_min"]) & (jra_lat_full <= bbox_regrid["lat_max"]))[0]
jlon_idx = np.where((jra_lon_full >= lon_min_regrid) & (jra_lon_full <= lon_max_regrid))[0]
jr0, jr1 = jlat_idx.min(), jlat_idx.max()
jc0, jc1 = jlon_idx.min(), jlon_idx.max()

if not (np.array_equal(jlat_idx, np.arange(jr0, jr1 + 1)) and np.array_equal(jlon_idx, np.arange(jc0, jc1 + 1))):
    raise RuntimeError("JRA55 lat/lon mask isn't a contiguous block -- check bbox_regrid.")

grid_jra = xr.Dataset({
    "lat": ("latitude", jra_lat_full[jr0:jr1 + 1]),
    "lon": ("longitude", jra_lon_full[jc0:jc1 + 1]),
})

grid_jra_conserv = xr.Dataset({
    "lat": ("latitude", jra_lat_full[jr0:jr1 + 1]),
    "lon": ("longitude", jra_lon_full[jc0:jc1 + 1]),
    "lat_b": ("latitude_b", jra_lat_b_full[jr0:jr1 + 2]),
    "lon_b": ("longitude_b", jra_lon_b_full[jc0:jc1 + 2]),
})

print("JRA55 native grid prepared (subset to bbox_regrid).")

regridder_jra_to_1deg_interp = xe.Regridder(
    grid_jra, dst_1deg, method="bilinear", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/jra55_to_1deg_interp.nc", reuse_weights=False,
)
print(" >> Built regridder_jra_to_1deg_interp.")

regridder_jra_to_1deg_cons = xe.Regridder(
    grid_jra_conserv, dst_1deg_b, method="conservative", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/jra55_to_1deg_cons.nc", reuse_weights=False,
)
print(" >> Built regridder_jra_to_1deg_cons.")

# ### 3. Processing functions
#
# `process_scalar` regrids `hi`/`aice` from the native ice grid (takes a `regridder`
# argument so the same function serves `interp`, `avg`, and the high-res Y regrid).
# `process_wind` does the JRA55-grid equivalent for `u_10`/`v_10`: daily-mean, then regrid
# with its own `interp`/`avg` regridder pair.

def process_scalar(file, var, regridder):
    """hi or aice: mask to region, regrid, fill/cast.

    `skipna=True` re-normalizes regridding weights over only the non-NaN (ocean) source
    cells contributing to each destination cell, instead of letting a single land-side NaN
    (the native grid's fill value over land, decoded to NaN on load) poison the whole
    weighted average to NaN. Without this, every destination cell whose interpolation
    stencil touched land got zeroed out by `fillna(0)` below -- silently corrupting real
    near-coast ice thickness/concentration as if it were open water there, in both X and Y.
    Destination cells fully inside land (no ocean contribution at all) still come back NaN
    from the regridder and are correctly zero-filled.
    """
    ds = xr.open_dataset(file)
    da = ds[var].rename({"nj": "nlat", "ni": "nlon"})
    da = da.isel(nlat=mask_ice_hr)

    da_reg = regridder(da, skipna=True)
    da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    da_reg = da_reg.fillna(0).astype(np.float32)
    ds.close()
    return da_reg


def process_wind(year, var, regridder):
    """u_10 or v_10 for one calendar year: subset to region, daily-mean, regrid, fill/cast.

    JRA55 forcing is 3-hourly; hi/aice are daily means labeled at day-*end* (a file's
    "1958-01-02" timestamp is the mean over calendar day 1958-01-01 -- confirmed against
    its time_bounds). `resample(..., label="right", closed="left")` reproduces that same
    day-end labeling, so the result lines up exactly with the ice time coordinate one day
    at a time; the caller still verifies this explicitly with `.sel(time=...)` rather than
    assuming it, since a plain default-labeled resample would silently shift every wind
    sample by one day relative to hi/aice.
    """
    file = wind_files[var][year]
    ds = xr.open_dataset(file)
    da = ds[var].isel(latitude=slice(jr0, jr1 + 1), longitude=slice(jc0, jc1 + 1))
    da = da.resample(time="1D", label="right", closed="left").mean()

    da_reg = regridder(da, skipna=True)
    da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    da_reg = da_reg.fillna(0).astype(np.float32)
    ds.close()
    return da_reg

# ---------- Static ocean-fraction channel (for X) ----------
# Land points are NaN in the native hi_d field (CICE's land fill value, decoded to NaN on
# load); ocean points are real numbers even where there's no ice (hi_d/aice_d == 0).
# Land/ocean geography doesn't change in time, so a NaN/not-NaN split on a single
# timestep is already a clean, static land indicator -- no need to load all 64 files or
# pull in a separate pop_tools/KMT mask. Regridding it (bilinear or conservative,
# matching each X method, same regridders already built above) onto the 1-degree grid
# gives a continuous ocean-fraction channel, so the UNet's encoder sees coastal geometry
# from its very first conv layer instead of only via the separate high-res land mask
# concatenated at the model's output.

_ds0 = xr.open_dataset(run_files["hi_d"][0])
_hi0 = _ds0["hi_d"].rename({"nj": "nlat", "ni": "nlon"}).isel(nlat=mask_ice_hr)
if "time" in _hi0.dims:
    _hi0 = _hi0.isel(time=0)
ocean_native = (~np.isnan(_hi0)).astype(np.float32)
_ds0.close()

ocean_frac_by_method = {}
for method, regridder in {"interp": regridder_hr_to_1deg_interp, "avg": regridder_hr_to_1deg_cons}.items():
    ocean_frac = regridder(ocean_native)
    ocean_frac = ocean_frac.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    ocean_frac_by_method[method] = ocean_frac.astype(np.float32)

print("Ocean-fraction channel built for:", list(ocean_frac_by_method.keys()))
print("Range (interp):", float(ocean_frac_by_method["interp"].min()), "-", float(ocean_frac_by_method["interp"].max()))

# ### 4. Build X (both regrid methods)
#
# `hi`/`aice`/`u_10`/`v_10`/`ocean_frac` channel stack, run once per method, saved as two
# separate files: `X_FOSI_HR_JRA55_daily_interp.nc` / `X_FOSI_HR_JRA55_daily_avg.nc`.

regridders_ice = {
    "interp": regridder_hr_to_1deg_interp,
    "avg": regridder_hr_to_1deg_cons,
}
regridders_wind = {
    "interp": regridder_jra_to_1deg_interp,
    "avg": regridder_jra_to_1deg_cons,
}
processed_ice_vars = ["hi_d", "aice_d"]
channel_order = processed_ice_vars + wind_vars + ["ocean_frac"]

for method in regridders_ice:
    print(f"=== Building X, method={method} ===")

    ice_regridder = regridders_ice[method]
    wind_regridder = regridders_wind[method]

    channels = {}
    for var in processed_ice_vars:
        print(f"Processing {var}...")
        parts = [process_scalar(f, var, ice_regridder) for f in run_files[var]]
        channels[var] = xr.concat(parts, dim="time")

    for var in wind_vars:
        print(f"Processing {var}...")
        parts = [process_wind(year, var, wind_regridder) for year in ice_years]
        channels[var] = xr.concat(parts, dim="time")
        # Align to the ice time axis by actual calendar date rather than assuming
        # position matches -- raises immediately if any day is missing on either side
        # instead of silently shifting one predictor relative to the others.
        channels[var] = channels[var].sel(time=channels["hi_d"].time)

    min_t = min(channels[c].sizes["time"] for c in channel_order if c in channels)
    stacked = [channels[c].isel(time=slice(0, min_t)) for c in processed_ice_vars + wind_vars]

    # Static ocean-fraction channel, broadcast across the same time axis as the others.
    ocean_frac_t = ocean_frac_by_method[method].expand_dims(time=stacked[0]["time"])
    stacked.append(ocean_frac_t)

    X_ds = xr.concat(stacked, dim="channel")
    X_ds.name = "X"
    X_ds = X_ds.assign_coords(channel=channel_order)
    X_ds = X_ds.expand_dims({"ensemble": [0]})  # single realization, kept for shape parity

    X_ds.attrs["description"] = (
        f"Prescribed-atmosphere (JRA55-forced) t13 hindcast predictors, regridded to "
        f"the 1-degree regional grid via {method}. Single realization, not an ensemble."
    )
    X_ds.attrs["source_run"] = run_name
    X_ds.attrs["regrid_method"] = method
    X_ds.attrs["notes"] = (
        "hi_d/aice_d regridded from the native ice (t13) grid; u_10/v_10 (true "
        "east/north wind, not grid-relative) regridded from the JRA55 TL319 forcing "
        "grid after resampling 3-hourly -> daily-mean aligned to hi_d/aice_d's day-end "
        "time labeling. All regridded with skipna=True so coastal ocean cells are a "
        "proper ocean-only weighted average instead of being zeroed out by land-side "
        "NaN contamination; genuinely land-locked/no-data destination cells still come "
        "back NaN and are filled with zero."
    )
    X_ds.attrs["created_by"] = "Sky Gale"
    X_ds.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    X_ds.attrs["variables"] = (
        "hi_d: sea ice thickness (m); "
        "aice_d: sea ice concentration; "
        "u_10: JRA55 10m eastward wind (m s-1); "
        "v_10: JRA55 10m northward wind (m s-1); "
        "ocean_frac: static ocean fraction (1=ocean, 0=land), time-invariant"
    )

    X_ds = X_ds.transpose("ensemble", "time", "channel", "lat", "lon")

    save_path = f"/glade/derecho/scratch/skygale/Downscaling_Data/X_{run_name}_{method}.nc"
    X_ds.to_netcdf(save_path)
    print("Saved to:", save_path)

# ### 5. Build Y (high-res target)
#
# `hi` only, regridded to the fine 0.1-degree regional grid — the "truth" half of the
# perfect-model pair, saved as `Y_FOSI_HR_JRA55_daily.nc`.

print(f"=== Building Y, target_var={target_var} ===")

y_parts = [process_scalar(f, target_var, regridder_hr_to_0p1deg) for f in run_files[target_var]]
Y_da = xr.concat(y_parts, dim="time")
Y_da.name = "Y"

Y_ds = Y_da.expand_dims({"channel": [0]})
Y_ds = Y_ds.expand_dims({"ensemble": [0]})  # kept for shape parity with X (see run_pipeline)
Y_ds = Y_ds.transpose("ensemble", "time", "channel", "lat", "lon")

Y_ds.attrs["description"] = (
    "Prescribed-atmosphere (JRA55-forced) t13 hindcast, hi_d regridded to the fine "
    "0.1-degree regional grid -- perfect-model target paired with X_{run}_*.nc."
).format(run=run_name)
Y_ds.attrs["source_run"] = run_name
Y_ds.attrs["notes"] = "NaNs filled with zero."
Y_ds.attrs["created_by"] = "Sky Gale"
Y_ds.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
Y_ds.attrs["variables"] = "hi_d: sea ice thickness (m)"

save_path = f"/glade/derecho/scratch/skygale/Downscaling_Data/Y_{run_name}.nc"
Y_ds.to_netcdf(save_path)
print("Saved to:", save_path)

# ### 6. Quick checks
#
# Shapes against the existing MESACLIP tensors (same `channel`/spatial extent expected;
# `ensemble` differs: 1 vs 10), plus a tight-domain wind quiver check zoomed to the actual
# regional bbox so individual arrows are visible rather than rendering as solid texture.

X_interp = xr.open_dataset(f"/glade/derecho/scratch/skygale/Downscaling_Data/X_{run_name}_interp.nc").X
X_avg = xr.open_dataset(f"/glade/derecho/scratch/skygale/Downscaling_Data/X_{run_name}_avg.nc").X
Y_check = xr.open_dataset(f"/glade/derecho/scratch/skygale/Downscaling_Data/Y_{run_name}.nc").Y
print("X interp:", X_interp.shape, X_interp.channel.values)
print("X avg:   ", X_avg.shape, X_avg.channel.values)
print("Y:       ", Y_check.shape)

# Diagnostic-only: X/Y are already saved above, so a failure here (e.g. cartopy needing
# to fetch Natural Earth coastline data with no internet on a compute node) shouldn't
# fail a batch run of this notebook-as-script.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    t = 0
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": ccrs.NorthPolarStereo()})
    ax.set_extent([bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"]],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.coastlines()

    lon2d, lat2d = np.meshgrid(X_avg.lon.values, X_avg.lat.values)
    u = X_avg.sel(channel="u_10").isel(ensemble=0, time=t).values
    v = X_avg.sel(channel="v_10").isel(ensemble=0, time=t).values

    ax.quiver(lon2d, lat2d, u, v, transform=ccrs.PlateCarree())
    ax.set_title(f"JRA55 wind (true east/north, avg method), t={t}")

    fig_path = f"/glade/derecho/scratch/skygale/Downscaling_Data/{run_name}_wind_check.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print("Wind check figure saved to:", fig_path)
except Exception as exc:
    print(f"Skipping diagnostic wind plot (non-fatal): {exc}")

