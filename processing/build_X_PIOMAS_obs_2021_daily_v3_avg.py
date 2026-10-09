"""
build_X_PIOMAS_obs_2021_daily_v3_avg.py

Area-averaged ("avg", conservative) version of the 2021 PIOMAS observational-application
predictors, so they are coarsened the same way as the checkpoints' own DATA_VARIANT=avg training
data (X_FOSI_HR_JRA55_daily_avg.nc, where every channel is conservatively regridded). The v2
file (X_PIOMAS_obs_2021_daily_v2_interp.nc) is bilinear for all four channels.

Per channel:
- hi_d: PIOMAS daily hi, land-masked with PSC's io.dat_360_120.output, conservative regrid using
  T-cell corners from grid.dat.pop U-points (same construction as
  Version5/processing/build_X_Y_PIOMAS_obs_2020_daily_v3_conservative.py).
- aice_d: NSIDC/NOAA CDR v4 concentration, conservative regrid. Cell corners are exact: the CDR
  grid is NSIDC polar stereographic north (EPSG:3411, 25 km), so corners are xgrid/ygrid +-12.5 km
  inverse-projected (the projection reproduces the file's own latitude/longitude to ~1e-13 deg).
- u_10/v_10: JRA55 daily-mean winds, conservative regrid with cell bounds from bounds_1d, same as
  build_X_Y_from_FOSI-HR_daily.py's regridder_jra_to_1deg_cons. Daily averaging and time
  selection are unchanged from build_X_Y_PIOMAS_obs_2021_daily.py.

Every regrid uses skipna=True followed by fillna(0), matching the FOSI build. Y is unchanged:
Y_PIOMAS_obs_2021_daily_v2.nc (bilinear, the project's convention for the 0.1-degree target).

Output: X_PIOMAS_obs_2021_daily_v3_avg.nc in /glade/derecho/scratch/skygale/downscaling.
"""

import glob
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pyproj
import xarray as xr
import xesmf as xe

warnings.filterwarnings("ignore")

YEAR = int(os.environ.get("PIOMAS_YEAR", "2021"))  # 2026-09-24: year overridable; default unchanged
JRA55_FILENAME_SUFFIX = os.path.basename(glob.glob(f"/glade/p/cesmdata/cseg/inputdata/ocn/jra55/v1.5_noleap/JRA.v1.5.u_10.TL319.{YEAR}.*.nc")[0]).split(".")[-2]  # per-year date suffix (e.g. 220505 for 2021)
GRID_DAT_PATH = "/glade/campaign/cgd/ccr/yeager/OBS/seaice/PIOMAS/grid.dat"
UTIL_DIR = "/glade/derecho/scratch/skygale/observations/PIOMAS_daily/utilities"
GRID_DAT_POP_PATH = f"{UTIL_DIR}/grid.dat.pop"
IO_MASK_PATH = f"{UTIL_DIR}/io.dat_360_120.output"
PIOMAS_DAILY_PATH = f"/glade/derecho/scratch/skygale/observations/PIOMAS_daily/PIOMAS_hiday_{YEAR}.nc"
# 2026-09-29: CDR source overridable (the 2010-2023 build uses a v6 file converted to this layout by
# observations/processing/convert_cdr_v6_to_noleap.py); defaults unchanged.
CDR_PATH = os.environ.get("CDR_PATH", "/glade/campaign/cesm/development/pcwg/ssmi/CDR/cdr_seaice_conc_daily_nh.cdr.noleap.19790101-20211231.nc")
CDR_START = os.environ.get("CDR_START", "1979-01-01")
CDR_END = os.environ.get("CDR_END", "2021-12-31")
FILE_TAG = os.environ.get("FILE_TAG", "")  # 2026-09-29: e.g. "_cdrv6" so the 2010-2023 rebuild doesn't overwrite existing years
JRA55_DIR = "/glade/p/cesmdata/cseg/inputdata/ocn/jra55/v1.5_noleap"
V2_X_PATH = f"/glade/derecho/scratch/skygale/downscaling/X_PIOMAS_obs_{YEAR}{FILE_TAG}_daily_v2_interp.nc"
FOSI_AVG_X_PATH = "/glade/derecho/scratch/skygale/downscaling/X_FOSI_HR_JRA55_daily_avg.nc"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/projects/SeaIceDownscaling/weighted_grids"
OUT_DIR = "/glade/derecho/scratch/skygale/downscaling"

NLAT, NLON = 120, 360

bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360
bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360

dst_lat_c = np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 1, 1.0)
dst_lon_c = np.arange(lon_min_regrid, lon_max_regrid + 1, 1.0)
dst_1deg_b = xr.Dataset({
    "lat": ("lat", dst_lat_c), "lon": ("lon", dst_lon_c),
    "lat_b": ("lat_b", np.concatenate([[dst_lat_c[0] - 0.5], dst_lat_c + 0.5])),
    "lon_b": ("lon_b", np.concatenate([[dst_lon_c[0] - 0.5], dst_lon_c + 0.5])),
})


def crop(da):
    da = da.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    return da.fillna(0).astype(np.float32).reset_coords(drop=True)


# ---------------------------------------------------------------------------
# 1. hi_d: PIOMAS, land-masked, conservative from grid.dat.pop corners
# ---------------------------------------------------------------------------

grid_vals = np.loadtxt(GRID_DAT_PATH, dtype=np.float32).reshape(2, NLAT, NLON)
tlon_full = grid_vals[0] % 360
tlat_full = grid_vals[1]
pop_vals = np.loadtxt(GRID_DAT_POP_PATH).reshape(7, NLAT, NLON)
ulat_full = pop_vals[0].astype(np.float32)
ulon_full = pop_vals[1].astype(np.float32) % 360

mask_piomas = np.any(
    (tlat_full >= bbox_regrid["lat_min"]) & (tlat_full <= bbox_regrid["lat_max"])
    & (tlon_full >= lon_min_regrid) & (tlon_full <= lon_max_regrid),
    axis=1,
)
rows = np.where(mask_piomas)[0]
r0, r1 = rows.min(), rows.max()
if not np.array_equal(rows, np.arange(r0, r1 + 1)):
    raise RuntimeError("PIOMAS rows in bbox_regrid are not contiguous -- corner construction assumes this.")
if r0 < 1 or r1 + 1 > NLAT - 1:
    raise RuntimeError("PIOMAS bbox rows touch the native grid edge/pole fold -- not handled.")

ulon_i = np.pad(ulon_full, ((0, 0), (1, 0)), mode="wrap")
ulat_i = np.pad(ulat_full, ((0, 0), (1, 0)), mode="wrap")
grid_piomas_conserv = xr.Dataset({
    "lat": (["nlat", "nlon"], tlat_full[mask_piomas]),
    "lon": (["nlat", "nlon"], tlon_full[mask_piomas]),
    "lat_b": (["nlat_b", "nlon_b"], ulat_i[r0 - 1:r1 + 1, :]),
    "lon_b": (["nlat_b", "nlon_b"], ulon_i[r0 - 1:r1 + 1, :]),
})
regridder_piomas = xe.Regridder(
    grid_piomas_conserv, dst_1deg_b, method="conservative", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/piomas_to_1deg_conservative_2021build.nc", reuse_weights=False,
)
print(" >> Built PIOMAS conservative regridder.")

io_rows = []
with open(IO_MASK_PATH) as f:
    for line in f:
        line = line.rstrip("\n")
        io_rows.append([int(line[i:i + 2]) for i in range(0, len(line), 2)])
land_native = np.array(io_rows) == 0

ds_daily = xr.open_dataset(PIOMAS_DAILY_PATH)
if not (np.allclose(ds_daily["lat"].values, tlat_full, atol=1e-3)
        and np.allclose(ds_daily["lon"].values % 360, tlon_full, atol=1e-3)):
    raise RuntimeError("PIOMAS daily file's lat/lon don't match grid.dat T-points.")
hi_da = ds_daily["hi"].where(~xr.DataArray(land_native, dims=["nlat", "nlon"])).isel(nlat=mask_piomas)
daily_time = ds_daily["time"].values
print(f"Regridding PIOMAS hi -> 1deg (conservative), {daily_time.size} days...")
hi_1deg_raw = regridder_piomas(hi_da, skipna=True)
ds_daily.close()

# Optional coastal fill (2026-09-25, PIOMAS_COAST_FILL=1 -> writes ..._v4_avg_coastfill.nc instead of v3):
# PIOMAS's native grid treats parts of the Alaskan/Chukotka coast as land (io.dat), so ~36 of the
# 376 FOSI-ocean 1-degree cells in the training crop (Norton/Kotzebue Sound, Kivalina, Shishmaref,
# Chukotka) have no PIOMAS ocean under them. The regrid returns NaN there and crop()'s fillna(0)
# wrote hi_d = 0 m under ~98% CDR concentration -- a combination FOSI training never contains, so
# the model erased coastal ice (Nome 0.02-0.05 m vs truth 0.62). Here those cells -- NaN after the
# regrid but ocean in FOSI (FOSI aice ever > 0) -- are filled per day with the mean of valid
# neighbouring cells (3x3), iterated until none remain. Cells that are land in both stay 0.
COAST_FILL = os.environ.get("PIOMAS_COAST_FILL", "0") == "1"
if COAST_FILL:
    fosi_aice = xr.open_dataset(FOSI_AVG_X_PATH)["X"].isel(ensemble=0).sel(channel="aice_d")
    fosi_ocean = (fosi_aice.max("time") > 0).reset_coords(drop=True)
    hi_r = hi_1deg_raw.transpose("time", "lat", "lon")
    # FOSI X covers lat 60-80 / lon 170-220 only; the larger PIOMAS regrid box outside it is never
    # a fill target (it's outside every training crop anyway).
    fosi_ocean = fosi_ocean.astype(float).reindex_like(hi_r.isel(time=0), method="nearest", tolerance=1e-3,
                                                       fill_value=0.0).fillna(0.0)
    if int(fosi_ocean.sum()) < 300:
        raise RuntimeError("FOSI ocean mask didn't align with the PIOMAS regrid grid.")
    target = (fosi_ocean.values.astype(bool)) & np.isnan(hi_r.values).all(axis=0)
    if not np.array_equal(np.isnan(hi_r.values).any(axis=0), np.isnan(hi_r.values).all(axis=0)):
        raise RuntimeError("PIOMAS NaN pattern varies by day -- fill assumes a static land mask.")
    vals = hi_r.values.copy()
    todo = target.copy()
    for it in range(50):
        if not todo.any():
            break
        valid = ~np.isnan(vals)
        v0 = np.where(valid, vals, 0.0)
        pad_v = np.pad(v0, ((0, 0), (1, 1), (1, 1)))
        pad_n = np.pad(valid.astype(float), ((0, 0), (1, 1), (1, 1)))
        H, W = vals.shape[1:]
        s = sum(pad_v[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        n = sum(pad_n[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        fillable = todo & (n[0] > 0)
        vals[:, fillable] = (s / np.maximum(n, 1))[:, fillable]
        todo &= ~fillable
    else:
        raise RuntimeError("Coastal fill didn't converge in 50 iterations.")
    print(f" >> Coastal fill: {int(target.sum())} PIOMAS-land/FOSI-ocean cells in the regrid box filled "
          f"from neighbours in {it} iteration(s); {int(todo.sum())} unreachable.")
    hi_1deg_raw = hi_r.copy(data=vals)
hi_1deg = crop(hi_1deg_raw)

# ---------------------------------------------------------------------------
# 2. aice_d: CDR, conservative from exact EPSG:3411 cell corners
# ---------------------------------------------------------------------------

ds_cdr = xr.open_dataset(CDR_PATH, chunks={"time": 365})
cdr_dates = xr.date_range(start=CDR_START, end=CDR_END, freq="D", calendar="noleap", use_cftime=True)
ds_cdr = ds_cdr.assign_coords(time=cdr_dates)
cdr_lat = ds_cdr["latitude"].values
cdr_lon = ds_cdr["longitude"].values % 360
in_box = ((cdr_lat >= bbox_regrid["lat_min"] - 2) & (cdr_lat <= bbox_regrid["lat_max"] + 2)
          & (cdr_lon >= lon_min_regrid - 4) & (cdr_lon <= lon_max_regrid + 4))
jj, ii = np.where(in_box)
j0, j1, i0, i1 = jj.min(), jj.max(), ii.min(), ii.max()

x = ds_cdr["xgrid"].values[i0:i1 + 1].astype(np.float64)
y = ds_cdr["ygrid"].values[j0:j1 + 1].astype(np.float64)
dx, dy = np.diff(x).mean(), np.diff(y).mean()
x_b = np.concatenate([x - dx / 2, [x[-1] + dx / 2]])
y_b = np.concatenate([y - dy / 2, [y[-1] + dy / 2]])
proj = pyproj.Proj("epsg:3411")
XB, YB = np.meshgrid(x_b, y_b)
lon_b_cdr, lat_b_cdr = proj(XB, YB, inverse=True)
XC, YC = np.meshgrid(x, y)
lon_c_cdr, lat_c_cdr = proj(XC, YC, inverse=True)
if np.nanmax(np.abs(lat_c_cdr - cdr_lat[j0:j1 + 1, i0:i1 + 1])) > 1e-6:
    raise RuntimeError("EPSG:3411 centers don't reproduce the CDR file's latitudes.")

grid_cdr_conserv = xr.Dataset({
    "lat": (["nlat", "nlon"], lat_c_cdr), "lon": (["nlat", "nlon"], lon_c_cdr % 360),
    "lat_b": (["nlat_b", "nlon_b"], lat_b_cdr), "lon_b": (["nlat_b", "nlon_b"], lon_b_cdr % 360),
})
regridder_cdr = xe.Regridder(
    grid_cdr_conserv, dst_1deg_b, method="conservative", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/cdr_to_1deg_conservative_2021build.nc", reuse_weights=False,
)
print(" >> Built CDR conservative regridder.")

sic = (ds_cdr["cdr_seaice_conc_daily"].sel(time=daily_time)
       .isel(jdim=slice(j0, j1 + 1), idim=slice(i0, i1 + 1))
       .rename({"jdim": "nlat", "idim": "nlon"}).drop_vars(["nlat", "nlon"], errors="ignore"))
if float(sic.max()) > 100:
    raise RuntimeError("CDR values > 100 (flag values) inside the regrid box -- mask them first.")
print("Regridding CDR concentration -> 1deg (conservative)...")
aice_1deg = crop(regridder_cdr(sic.compute(), skipna=True) / 100.0)
ds_cdr.close()

# ---------------------------------------------------------------------------
# 3. u_10/v_10: JRA55, conservative (same regridder construction as the FOSI build)
# ---------------------------------------------------------------------------


def bounds_1d(centers):
    centers = np.asarray(centers)
    mids = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[centers[0] - (mids[0] - centers[0])], mids, [centers[-1] + (centers[-1] - mids[-1])]])


_ds = xr.open_dataset(f"{JRA55_DIR}/JRA.v1.5.u_10.TL319.{YEAR}.{JRA55_FILENAME_SUFFIX}.nc")
jra_lat, jra_lon = _ds["latitude"].values, _ds["longitude"].values
_ds.close()
jlat = np.where((jra_lat >= bbox_regrid["lat_min"]) & (jra_lat <= bbox_regrid["lat_max"]))[0]
jlon = np.where((jra_lon >= lon_min_regrid) & (jra_lon <= lon_max_regrid))[0]
jr0, jr1, jc0, jc1 = jlat.min(), jlat.max(), jlon.min(), jlon.max()
grid_jra_conserv = xr.Dataset({
    "lat": ("latitude", jra_lat[jr0:jr1 + 1]), "lon": ("longitude", jra_lon[jc0:jc1 + 1]),
    "lat_b": ("latitude_b", bounds_1d(jra_lat)[jr0:jr1 + 2]),
    "lon_b": ("longitude_b", bounds_1d(jra_lon)[jc0:jc1 + 2]),
})
regridder_jra = xe.Regridder(
    grid_jra_conserv, dst_1deg_b, method="conservative", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/jra55_to_1deg_conservative_2021build.nc", reuse_weights=False,
)
print(" >> Built JRA55 conservative regridder.")


def process_wind(var):
    ds = xr.open_dataset(f"{JRA55_DIR}/JRA.v1.5.{var}.TL319.{YEAR}.{JRA55_FILENAME_SUFFIX}.nc")
    da = ds[var].isel(latitude=slice(jr0, jr1 + 1), longitude=slice(jc0, jc1 + 1))
    da = da.resample(time="1D").mean()
    out = crop(regridder_jra(da, skipna=True)).sel(time=daily_time)
    ds.close()
    return out


print("Regridding JRA55 winds -> 1deg (conservative)...")
u_1deg = process_wind("u_10")
v_1deg = process_wind("v_10")

# ---------------------------------------------------------------------------
# 4. Assemble X on the v2 file's exact coordinates, save, check
# ---------------------------------------------------------------------------

X_v2 = xr.open_dataset(V2_X_PATH)["X"]
channel_order = list(X_v2.channel.values)
if channel_order != ["hi_d", "aice_d", "u_10", "v_10"]:
    raise RuntimeError(f"Unexpected channel order in v2 X: {channel_order}")
new = {"hi_d": hi_1deg, "aice_d": aice_1deg, "u_10": u_1deg, "v_10": v_1deg}
X_new = X_v2.copy(deep=True)
for c, da in new.items():
    da = da.transpose("time", "lat", "lon")
    if not (np.allclose(da.lat.values, X_v2.lat.values) and np.allclose(da.lon.values, X_v2.lon.values)
            and da.sizes["time"] == X_v2.sizes["time"]):
        raise RuntimeError(f"{c}: grid/time mismatch against v2 X.")
    X_new.values[:, :, channel_order.index(c), :, :] = da.values[None]
if np.isnan(X_new.values).any():
    raise RuntimeError("NaN in the assembled X.")

X_new.attrs["description"] = (
    "Area-averaged (conservative) version of X_PIOMAS_obs_2021_daily_v2_interp.nc: all four "
    "channels (PIOMAS hi_d, CDR aice_d, JRA55 u_10/v_10) conservatively regridded to 1 degree, "
    "matching the DATA_VARIANT=avg training data. See build_X_PIOMAS_obs_2021_daily_v3_avg.py."
)
X_new.attrs["regrid_method"] = "avg (conservative)"
X_new.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
save_path = (f"{OUT_DIR}/X_PIOMAS_obs_{YEAR}{FILE_TAG}_daily_v4_avg_coastfill.nc" if COAST_FILL
             else f"{OUT_DIR}/X_PIOMAS_obs_{YEAR}{FILE_TAG}_daily_v3_avg.nc")
if COAST_FILL:
    X_new.attrs["description"] += (" v4: PIOMAS-land/FOSI-ocean coastal cells in hi_d filled from neighbouring "
                                   "PIOMAS cells instead of 0 (PIOMAS_COAST_FILL=1).")
X_new.to_dataset(name="X").to_netcdf(save_path)
print("Saved X to:", save_path)

X_chk = xr.open_dataset(save_path)["X"]
X_fosi = xr.open_dataset(FOSI_AVG_X_PATH)["X"]
if not (np.allclose(X_chk.lat.values, X_fosi.lat.values) and np.allclose(X_chk.lon.values, X_fosi.lon.values)):
    raise RuntimeError("Grid mismatch against X_FOSI_HR_JRA55_daily_avg.nc.")
print("X low-res grid matches FOSI avg: True")
for c in channel_order:
    a, b = X_chk.sel(channel=c).values, X_v2.sel(channel=c).values
    print(f"{c}: v3 mean={a.mean():.4f} range=[{a.min():.3f},{a.max():.3f}] | "
          f"v2 mean={b.mean():.4f} | mean |v3-v2|={np.abs(a - b).mean():.4f}")
print("Y unchanged: use Y_PIOMAS_obs_2021_daily_v2.nc.")
print("Done.")
