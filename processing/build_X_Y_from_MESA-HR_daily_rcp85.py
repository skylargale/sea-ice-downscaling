"""RCP8.5 continuation of build_X_Y_from_MESA-HR_daily.py.

Builds X_MESA_HR_RCP85_daily_{interp,avg}.nc / Y_MESA_HR_RCP85_daily.nc, the
2006-onward scenario continuation of the HIST daily MESA-HR pair, so the
combined HIST+RCP85 record can cover training windows/test years past 2006
(the HIST daily file's hard stop).

Differences from the HIST script (not a hand-edit of it -- the HIST script is
still auto-generated from its notebook; this is a standalone twin):

1. Reads from d651009 (BRCP85C5 ne120_t12 high-res) instead of d651007
   (BHISTC5). d651030/BRCP85 (the true low-res physical member) is NOT read
   at all, matching the HIST script's actual behavior: X is built from
   `coarsen_files` (the high-res member's own hi_d/aice_d/U10, coarsened to
   1deg), not from `low_res_files` (computed in the HIST script but never
   used to build X -- confirmed by reading the HIST script's build-X loop,
   which iterates `coarsen_files`).

2. Member availability differs from HIST and further restricts the usable
   set. Checked directly against the actual files on disk (2026-08-06):
     - .002, .003: no ice/proc/tseries/day_1 files at all for RCP85 (only
       monthly ice output) -- excluded.
     - sehires38.001: same as HIST, no daily ice -- excluded (as in HIST).
     - .009: has daily ice but zero atm/proc/tseries/day_1 U10 files
       (only FLUT/IVT/OMEGA850/PRECT/RHREFHT/TMQ/TS/Z500 at daily
       frequency, no U10) -- excluded.
   Usable RCP85 members: .004, .005, .006, .007, .008, .010 (6 members),
   vs. HIST's 9 (.002-.010). The combined HIST+RCP85 stitching step
   (build_X_Y_from_MESA-HR_daily_stitch.py) uses this same 6-member subset
   for the HIST portion too, so every training window in the sweep draws
   from a consistent ensemble rather than 9 members pre-2006 and 6 after.

3. `collect_files` gains an `end_year` cap. Each RCP8.5 file is a 10-year
   chunk (e.g. `...20060101-20160101.nc`, `...20160102-20260101.nc`, ...,
   out to 2100). The sweep only needs through the 2021 test year, so this
   caps at 2026 (i.e. keeps the 2006-2016 and 2016-2026 chunks only) to
   avoid regridding ~80 unneeded years per member/variable.

4. Same regridders (same bbox/bbox_regrid/dst grids, same physical domain)
   are reused via `reuse_weights=True` against the exact same
   WEIGHTED_GRIDS_DIR cache files the HIST script wrote -- no need to
   rebuild the ice/atm regridding weight matrices.
"""

import glob
import torch
import warnings
import pop_tools
import numpy as np
import xesmf as xe
import xarray as xr
from datetime import datetime, timezone

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")
warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ### 1. Collect files

EXCLUDED_MEMBER_TAGS = ["sehires38", ".30-2006-2100.002", ".31-2006-2100.003", "-2006-2100.009"]


def collect_files(dirs, vars, start_year, end_year=None):
    out = []
    for d in dirs:
        member_files = {}
        for v in vars:
            c = comps[v]
            pattern = f"{d}/{c}/proc/tseries/day_1/*.{v}.*.nc"
            files = sorted(glob.glob(pattern))

            filtered = []
            for f in files:
                year = int(f.split('.')[-2][:4])
                if year >= start_year and (end_year is None or year <= end_year):
                    filtered.append(f)

            member_files[v] = filtered

        out.append(member_files)
    return out


# ---------- Directories on glade ----------

high_res_dirs = sorted(
    glob.glob('/glade/campaign/collections/gdex/data/d651009/b.e13.*')
)

high_res_dirs = [d for d in high_res_dirs if not any(tag in d for tag in EXCLUDED_MEMBER_TAGS)]

print("Usable RCP8.5 high-res members:")
for d in high_res_dirs:
    print(" ", d)

# ---------- Variables ----------

low_vars = ['hi_d', 'aice_d', 'U10']
comps = {
    'hi_d': 'ice',
    'aice_d': 'ice',
    'U10': 'atm',
}

target_var = ['hi_d']

# ---------- Collect files ----------
# ice (hi_d/aice_d) files are 10-year chunks named by start year
# (20060101-20160101, 20160102-20260101, ...); atm (U10) files are 5-year
# chunks (20060101-20101231, 20110101-20151231, ...). end_year=2021 keeps
# exactly the chunks needed to cover 2006 through the 2021 test year for
# both chunkings (2 ice chunks, 4 atm chunks) without pulling in a chunk
# that starts after 2021 (verified directly against the file listing).

high_res_files = collect_files(high_res_dirs, target_var, start_year=2006, end_year=2021)
coarsen_files = collect_files(high_res_dirs, low_vars, start_year=2006, end_year=2021)

print('High-res | # ens:', len(high_res_files), '| # vars:', len(high_res_files[0]))
print('Coarsen  | # ens:', len(coarsen_files), '| # vars:', len(coarsen_files[0]))

# ### 2. Set up rectilinear grids and regional subsetting (identical to the
# HIST script -- same physical domain, so the same cached regridder weights
# apply unchanged)

bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360

bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360
print("Region select done.")

atm_dir = "/glade/p/cesmdata/cseg/inputdata/share/scripgrids/"
nat_atm_hr = xr.open_dataset(atm_dir + "ne120np4_pentagons_100310.nc")

atm_lat = nat_atm_hr.grid_center_lat.values
atm_lon = nat_atm_hr.grid_center_lon.values % 360

mask_atm_hr = (
    (atm_lat >= bbox_regrid["lat_min"])
    & (atm_lat <= bbox_regrid["lat_max"])
    & (atm_lon >= lon_min_regrid)
    & (atm_lon <= lon_max_regrid)
)

grid_atm_hr = xr.Dataset({
    "lat": ("ncol", nat_atm_hr.grid_center_lat.values[mask_atm_hr]),
    "lon": ("ncol", nat_atm_hr.grid_center_lon.values[mask_atm_hr] % 360),
})

nat_ice_hr = pop_tools.get_grid("POP_tx0.1v2")

ice_lon = nat_ice_hr.TLONG % 360

mask_ice_hr = np.any(
    ((nat_ice_hr.TLAT >= bbox_regrid["lat_min"])
     & (nat_ice_hr.TLAT <= bbox_regrid["lat_max"])
     & (ice_lon >= lon_min_regrid)
     & (ice_lon <= lon_max_regrid)),
    axis=1,
)

grid_ice_hr = xr.Dataset({
    "lat": (["nlat", "nlon"], nat_ice_hr.TLAT.isel(nlat=mask_ice_hr).values),
    "lon": (["nlat", "nlon"], ice_lon.isel(nlat=mask_ice_hr).values),
})

print("Native grids prepared.")

dst_1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 1, 1.0)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 1, 1.0)),
})

dst_0p1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 0.1, 0.1)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 0.1, 0.1)),
})

print("Destination grids set up.")

# ---------- Build regridders (reuse the HIST script's cached weight files --
# same source/dest grids, so the weight matrices are identical) ----------

print("Loading cached regridders...")

WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

regridder_ice_to_1deg_interp = xe.Regridder(
    grid_ice_hr,
    dst_1deg,
    method="bilinear",
    periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/ice_hr_to_1deg_interp.nc",
    reuse_weights=True,
)
print(" >> Loaded regridder_ice_to_1deg_interp.")

# Conservative ("avg") ice regridder needs the same exact-corner source grid
# the HIST script built. Reconstruct it here (cheap, no I/O beyond the POP
# grid already loaded above) so xe.Regridder's shape check against the
# cached weight file passes.
has_upoints = hasattr(nat_ice_hr, "ULONG") and hasattr(nat_ice_hr, "ULAT")
if not has_upoints:
    raise RuntimeError("Expected ULONG/ULAT on the POP grid object for exact ice cell corners.")

tlon_full = (nat_ice_hr.TLONG.values % 360)
tlat_full = nat_ice_hr.TLAT.values
ulon_full = (nat_ice_hr.ULONG.values % 360)
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

dst_lat_c = dst_1deg.lat.values
dst_lon_c = dst_1deg.lon.values
dst_lat_b = np.concatenate([[dst_lat_c[0] - 0.5], dst_lat_c + 0.5])
dst_lon_b = np.concatenate([[dst_lon_c[0] - 0.5], dst_lon_c + 0.5])

dst_1deg_b = xr.Dataset({
    "lat": ("lat", dst_lat_c),
    "lon": ("lon", dst_lon_c),
    "lat_b": ("lat_b", dst_lat_b),
    "lon_b": ("lon_b", dst_lon_b),
})

regridder_ice_to_1deg_cons = xe.Regridder(
    grid_ice_hr_conserv,
    dst_1deg_b,
    method="conservative",
    periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/ice_hr_to_1deg_cons.nc",
    reuse_weights=True,
)
print(" >> Loaded regridder_ice_to_1deg_cons.")

regridder_ice_to_0p1deg = xe.Regridder(
    grid_ice_hr,
    dst_0p1deg,
    method="bilinear",
    periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/ice_hr_to_0p1deg_kivalina_masked.nc",
    reuse_weights=True,
)
print(" >> Loaded regridder_ice_to_0p1deg.")

regridder_atm_to_1deg_interp = xe.Regridder(
    grid_atm_hr,
    dst_1deg,
    method="nearest_s2d",
    locstream_in=True,
    periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/atm_hr_to_1deg_interp.nc",
    reuse_weights=True,
)
print(" >> Loaded regridder_atm_to_1deg_interp.")


def atm_bin_average(da, native_lat, native_lon, dst_lat, dst_lon):
    lat_step = float(dst_lat[1] - dst_lat[0])
    lon_step = float(dst_lon[1] - dst_lon[0])
    lat_edges = np.concatenate([[dst_lat[0] - lat_step / 2], dst_lat[:-1] + lat_step / 2, [dst_lat[-1] + lat_step / 2]])
    lon_edges = np.concatenate([[dst_lon[0] - lon_step / 2], dst_lon[:-1] + lon_step / 2, [dst_lon[-1] + lon_step / 2]])

    lat_idx = np.clip(np.digitize(native_lat, lat_edges) - 1, 0, len(dst_lat) - 1)
    lon_idx = np.clip(np.digitize(native_lon, lon_edges) - 1, 0, len(dst_lon) - 1)
    bin_id = lat_idx * len(dst_lon) + lon_idx

    da = da.assign_coords(bin_id=("ncol", bin_id))
    binned_mean = da.groupby("bin_id").mean()

    n_bins = len(dst_lat) * len(dst_lon)
    full = np.full(da.shape[:-1] + (n_bins,), np.nan, dtype=np.float32)
    full[..., binned_mean["bin_id"].values] = binned_mean.values

    out = full.reshape(da.shape[:-1] + (len(dst_lat), len(dst_lon)))
    dims = list(da.dims[:-1]) + ["lat", "lon"]
    coords = {d: da.coords[d] for d in da.dims[:-1] if d in da.coords}
    coords["lat"] = dst_lat
    coords["lon"] = dst_lon
    return xr.DataArray(out, dims=dims, coords=coords)


def process_file(args):
    file, var = args
    ds = xr.open_dataset(file)

    if var in ice_vars:
        da = ds[var].rename({"nj": "nlat", "ni": "nlon"})
        da = da.isel(nlat=mask_ice_hr)

        if method == "interp":
            da_reg = regridder_ice_to_1deg_interp(da, skipna=True)
        elif method == "avg":
            da_reg = regridder_ice_to_1deg_cons(da, skipna=True)
        else:
            ds.close()
            raise ValueError(f"Unknown method: {method}")

        da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
        da_reg = da_reg.fillna(0).astype(np.float32)
        ds.close()
        return da_reg

    elif var in atm_vars:
        da = ds[var]
        if "lev" in da.dims:
            da = da.isel(lev=-1, drop=True)
        da = da.load()
        da = da.isel(ncol=mask_atm_hr)

        if method == "interp":
            da_reg = regridder_atm_to_1deg_interp(da, skipna=True)
        elif method == "avg":
            da_reg = atm_bin_average(
                da,
                nat_atm_hr.grid_center_lat.values[mask_atm_hr],
                nat_atm_hr.grid_center_lon.values[mask_atm_hr] % 360,
                dst_1deg.lat.values,
                dst_1deg.lon.values,
            )
        else:
            ds.close()
            raise ValueError(f"Unknown method: {method}")

        da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
        da_reg = da_reg.fillna(0).astype(np.float32)
        ds.close()
        return da_reg

    else:
        ds.close()
        raise ValueError(f"Unknown variable: {var}")


def load_wrapper(files, var):
    args = [(f, var) for f in files]
    data_list = [process_file(arg) for arg in args]
    out = xr.concat(data_list, dim="time")
    return out


# ---------- Build X (Low-Res Predictors, coarsened from the same high-res
# member -- perfect-model style, matching the HIST script's actual
# behavior) ----------

ice_vars, atm_vars = ["hi_d", "aice_d"], ["U10"]

for method in ("interp", "avg"):
    X_list = []

    for i, member in enumerate(coarsen_files):
        print(f"Processing Ensemble #{i+1}, method={method}...")
        channels = [load_wrapper(member[var], var) for var in low_vars]
        print(' >> All channels regridded and loaded.')

        min_t = min(c.sizes["time"] for c in channels)
        channels = [c.isel(time=slice(0, min_t)) for c in channels]

        member_da = xr.concat(channels, dim="channel")
        member_da.name = "X"
        member_da = member_da.expand_dims({"ensemble": [i]})

        X_list.append(member_da)

    min_t_global = min(x.sizes["time"] for x in X_list)
    X_list = [x.isel(time=slice(0, min_t_global)) for x in X_list]

    X_ds = xr.concat(X_list, dim="ensemble")

    if ("channel" in X_ds.dims and "channel" not in X_ds.coords):
        X_ds = X_ds.assign_coords(channel=np.arange(X_ds.sizes["channel"]))

    X_ds = X_ds.assign_coords(channel=low_vars)

    X_ds.attrs["description"] = "Low-resolution MESACLIP RCP8.5 predictors on a common 1-degree rectilinear grid."
    X_ds.attrs["notes"] = f"Regridding method: {method}; NaNs filled with zero for ML pipeline. RCP8.5 2006-2026 subset (only through 2021 actually needed)."
    X_ds.attrs["created_by"] = "Sky Gale"
    X_ds.attrs["date_created"] = (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    X_ds.attrs["variables"] = (
        "hi_d: sea ice thickness (m); "
        "aice_d: sea ice concentration (%); "
        "U10: 10m wind speed (m s-1)"
    )
    X_ds.attrs["ensemble_members"] = ", ".join(high_res_dirs)

    X_ds = X_ds.transpose("ensemble", "time", "channel", "lat", "lon")

    save_path = f"/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_RCP85_daily_{method}.nc"
    X_ds.to_netcdf(save_path)
    print("\nSaved to:", save_path)

# ### Build Y (High-Res Predictand)


def process_file_hi(args):
    file, target_var = args
    ds = xr.open_dataset(file)
    da = ds[target_var[0]]
    da = da.rename({"nj": "nlat", "ni": "nlon"})
    da = da.isel(nlat=mask_ice_hr)

    da_reg = regridder_ice_to_0p1deg(da, skipna=True)
    da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
    da_reg = da_reg.fillna(0).astype(np.float32)
    ds.close()
    return da_reg


def load_wrapper_hi(files, var):
    args = [(f, var) for f in files]
    data_list = [process_file_hi(arg) for arg in args]
    out = xr.concat(data_list, dim="time")
    return out


Y_list = []

for i, member in enumerate(high_res_files):
    print(f"Processing Ensemble #{i+1}...")
    var_da = load_wrapper_hi(member[target_var[0]], target_var)
    var_da.name = "Y"
    member_da = var_da.expand_dims({"ensemble": [i], "channel": [0]})
    member_da = member_da.transpose("ensemble", "time", "channel", "lat", "lon")
    Y_list.append(member_da)

Y_ds = xr.concat(Y_list, dim="ensemble")

if "channel" in Y_ds.dims and "channel" not in Y_ds.coords:
    Y_ds = Y_ds.assign_coords(channel=np.arange(Y_ds.sizes["channel"]))

Y_ds.attrs["description"] = "High-resolution daily MESACLIP RCP8.5 regridded to a regional 0.1-degree rectilinear grid."
Y_ds.attrs["notes"] = "NaNs filled with zero. RCP8.5 2006-2026 subset (only through 2021 actually needed)."
Y_ds.attrs["created_by"] = "Sky Gale"
Y_ds.attrs["date_created"] = (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
Y_ds.attrs["variables"] = "hi: sea ice thickness (m)"
Y_ds.attrs["ensemble_members"] = ", ".join(high_res_dirs)

save_path = "/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_RCP85_daily.nc"
Y_ds.to_netcdf(save_path)
print("\nSaved to:", save_path)

# ### Check shapes

X1 = xr.open_dataset('/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_RCP85_daily_interp.nc').X
X2 = xr.open_dataset('/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_RCP85_daily_avg.nc').X
Y = xr.open_dataset('/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_RCP85_daily.nc').Y

print(X1.shape, X2.shape, Y.shape)
print("Time range:", str(Y.time.values[0]), "to", str(Y.time.values[-1]))
