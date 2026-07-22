"""Auto-generated from build_X_Y_from_MESA-HR_daily.ipynb. Do not hand-edit; edit the notebook and regenerate."""

import glob
import torch
import warnings
import pop_tools
import numpy as np
import xesmf as xe
import xarray as xr
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")
warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ### 1. Collect files

def collect_files(dirs, vars, start_year):
    out = []
    for d in dirs:
        member_files = {}
        for v in vars:
            c = comps[v]
            pattern = f"{d}/{c}/proc/tseries/day_1/*.{v}.*.nc"
            files = sorted(glob.glob(pattern))

            # Keep only files from 1920/2006 on
            filtered = []
            for f in files:
                year = int(f.split('.')[-2][:4])
                if year >= start_year:
                    filtered.append(f)

            member_files[v] = filtered

        out.append(member_files)
    return out


# ---------- Directories on glade ----------

low_res_dirs = sorted(
    glob.glob('/glade/campaign/collections/gdex/data/d651030/BHIST/*')  # /BHIST or /BRCP85
)
high_res_dirs = sorted(
    glob.glob('/glade/campaign/collections/gdex/data/d651007/b.e13.*')  # /d651007 or /d651009
)

# ---------- Variables ----------

low_vars = ['hi_d', 'aice_d', 'U10']
comps = {
    'hi_d': 'ice',
    'aice_d': 'ice',
    'U10': 'atm',
}

target_var = ['hi_d']

# ---------- Collect files ----------

low_res_files = collect_files(low_res_dirs, low_vars, start_year=1920)      # 1920 or 2006
high_res_files = collect_files(high_res_dirs, target_var, start_year=1920)  # 1920 or2006
coarsen_files = collect_files(high_res_dirs, low_vars, start_year=1920)     # 1920 or 2006

print('Low-res  | # ens:', len(low_res_files), '| # vars:', len(low_res_files[0]))
print('High-res | # ens:', len(high_res_files), '| # vars:', len(high_res_files[0]))
print('Coarsen  | # ens:', len(coarsen_files), '| # vars:', len(coarsen_files[0]))

# ### 2. Set up rectilinear grids and regional subsetting

# ---------- Region select ----------

# Cambridge Bay
# bbox = {"lon_min": -130, "lon_max": -80, "lat_min": 60, "lat_max": 80}

# Kivalina
# Larger region for regridding
bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360

# Actual ML domain
bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360
print("Region select done.")

# ---------- Native grids ----------

atm_dir = "/glade/p/cesmdata/cseg/inputdata/share/scripgrids/"
# nat_atm_lr = xr.open_dataset(atm_dir + "ne30np4_091226_pentagons.nc")
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

# nat_ice_lr = pop_tools.get_grid("POP_gx1v7")
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

# ---------- Destination grids ----------

dst_1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 1, 1.0)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 1, 1.0)),
})

dst_0p1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 0.1, 0.1)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 0.1, 0.1)),
})

print("Destination grids set up.")

# ---------- Corner arrays for conservative ("avg") regridding ----------

# Ice: exact corners from POP's native U-points (NE corner of each T-cell)
has_upoints = hasattr(nat_ice_hr, "ULONG") and hasattr(nat_ice_hr, "ULAT")
if not has_upoints:
    raise RuntimeError(
        "Expected ULONG/ULAT on the POP grid object for exact ice cell "
        "corners -- check pop_tools.get_grid output."
    )

tlon_full = (nat_ice_hr.TLONG.values % 360)
tlat_full = nat_ice_hr.TLAT.values
ulon_full = (nat_ice_hr.ULONG.values % 360)
ulat_full = nat_ice_hr.ULAT.values
ny_full = tlon_full.shape[0]

rows = np.where(mask_ice_hr)[0]
r0, r1 = rows.min(), rows.max()

if not np.array_equal(rows, np.arange(r0, r1 + 1)):
    raise RuntimeError(
        "mask_ice_hr rows are not contiguous in nlat -- the corner "
        "construction below assumes a contiguous latitude band."
    )

if r0 < 1 or r1 + 1 > ny_full - 1:
    raise RuntimeError(
        "Masked ice region touches the native grid's j-edge/pole fold -- "
        "corner construction near the true boundary isn't handled here."
    )

# Pad one ghost column in i for periodic longitude; no j-padding needed
# since the masked region is comfortably interior to the native grid.
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

print("Ice conservative source grid built (exact POP corners).")

# Destination 1deg grid with corners (exact, trivial for a regular grid)
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

# ---------- Build regridders ----------

print("Building/locating regridders...")

WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

# Ice interpolated
regridder_ice_to_1deg_interp = xe.Regridder(
    grid_ice_hr,
    dst_1deg,
    method="bilinear",
    periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/ice_hr_to_1deg_interp.nc",
    reuse_weights=True,
)

print(" >> Built regridder_ice_to_1deg_interp.")

# Ice conservative grid-cell average
regridder_ice_to_1deg_cons = xe.Regridder(
    grid_ice_hr_conserv,
    dst_1deg_b,
    method="conservative",
    periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/ice_hr_to_1deg_cons.nc",
    reuse_weights=True,
)

print(" >> Built regridder_ice_to_1deg_cons.")

# Ice high resolution (source grid pre-masked to the Kivalina bbox_regrid --
# NOT the same weight matrix as the full-native-grid ice_hr_to_0p1deg
# regridders used elsewhere, hence the distinct filename)
regridder_ice_to_0p1deg = xe.Regridder(
    grid_ice_hr,
    dst_0p1deg,
    method="bilinear",
    periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/ice_hr_to_0p1deg_kivalina_masked.nc",
    reuse_weights=False,
)

print(" >> Built regridder_ice_to_0p1deg.")

# Atm interpolated
regridder_atm_to_1deg_interp = xe.Regridder(
    grid_atm_hr,
    dst_1deg,
    method="nearest_s2d",
    locstream_in=True,
    periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/atm_hr_to_1deg_interp.nc",
    reuse_weights=True,
)

print(" >> Built regridder_atm_to_1deg_interp.")

# Atm "avg" method: a real bin average, not an xESMF regridder. xESMF's conservative
# method needs a *structured* (ny, nx) grid with (ny+1, nx+1) corners
# (ds_to_ESMFgrid -- confirmed by trying it: "AssertionError: lon_b should be size
# (Nx+1, Ny+1)"); it does not accept the ne120 SCRIP mesh's unstructured (ncol, nv)
# per-cell corner list the way grid_ice_hr_conserv's curvilinear corners work for the
# POP grid above. atm_bin_average (defined in the next section, next to process_file)
# does the scattered-point equivalent instead: average every native ne120 column
# falling inside each destination 1-degree cell. Reasonable here since a 1-degree cell
# spans ~O(100) ne120 columns (each ~0.25deg), so this differs little from true
# area-weighted conservative remapping.

# ### 3. Build low resolution (coarsened) predictors

def atm_bin_average(da, native_lat, native_lon, dst_lat, dst_lon):
    """Bin-average scattered native (ncol) atm points into the dst_lat/dst_lon 1-degree
    grid -- the atm "avg" method's replacement for a proper xESMF conservative regrid
    (see the note above regridder_atm_to_1deg_interp: xESMF's conservative method
    requires a structured grid with (ny+1, nx+1) corners, which the unstructured ne120
    SCRIP mesh doesn't provide). Each destination cell's value is the plain mean of
    every native column whose center falls inside it; destination cells with no native
    columns come back NaN, same as an xESMF regridder would leave uncovered cells,
    so the existing `.fillna(0)` below still applies uniformly.
    """
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

    # ---------- Sea ice ----------

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

        # Select ML domain
        da_reg = da_reg.sel(
            lat=slice(bbox["lat_min"], bbox["lat_max"]),
            lon=slice(lon_min, lon_max)
        )

        da_reg = da_reg.fillna(0).astype(np.float32)
        ds.close()

        return da_reg

    # ---------- Atmosphere ----------
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

        da_reg = da_reg.sel(
            lat=slice(bbox["lat_min"], bbox["lat_max"]),
            lon=slice(lon_min, lon_max)
        )

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


# ---------- Build X (Low-Res Predictors) ----------
# This HR CESM-LE run (d651007) only carries scalar 10m wind speed (U10, m/s) at daily
# frequency, not vector U/V components like the original CESM-LE pipeline this was
# adapted from -- so ice_vars/atm_vars use the real on-disk variable names (hi_d, aice_d,
# U10) and there's no "V" channel to build. (The previous version of this cell checked
# var against ["hi","aice"]/["U","V"], which never matched the actual low_vars
# ['hi_d','aice_d','U10'] being passed in, so it always raised "Unknown variable" before
# regridding anything -- and its avg-method atm branch called an atm_bin_average()
# function that was never defined anywhere in the repo. It's defined for real above.)
ice_vars, atm_vars = ["hi_d", "aice_d"], ["U10"]

for method in ("interp", "avg"):

    X_list = []

    for i, member in enumerate(coarsen_files):

        print(f"Processing Ensemble #{i+1}, method={method}...")

        channels = [load_wrapper(member[var], var) for var in low_vars]
        print(' >> All channels regridded and loaded.')

        min_t = min(c.sizes["time"] for c in channels)  # align time across channels

        channels = [c.isel(time=slice(0, min_t)) for c in channels]

        member_da = xr.concat(channels, dim="channel")
        member_da.name = "X"
        member_da = member_da.expand_dims({"ensemble": [i]})

        X_list.append(member_da)

    min_t_global = min(x.sizes["time"] for x in X_list)  # align time across ensembles
    X_list = [x.isel(time=slice(0, min_t_global)) for x in X_list]

    X_ds = xr.concat(X_list, dim="ensemble")  # combine ensembles

    if ("channel" in X_ds.dims and "channel" not in X_ds.coords):  # add channel coordinate
        X_ds = X_ds.assign_coords(channel=np.arange(X_ds.sizes["channel"]))

    X_ds = X_ds.assign_coords(channel=low_vars)  # assign channel names

    # Metadata
    X_ds.attrs["description"] = "Low-resolution MESACLIP predictors on a common 1-degree rectilinear grid."
    X_ds.attrs["notes"] = f"Regridding method: {method}; NaNs filled with zero for ML pipeline."
    X_ds.attrs["created_by"] = "Sky Gale"
    X_ds.attrs["date_created"] = (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    X_ds.attrs["variables"] = (
        "hi_d: sea ice thickness (m); "
        "aice_d: sea ice concentration (%); "
        "U10: 10m wind speed (m s-1)"
    )

    X_ds = X_ds.transpose("ensemble", "time", "channel", "lat", "lon")  # arrange dimensions

    # Save
    save_path = f"/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_HIST_daily_{method}.nc"
    X_ds.to_netcdf(save_path)
    print("\nSaved to:", save_path)

# ### 4. Build high resolution predictand

def process_file_hi(args):

    file, target_var = args

    ds = xr.open_dataset(file)

    da = ds[target_var[0]]
    da = da.rename({"nj": "nlat", "ni": "nlon"})

    da = da.isel(nlat=mask_ice_hr)

    # Regrid to regional 0.1deg grid
    da_reg = regridder_ice_to_0p1deg(da, skipna=True)

    # Select ML domain
    da_reg = da_reg.sel(
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
        lon=slice(lon_min, lon_max),
    )

    da_reg = da_reg.fillna(0).astype(np.float32)
    ds.close()

    return da_reg


def load_wrapper_hi(files, var):
    args = [(f, var) for f in files]
    data_list = [process_file_hi(arg) for arg in args]
    out = xr.concat(data_list, dim="time")
    return out


# ---------- Build Y (High-Res Target) ----------

Y_list = []

for i, member in enumerate(high_res_files):

    print(f"Processing Ensemble #{i+1}...")

    var_da = load_wrapper_hi(member[target_var[0]], target_var)

    var_da.name = "Y"
    member_da = var_da.expand_dims({"ensemble": [i], "channel": [0]})
    member_da = member_da.transpose("ensemble", "time", "channel", "lat", "lon")

    Y_list.append(member_da)

Y_ds = xr.concat(Y_list, dim="ensemble")  # combine all ensembles

if "channel" in Y_ds.dims and "channel" not in Y_ds.coords:  # add channel dimension
    Y_ds = Y_ds.assign_coords(channel=np.arange(Y_ds.sizes["channel"]))

# Metadata
Y_ds.attrs["description"] = "High-resolution daily MESACLIP regridded to a regional 0.1-degree rectilinear grid."
Y_ds.attrs["notes"] = "NaNs filled with zero."
Y_ds.attrs["created_by"] = "Sky Gale"
Y_ds.attrs["date_created"] = (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
Y_ds.attrs["variables"] = (
    "hi: sea ice thickness (m)"
)

# Save
save_path = "/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_HIST_daily.nc"
Y_ds.to_netcdf(save_path)
print("\nSaved to:", save_path)

# ### 5. Check shapes

X1 = xr.open_dataset('/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_HIST_daily_interp.nc').X
X2 = xr.open_dataset('/glade/derecho/scratch/skygale/Downscaling_Data/X_MESA_HR_HIST_daily_avg.nc').X
Y = xr.open_dataset('/glade/derecho/scratch/skygale/Downscaling_Data/Y_MESA_HR_HIST_daily.nc').Y

print(X1.shape, X2.shape, Y.shape)

