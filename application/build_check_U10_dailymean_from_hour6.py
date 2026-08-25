"""Vet a true daily-mean 10m wind speed channel (built from hour_6 U10 snapshots)
against the daily-max U10 channel already baked into X_MESA_HR_RCP85_daily_avg.nc.

Why: MESA's only daily wind product is `U10` from day_1 (cice/cam.h1) archives,
which is a daily *maximum* (raw file attrs: long_name '10m wind speed', cell_methods
'time: maximum') -- a real convention mismatch against PIOMAS/FOSI's JRA55-forced
u_10/v_10 channels, which are daily means.

The hour_6 directory carries U10 on TWO separate history streams, not one:
  - cam.h2.U10: instantaneous 00/06/12/18Z snapshots, no cell_methods.
  - cam.h4.U10: cell_methods 'time: mean' -- a real 6-hour PERIOD-AVERAGED wind
    speed (confirmed via raw file attrs; same time coordinates as h2 but
    different values -- not a duplicate).
This script uses h4 only: averaging 4 true 6-hour period-means/day gives an
EXACT daily mean (not an approximation from 4 point samples like h2 would give),
better-founded than the original UBOT/VBOT plan too, since it's still the CAM
near-surface-diagnosed wind speed (long_name '10m wind speed'), not a
lowest-model-level proxy (UBOT/VBOT: long_name 'Lowest model level zonal wind',
which would carry a real height bias).

This script:
  1. Rebuilds the daily-mean U10 (from hour_6) for 2016-2020, the 6 usable RCP8.5
     high-res members (.004/.005/.006/.007/.008/.010 -- same exclusion list as
     build_X_Y_from_MESA-HR_daily_rcp85.py), regridded with the identical
     atm_bin_average "avg" method/grid used to build X_MESA_HR_RCP85_daily_avg.nc,
     so the two are directly comparable cell-for-cell.
  2. Loads the existing daily-max U10 channel from X_MESA_HR_RCP85_daily_avg.nc for
     the same members/years.
  3. Reports bias/ratio/correlation between the two, broken out by the user's
     intended train (2016-2018) / val (2019) / test (2020) split, and saves the new
     daily-mean field to disk for reuse if the comparison looks sane.

Not yet wired into the training pipeline -- this is a vetting step before deciding
whether to swap MESA's U10 channel for this daily-mean version.
"""

import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")

# ---------- Config ----------

YEARS = [2016, 2017, 2018, 2019, 2020]
SPLITS = {
    "train (2016-2018)": (2016, 2018),
    "val (2019)": (2019, 2019),
    "test (2020)": (2020, 2020),
}

DATA_DIR = "/glade/derecho/scratch/skygale/Downscaling_Data"
EXISTING_X_PATH = f"{DATA_DIR}/X_MESA_HR_RCP85_daily_avg.nc"
OUT_PATH = f"{DATA_DIR}/U10_dailymean_from_hour6_RCP85_2016_2020.nc"
OUT_FIG_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020"

EXCLUDED_MEMBER_TAGS = ["sehires38", ".30-2006-2100.002", ".31-2006-2100.003", "-2006-2100.009"]

bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360

bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360

# ---------- Member directories (must match build_X_Y_from_MESA-HR_daily_rcp85.py's
# ordering exactly, since we're comparing against that file's `ensemble` index) ----------

high_res_dirs = sorted(glob.glob("/glade/campaign/collections/gdex/data/d651009/b.e13.*"))
high_res_dirs = [d for d in high_res_dirs if not any(tag in d for tag in EXCLUDED_MEMBER_TAGS)]

print(f"Usable RCP8.5 members ({len(high_res_dirs)}):")
for d in high_res_dirs:
    print(" ", d.split("/")[-1])

# ---------- Native atm grid + Arctic mask (identical to the MESA build scripts) ----------

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

native_lat_masked = atm_lat[mask_atm_hr]
native_lon_masked = atm_lon[mask_atm_hr]

dst_1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 1, 1.0)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 1, 1.0)),
})


def atm_bin_average(da, native_lat, native_lon, dst_lat, dst_lon):
    """Verbatim copy of the bin-average method used to build X_MESA_HR_*_avg.nc's
    atm channel, so the new daily-mean field lands on the identical grid/cells.
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


# ---------- Build daily-mean U10 from hour_6, one member/year-file at a time ----------

member_daily_means = []

for m_i, mdir in enumerate(high_res_dirs):
    year_das = []
    for year in YEARS:
        # h4 only: real 6-hour period-mean U10 (cell_methods='time: mean'), not the
        # h2 instantaneous-snapshot stream -- see module docstring.
        pattern = f"{mdir}/atm/proc/tseries/hour_6/*.cam.h4.U10.{year}*.nc"
        files = sorted(glob.glob(pattern))
        if not files:
            raise RuntimeError(f"No hour_6 h4 U10 file found for {mdir.split('/')[-1]} year {year} (pattern: {pattern})")
        if len(files) > 1:
            raise RuntimeError(f"Expected exactly 1 h4 U10 file for {mdir.split('/')[-1]} {year}, got {len(files)}: {files}")

        for f in files:
            print(f"Member {m_i+1}/{len(high_res_dirs)} ({mdir.split('/')[-1]}), file {f.split('/')[-1]} ...")
            ds = xr.open_dataset(f)
            da = ds["U10"].load()
            assert ds["U10"].attrs.get("cell_methods") == "time: mean", (
                f"Expected h4 U10 to be a period mean, got attrs={ds['U10'].attrs}"
            )
            ds.close()

            da = da.isel(ncol=mask_atm_hr)
            da_daily = da.resample(time="1D").mean()
            da_reg = atm_bin_average(da_daily, native_lat_masked, native_lon_masked,
                                      dst_1deg.lat.values, dst_1deg.lon.values)
            da_reg = da_reg.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max))
            da_reg = da_reg.fillna(0).astype(np.float32)
            year_das.append(da_reg)

    member_da = xr.concat(year_das, dim="time")
    member_da = member_da.expand_dims({"ensemble": [m_i]})
    member_daily_means.append(member_da)
    print(f"  >> member {m_i+1} done, {member_da.sizes['time']} days")

U10_dailymean = xr.concat(member_daily_means, dim="ensemble")
U10_dailymean.name = "U10_dailymean"
U10_dailymean.attrs["description"] = (
    "Daily-mean 10m wind speed (m/s), built by averaging 4x daily instantaneous "
    "hour_6 CAM U10 snapshots (same near-surface diagnostic as the daily-max U10 "
    "channel, not a lowest-model-level proxy), regridded with the same atm_bin_average "
    "method used for X_MESA_HR_RCP85_daily_avg.nc's U10 channel."
)
U10_dailymean.to_netcdf(OUT_PATH)
print(f"\nSaved daily-mean U10 to: {OUT_PATH}")
print(U10_dailymean.shape, U10_dailymean.dims)

# ---------- Load existing daily-max U10 for the same members/years ----------

X_ds = xr.open_dataset(EXISTING_X_PATH)
U10_dailymax = X_ds["X"].sel(channel="U10").isel(ensemble=slice(0, len(high_res_dirs)))
U10_dailymax = U10_dailymax.sel(time=slice(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31"))

# Align time coordinates exactly (both should be daily, same calendar, same length)
U10_dailymean = U10_dailymean.assign_coords(time=U10_dailymax.time.values[: U10_dailymean.sizes["time"]])
n_t = min(U10_dailymean.sizes["time"], U10_dailymax.sizes["time"])
U10_dailymean = U10_dailymean.isel(time=slice(0, n_t))
U10_dailymax = U10_dailymax.isel(time=slice(0, n_t))

diff = U10_dailymean.values - U10_dailymax.values
ratio = U10_dailymean.values / np.clip(U10_dailymax.values, 1e-6, None)

print("\n=== Overall comparison (daily-mean [hour_6] vs. daily-max [day_1], both regridded) ===")
print(f"mean(daily-mean) = {np.nanmean(U10_dailymean.values):.3f} m/s")
print(f"mean(daily-max)  = {np.nanmean(U10_dailymax.values):.3f} m/s")
print(f"mean(diff)       = {np.nanmean(diff):.3f} m/s")
print(f"mean(ratio)      = {np.nanmean(ratio):.3f}")
corr = np.corrcoef(U10_dailymean.values.ravel(), U10_dailymax.values.ravel())[0, 1]
print(f"pointwise correlation = {corr:.3f}")

# ---------- Split-wise breakdown ----------

times = pd.to_datetime(U10_dailymean.time.values.astype(str))
rows = []
for label, (y0, y1) in SPLITS.items():
    sel = (times.year >= y0) & (times.year <= y1)
    if sel.sum() == 0:
        continue
    dm = U10_dailymean.values[:, sel, :, :]
    dx = U10_dailymax.values[:, sel, :, :]
    d = dm - dx
    r = dm / np.clip(dx, 1e-6, None)
    c = np.corrcoef(dm.ravel(), dx.ravel())[0, 1]
    rows.append({
        "split": label,
        "n_days": int(sel.sum()),
        "mean_dailymean": float(np.nanmean(dm)),
        "mean_dailymax": float(np.nanmean(dx)),
        "mean_diff": float(np.nanmean(d)),
        "mean_ratio": float(np.nanmean(r)),
        "corr": float(c),
    })

summary = pd.DataFrame(rows)
print("\n=== Split-wise summary ===")
print(summary.to_string(index=False))

csv_path = f"{OUT_FIG_DIR}/U10_dailymean_vs_dailymax_summary.csv"
summary.to_csv(csv_path, index=False)
print(f"\nSaved split summary to: {csv_path}")

# ---------- Quick domain-mean time series figure (member 0, full period) ----------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dm_ts = U10_dailymean.isel(ensemble=0).mean(dim=["lat", "lon"]).values
    dx_ts = U10_dailymax.isel(ensemble=0).mean(dim=["lat", "lon"]).values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, dx_ts, label="daily-max U10 (existing, day_1)", alpha=0.7)
    ax.plot(times, dm_ts, label="daily-mean U10 (new, from hour_6)", alpha=0.7)
    ax.set_ylabel("Domain-mean U10 (m/s)")
    ax.set_title(f"MESA member {high_res_dirs[0].split('/')[-1]}: daily-mean vs. daily-max U10")
    ax.legend()
    fig.tight_layout()
    fig_path = f"{OUT_FIG_DIR}/U10_dailymean_vs_dailymax_timeseries.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved comparison figure to: {fig_path}")
except Exception as exc:
    print(f"Skipping diagnostic plot (non-fatal): {exc}")

print("\nDone.")
