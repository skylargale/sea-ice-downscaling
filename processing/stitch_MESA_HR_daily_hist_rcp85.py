"""Stitches the HIST (1920-2006) and RCP8.5 (2006-2021ish) daily MESA-HR X/Y
files into one continuous per-member record, so submit_engressnet_daily.sh
(a single x_path/y_path, continuous time, no HIST/future splicing support in
this Version4 line -- unlike Version3's --x-path-future/--y-path-future) can
train straight across the 2006 HIST/scenario boundary.

Member alignment: HIST used 9 members (.002-.010, d651007, sorted glob
order -> ensemble index 0-8). RCP8.5 could only use 6 of those 9 (.002/.003
lack daily ice, .009 lacks daily wind in the RCP8.5 archive -- see
build_X_Y_from_MESA-HR_daily_rcp85.py's docstring) -- .004,.005,.006,.007,
.008,.010, sorted glob order -> RCP ensemble index 0-5. The physical-member
mapping from RCP index to HIST index is therefore:
    RCP idx 0 (.004) -> HIST idx 2
    RCP idx 1 (.005) -> HIST idx 3
    RCP idx 2 (.006) -> HIST idx 4
    RCP idx 3 (.007) -> HIST idx 5
    RCP idx 4 (.008) -> HIST idx 6
    RCP idx 5 (.010) -> HIST idx 8
verified directly against both scripts' sorted `high_res_dirs` listings.
The combined file uses this same 6-member set throughout (including for the
pre-2006 portion) so every training window in the sweep draws from a
consistent ensemble, not 9 members pre-2006 and 6 after.

Output is trimmed to TIME_START onward (default 1999-01-01) -- the sweep's
earliest training window starts 2000, and the full HIST record back to 1920
would be ~14x more data than needed (the original HIST Y file alone is
~113GB for 86 years; run_pipeline() loads the whole array into memory before
filtering by year, so trimming here directly saves the training job's
memory/load time too, same reasoning as the daily FOSI mem bump).

Drops any NaN-containing timestep on either side of the join before
concatenating (2026-08-06), found in two rounds:
(1) HIST X's very last timestep (2006-01-01) is entirely NaN across all
    channels/members in the original X_MESA_HR_HIST_daily_{interp,avg}.nc
    files -- a pre-existing issue in that HIST build (confirmed directly
    against the untouched HIST-only file, not introduced by this script),
    never hit before because no prior run's train/test window landed
    exactly on that boundary day. Y's HIST file doesn't have this problem
    (0 NaN at the same timestep).
(2) RCP8.5 X's own first timestep is *also* NaN in 2 of 3 channels -- and,
    separately, X_MESA_HR_RCP85_daily_*.nc's first timestep is 2006-01-01,
    one day earlier than Y_MESA_HR_RCP85_daily.nc's 2006-01-02 (X and Y use
    different day-labeling conventions for daily-average output in this
    archive, similar in spirit to the day-2-start convention already seen
    in the FOSI daily files). Looks like a genuine CESM branch-point
    artifact: 2006-01-01 is the RCP8.5 run's very first archived day, with
    no valid prior-day state within that file to average from.

IMPORTANT (found the hard way, 2026-08-06, after all 48 first-round MESA
combo-sweep jobs crashed identically on a boolean-index shape mismatch):
dropping bad timesteps independently per file breaks X/Y time alignment,
since Y has zero NaN of its own and so never drops anything -- X ends up
2 timesteps shorter than Y, and run_pipeline's split_train_test (which
indexes X and Y by the same positional time mask) crashes immediately.
X-interp, X-avg, and Y are therefore stitched independently first, but then
explicitly re-aligned to their common time-coordinate intersection (xr.align
join="inner") as a final step before saving, rather than trusting that
per-file NaN-dropping happens to leave them in lockstep.
"""

import xarray as xr

DATA_DIR = "/glade/derecho/scratch/skygale/Downscaling_Data"
TIME_START = "1999-01-01"

HIST_TO_RCP_ENSEMBLE_IDX = [2, 3, 4, 5, 6, 8]  # HIST idx for RCP idx 0..5


def drop_fully_nan_time(da, label):
    """Drops any timestep with ANY NaN cell (not just ones that are
    entirely NaN) -- see module docstring for the two real boundary
    artifacts this catches (HIST's last day, RCP8.5's first day). Started
    as an all-NaN check, tightened to any-NaN (2026-08-06) after finding
    RCP8.5's first-day corruption is actually partial (2 of 3 channels NaN,
    not all 3 -- looks like a wind-vs-ice-state day-1 archive gap at the
    scenario branch point, ice state carries over from the restart but wind
    forcing doesn't have a valid record yet), which an all-NaN check missed
    entirely. By this point in the pipeline every channel has already been
    through the build script's own fillna(0) for benign missing regridder
    coverage, so any NaN still present represents real unresolved missing
    data, not a benign gap -- any-NaN is the right bar. Generic on purpose:
    applied to both sides of the join rather than hardcoding either date."""
    non_time_dims = [d for d in da.dims if d != "time"]
    bad_time = da.isnull().any(dim=non_time_dims).compute()
    n_bad = int(bad_time.sum())
    if n_bad > 0:
        bad_times = da["time"].values[bad_time.values]
        print(f"  Dropping {n_bad} NaN-containing {label} timestep(s): {bad_times}")
        da = da.sel(time=~bad_time)
    return da


def stitch_one(hist_path, rcp_path, var_name):
    """Builds the stitched (HIST+RCP8.5, NaN-timesteps dropped) DataArray
    for one variable/method, but does NOT save it yet -- callers must
    re-align every variable's output to a common time intersection first
    (see module docstring)."""
    hist_ds = xr.open_dataset(hist_path)
    rcp_ds = xr.open_dataset(rcp_path)

    hist_da = hist_ds[var_name].isel(ensemble=HIST_TO_RCP_ENSEMBLE_IDX)
    hist_da = hist_da.assign_coords(ensemble=rcp_ds["ensemble"].values)

    rcp_first_time = rcp_ds["time"].values[0]
    hist_da = hist_da.sel(time=slice(TIME_START, None))
    hist_da = hist_da.sel(time=hist_da["time"] < rcp_first_time)
    hist_da = drop_fully_nan_time(hist_da, f"HIST {var_name}")

    rcp_da = drop_fully_nan_time(rcp_ds[var_name], f"RCP8.5 {var_name}")

    combined = xr.concat([hist_da, rcp_da], dim="time")
    combined.name = var_name
    combined.attrs["description"] = (
        (hist_ds[var_name].attrs.get("description", "")) +
        " | Stitched HIST+RCP8.5, 6-member intersection (.004/.005/.006/.007/.008/.010), "
        f"time >= {TIME_START}."
    )
    combined.load()  # materialize before closing the source datasets
    hist_ds.close()
    rcp_ds.close()
    return combined


x_interp = stitch_one(f"{DATA_DIR}/X_MESA_HR_HIST_daily_interp.nc", f"{DATA_DIR}/X_MESA_HR_RCP85_daily_interp.nc", "X")
x_avg = stitch_one(f"{DATA_DIR}/X_MESA_HR_HIST_daily_avg.nc", f"{DATA_DIR}/X_MESA_HR_RCP85_daily_avg.nc", "X")
y = stitch_one(f"{DATA_DIR}/Y_MESA_HR_HIST_daily.nc", f"{DATA_DIR}/Y_MESA_HR_RCP85_daily.nc", "Y")

print(f"\nPre-alignment shapes: X(interp)={x_interp.shape}, X(avg)={x_avg.shape}, Y={y.shape}")

# Re-align to the common time intersection (see module docstring): each
# variable/method dropped its own NaN timesteps independently, which is not
# guaranteed to leave X and Y (or even X-interp and X-avg) with identical
# time axes. run_pipeline requires X and Y to share the same time index
# positionally, so this is not optional.
#
# exclude=["channel", "ensemble", "lat", "lon"] is required, not cosmetic:
# X's channel coordinate is string-labeled (['hi_d','aice_d','U10'], dtype
# <U6) while Y's is integer ([0], dtype int64) -- letting xr.align touch
# "channel" too would try to inner-join two disjoint label sets of
# incompatible dtypes, silently producing an empty (0-length) channel
# dimension instead of erroring. Only "time" should ever need reconciling
# here; ensemble/lat/lon are already identical by construction.
x_interp, x_avg, y = xr.align(x_interp, x_avg, y, join="inner", exclude=["channel", "ensemble", "lat", "lon"])
print(f"Post-alignment shapes (must all share the same time length): "
      f"X(interp)={x_interp.shape}, X(avg)={x_avg.shape}, Y={y.shape}")
assert x_interp.sizes["time"] == x_avg.sizes["time"] == y.sizes["time"], "alignment failed to equalize time length"

x_interp.to_netcdf(f"{DATA_DIR}/X_MESA_HR_daily_interp.nc")
print(f"Saved X_MESA_HR_daily_interp.nc: shape={x_interp.shape}, "
      f"time {str(x_interp.time.values[0])} to {str(x_interp.time.values[-1])}")
x_avg.to_netcdf(f"{DATA_DIR}/X_MESA_HR_daily_avg.nc")
print(f"Saved X_MESA_HR_daily_avg.nc: shape={x_avg.shape}, "
      f"time {str(x_avg.time.values[0])} to {str(x_avg.time.values[-1])}")
y.to_netcdf(f"{DATA_DIR}/Y_MESA_HR_daily.nc")
print(f"Saved Y_MESA_HR_daily.nc: shape={y.shape}, "
      f"time {str(y.time.values[0])} to {str(y.time.values[-1])}")

print("\nVerifying combined files (chunked NaN check -- a whole-array in-memory"
      " check OOM-killed a plain read on the login node for the daily-FOSI"
      " files, see project memory)...")
for method in ("interp", "avg"):
    x = xr.open_dataset(f"{DATA_DIR}/X_MESA_HR_daily_{method}.nc", chunks={"time": 500}).X
    print(f"X ({method}):", x.shape, "NaN:", bool(x.isnull().any().compute()))
y_check = xr.open_dataset(f"{DATA_DIR}/Y_MESA_HR_daily.nc", chunks={"time": 500}).Y
print("Y:", y_check.shape, "NaN:", bool(y_check.isnull().any().compute()))
print("Y time range:", str(y_check.time.values[0]), "to", str(y_check.time.values[-1]))
assert y_check.sizes["time"] == xr.open_dataset(f"{DATA_DIR}/X_MESA_HR_daily_interp.nc").sizes["time"], \
    "X/Y time length mismatch survived to disk -- alignment bug"
