"""
build_fig_piomas_fosi_mesa_snapshots.py

Maps of PIOMAS hi_d (native grid AND land-fixed regridded), FOSI hi_d, and
MESACLIP ensemble-mean hi_d, for six 2020 snapshot dates. Row 1 (native
PIOMAS, land-masked via PSC's io.dat_360_120.output, no regridding at all)
is the "ground truth" for how coarse/smooth PIOMAS's real spatial
information actually is; rows 2-4 are all on the shared 1-degree training
grid (21x51, lat 60-80N, lon 170-220E), all regridded with the same
conservative ("avg") method -- PIOMAS's row (2) uses the v3 regridder built
from grid.dat.pop's real U-point corners (see
processing/build_X_Y_PIOMAS_obs_2020_daily_v3_conservative.py), matching
FOSI/MESACLIP's own training convention exactly. (Updated 2026-08-25: this
row originally used the old bilinear-regridded PIOMAS field, built before
the conservative regridder existed -- rows 2-4 are now a clean apples-to-
apples regridding-method comparison, not confounded by PIOMAS alone using a
different method.)
"""

import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/evaluation")
import functions_engressnet as fe

import cftime

DATA_DIR = "/glade/derecho/scratch/skygale/Downscaling_Data"
DATE_STRS = ["2020-12-01", "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]
DATES = [cftime.DatetimeNoLeap(int(d[:4]), int(d[5:7]), int(d[8:10])) for d in DATE_STRS]
OUT_PATH = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/14_piomas_fosi_mesa_snapshots.png"

piomas_regrid = xr.open_dataset(f"{DATA_DIR}/X_PIOMAS_obs_2020_daily_v3_avg.nc").X.sel(channel="hi_d").isel(ensemble=0)
fosi = xr.open_dataset(f"{DATA_DIR}/X_FOSI_HR_JRA55_daily_avg.nc").X.sel(channel="hi_d").isel(ensemble=0)
mesa = xr.open_dataset(f"{DATA_DIR}/X_MESA_HR_daily_avg.nc").X.sel(channel="hi_d").mean(dim="ensemble")

lat, lon = piomas_regrid.lat.values, piomas_regrid.lon.values
assert np.allclose(lat, fosi.lat.values) and np.allclose(lon, fosi.lon.values), "PIOMAS/FOSI grid mismatch"
assert np.allclose(lat, mesa.lat.values) and np.allclose(lon, mesa.lon.values), "PIOMAS/MESA grid mismatch"
print("Regridded grids confirmed identical across all three products.")

# ---------------------------------------------------------------------------
# PIOMAS native grid, land-masked, no regridding at all -- same method as
# processing/build_X_Y_PIOMAS_obs_2020_daily_v2_landfixed.py
# ---------------------------------------------------------------------------

IO_MASK_PATH = "/glade/derecho/scratch/skygale/PIOMAS_daily/utilities/io.dat_360_120.output"
rows_txt = []
with open(IO_MASK_PATH) as f:
    for line in f:
        line = line.rstrip("\n")
        rows_txt.append([int(line[i:i + 2]) for i in range(0, len(line), 2)])
io_mask = np.array(rows_txt)
land_native = io_mask == 0

ds_native = xr.open_dataset("/glade/derecho/scratch/skygale/PIOMAS_daily/PIOMAS_hiday_2020.nc")
hi_native = ds_native["hi"].where(~xr.DataArray(land_native, dims=["nlat", "nlon"]))
piomas_native_lat = ds_native["lat"].values
piomas_native_lon = ds_native["lon"].values % 360

plot_bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
proj, boundary_path, central_lon = fe.make_polar_proj(plot_bbox)
lon2d, lat2d = np.meshgrid(lon, lat)

rows = [
    ("PIOMAS hi\n(native grid, land-masked,\nNOT regridded)", None),
    ("PIOMAS hi_d\n(conservative-avg-regridded,\ngrid.dat.pop corners)", piomas_regrid),
    ("FOSI hi_d\n(conservative-avg-regridded)", fosi),
    ("MESACLIP ensemble-mean hi_d\n(conservative-avg-regridded)", mesa),
]
vmax = 3.0

fig, axes = plt.subplots(4, 6, figsize=(24, 16), subplot_kw={"projection": proj})
for r, (row_label, da) in enumerate(rows):
    for c, date in enumerate(DATES):
        ax = axes[r, c]
        fe.style_polar_ax(ax, proj, boundary_path, plot_bbox)
        if r == 0:
            field = hi_native.sel(time=date, method="nearest").values
            pc = ax.pcolormesh(piomas_native_lon, piomas_native_lat, field, transform=ccrs.PlateCarree(),
                                cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        else:
            field = da.sel(time=date, method="nearest").values
            pc = ax.pcolormesh(lon2d, lat2d, field, transform=ccrs.PlateCarree(),
                                cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        if r == 0:
            ax.set_title(DATE_STRS[c], fontsize=12)
        if c == 0:
            ax.text(-0.18, 0.5, row_label, transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=11, fontweight="bold")

fig.colorbar(pc, ax=axes, orientation="horizontal", pad=0.02, shrink=0.4, label="SIT (m)")
fig.suptitle("PIOMAS native vs. regridded, and FOSI/MESACLIP (ensemble mean), 2020 snapshots\n"
             "Rows 2-4 share one grid AND the same conservative (avg) regridding method throughout",
             y=0.995, fontsize=14)
fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print("Saved:", OUT_PATH)
