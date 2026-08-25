"""
observing_cryosat2_masked_comparison.py

Direct answer to "can we only look at cells where CryoSat actually has
data, and how much of the coastal band does that really cover?" Rebuilds
the December 2020 3-panel comparison with ALL THREE panels (CryoSat,
network, bilinear) masked to the exact same footprint -- cells where
CryoSat-2 has a valid retrieval that month -- instead of showing the
network/bilinear over the full domain while CryoSat shows gaps. Also draws
the coastal-band boundary (coastal_width=5, same definition used
throughout this project) directly on the map and reports the actual %
of coastal vs. interior cells CryoSat covers, per month, so "does CryoSat
really cover the coast" is answered with a number, not a visual impression.
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_engressnet as fe

YEAR = 2020
MONTHS = [1, 2, 3, 4, 10, 11, 12]
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

RUN_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_daily_5705277.casper-pbs"
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
BASE_OUT = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020"

fields = np.load(os.path.join(EVAL_DIR, "fields.npz"))
Y_pred_phys = fields["Y_pred_phys"][:, 0]
Y_base_phys = fields["Y_base_phys"][:, 0]
mask_test = fields["mask_test"]

with open(os.path.join(EVAL_DIR, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
target_lat = np.asarray(tile_geometry[0]["target_lat"])
target_lon = np.asarray(tile_geometry[0]["target_lon"])

sample_times = pd.read_csv(os.path.join(EVAL_DIR, "sample_times.csv"))
sample_times["time"] = pd.to_datetime(sample_times["time"])
sample_month = sample_times["time"].dt.month.values

ocean_hw = mask_test[0, 0] <= 0.5
land_mask_hw = mask_test[0, 0]
coastal_hw = fe.coastal_band_mask(land_mask_hw[None, None], coastal_width=5)[0, 0].numpy()
n_coastal_total = int(coastal_hw.sum())
n_interior_total = int((ocean_hw & ~coastal_hw).sum())

dst_grid = xr.Dataset({"lat": ("lat", target_lat), "lon": ("lon", target_lon)})
regridder = xe.Regridder(
    xr.Dataset({"lat": (["y", "x"], np.zeros((448, 304))), "lon": (["y", "x"], np.zeros((448, 304)))}),
    dst_grid, method="bilinear", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/cryosat2_rdeft4_to_medium_domain.nc", reuse_weights=True,
)

print(f"Coastal band total: {n_coastal_total} cells. Interior ocean total: {n_interior_total} cells.\n")
print(f"{'month':>5} {'coastal_covered':>16} {'coastal_%':>10} {'interior_covered':>18} {'interior_%':>11}")

cov_rows = []
cryo_maps = {}
for month in MONTHS:
    matches = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")]
    if not matches:
        continue
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, matches[0]))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)
    sit_da = sit.assign_coords(lat=(["y", "x"], ds["lat"].values), lon=(["y", "x"], ds["lon"].values % 360))
    cryo_map = regridder(sit_da, skipna=True).values
    cryo_maps[month] = cryo_map
    ds.close()

    valid = np.isfinite(cryo_map)
    n_coastal_cov = int((valid & coastal_hw).sum())
    n_interior_cov = int((valid & ~coastal_hw & ocean_hw).sum())
    pct_coastal = 100 * n_coastal_cov / n_coastal_total
    pct_interior = 100 * n_interior_cov / n_interior_total
    print(f"{month:>5} {n_coastal_cov:>16} {pct_coastal:>9.1f}% {n_interior_cov:>18} {pct_interior:>10.1f}%")
    cov_rows.append({"month": month, "coastal_covered": n_coastal_cov, "coastal_pct": pct_coastal,
                      "interior_covered": n_interior_cov, "interior_pct": pct_interior})

cov_df = pd.DataFrame(cov_rows)
cov_df.to_csv(os.path.join(RUN_DIR, "cryosat2_validation", "coastal_vs_interior_coverage_pct.csv"), index=False)
print(f"\nMean coastal coverage: {cov_df['coastal_pct'].mean():.1f}%  "
      f"Mean interior coverage: {cov_df['interior_pct'].mean():.1f}%")

# ---------------------------------------------------------------------------
# Masked 3-panel figure for December (all panels restricted to the exact
# same CryoSat-valid footprint), with the coastal-band boundary drawn on top
# ---------------------------------------------------------------------------

plot_month = 12
cryo_map = cryo_maps[plot_month]
valid = np.isfinite(cryo_map) & ocean_hw
day_idx = np.where(sample_month == plot_month)[0]
net_map = np.where(valid, Y_pred_phys[day_idx].mean(axis=0), np.nan)
bilin_map = np.where(valid, Y_base_phys[day_idx].mean(axis=0), np.nan)
cryo_masked = np.where(valid, cryo_map, np.nan)

plot_bbox = {"lon_min": -182, "lon_max": -151, "lat_min": 60, "lat_max": 75}
proj, boundary_path, central_lon = fe.make_polar_proj(plot_bbox)
lon2d, lat2d = np.meshgrid(target_lon, target_lat)

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), subplot_kw={"projection": proj})
panels = [("CryoSat-2 RDEFT4 (truth)", cryo_masked), ("Network (Stochastic UNet Mean)", net_map),
          ("Bilinear baseline", bilin_map)]
for ax, (title, field) in zip(axes, panels):
    fe.style_polar_ax(ax, proj, boundary_path, plot_bbox)
    pc = ax.pcolormesh(lon2d, lat2d, field, transform=ccrs.PlateCarree(),
                        cmap="viridis", vmin=0, vmax=2.5, shading="auto")
    ax.contour(lon2d, lat2d, coastal_hw.astype(float), levels=[0.5], colors="red",
               linewidths=1.2, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11)
fig.colorbar(pc, ax=axes, orientation="horizontal", pad=0.05, shrink=0.5, label="SIT (m)")
n_cov = int(valid.sum())
n_cov_coastal = int((valid & coastal_hw).sum())
fig.suptitle(f"2020-12: ALL THREE panels masked to CryoSat-2's actual footprint only "
             f"({n_cov} cells total, {n_cov_coastal}/{n_coastal_total} coastal -- red outline)", y=1.03)

fig_path = os.path.join(RUN_DIR, "cryosat2_validation", f"cryosat2_masked_comparison_{YEAR}{plot_month:02d}.png")
fig.savefig(fig_path, dpi=180, bbox_inches="tight")
print("\nSaved figure:", fig_path)
print("Done.")
