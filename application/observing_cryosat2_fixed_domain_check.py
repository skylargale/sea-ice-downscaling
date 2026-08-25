"""
observing_cryosat2_fixed_domain_check.py

Addresses a real methodological problem in every CryoSat-2 comparison so
far: each month's "valid cells" mask was whatever CryoSat happened to cover
that month (n_valid ranged from 30 coastal cells in October to 3043 in
Jan-Apr) -- so "coverage %"/RMSE aggregated across months were mixing
different, shifting subsets of the domain, not testing the same fixed set
of physical locations every time. CryoSat's own coverage gaps aren't
random (orbit tracks, QC flags, retrieval-failure regions), so this could
bias results either direction.

Fix: compute a STRICT common-cell mask -- cells with valid CryoSat data in
EVERY evaluated month -- and redo both the network-vs-bilinear RMSE
comparison and the ensemble 5-95% coverage check restricted to that fixed
set, split coastal/interior. Reports the common-mask size for two groupings
(all 7 months; just the 4 full-coverage winter months Jan-Apr) so it's
clear how much data survives the stricter, fairer test.
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

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_engressnet as fe

YEAR = 2020
ALL_MONTHS = [1, 2, 3, 4, 10, 11, 12]
FULL_COVERAGE_MONTHS = [1, 2, 3, 4]
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

RUN_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_daily_5705277.casper-pbs"
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
OUT_DIR = os.path.join(RUN_DIR, "cryosat2_validation")

fields = np.load(os.path.join(EVAL_DIR, "fields.npz"))
Y_pred_phys = fields["Y_pred_phys"][:, 0]
Y_base_phys = fields["Y_base_phys"][:, 0]
preds_all_phys = fields["preds_all_phys"][:, :, 0]
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
interior_ocean_hw = ocean_hw & ~coastal_hw

dst_grid = xr.Dataset({"lat": ("lat", target_lat), "lon": ("lon", target_lon)})
regridder = xe.Regridder(
    xr.Dataset({"lat": (["y", "x"], np.zeros((448, 304))), "lon": (["y", "x"], np.zeros((448, 304)))}),
    dst_grid, method="bilinear", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/cryosat2_rdeft4_to_medium_domain.nc", reuse_weights=True,
)

cryo_maps = {}
for month in ALL_MONTHS:
    matches = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")]
    if not matches:
        continue
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, matches[0]))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)
    sit_da = sit.assign_coords(lat=(["y", "x"], ds["lat"].values), lon=(["y", "x"], ds["lon"].values % 360))
    cryo_maps[month] = regridder(sit_da, skipna=True).values
    ds.close()

# ---------------------------------------------------------------------------
# Strict common-cell masks: valid in EVERY month of the group
# ---------------------------------------------------------------------------

def common_mask(months):
    m = np.ones_like(ocean_hw, dtype=bool)
    for mo in months:
        m &= np.isfinite(cryo_maps[mo])
    return m

for group_name, months in [("all_7_months", ALL_MONTHS), ("full_coverage_Jan-Apr", FULL_COVERAGE_MONTHS)]:
    cm = common_mask(months)
    n_coastal = int((cm & coastal_hw).sum())
    n_interior = int((cm & interior_ocean_hw).sum())
    print(f"Common-cell mask ({group_name}): {n_coastal} coastal cells, {n_interior} interior cells "
          f"(out of {int(coastal_hw.sum())} / {int(interior_ocean_hw.sum())} total)")

# Use the group with usable sample sizes for the actual re-analysis
GROUP_NAME, MONTHS = "full_coverage_Jan-Apr", FULL_COVERAGE_MONTHS
fixed_mask = common_mask(MONTHS)
fixed_coastal = fixed_mask & coastal_hw
fixed_interior = fixed_mask & interior_ocean_hw
print(f"\nUsing '{GROUP_NAME}' as the fixed, fair comparison domain: "
      f"{int(fixed_coastal.sum())} coastal cells, {int(fixed_interior.sum())} interior cells, "
      f"IDENTICAL set of physical grid cells every month.")

# ---------------------------------------------------------------------------
# Redo RMSE (network vs. bilinear) on the fixed common-cell mask
# ---------------------------------------------------------------------------

rmse_rows = []
cov_rows = []
for month in MONTHS:
    cryo_map = cryo_maps[month]
    day_idx = np.where(sample_month == month)[0]
    net_map = Y_pred_phys[day_idx].mean(axis=0)
    bilin_map = Y_base_phys[day_idx].mean(axis=0)
    member_month_mean = preds_all_phys[day_idx].mean(axis=0)
    p5 = np.percentile(member_month_mean, 5, axis=0)
    p95 = np.percentile(member_month_mean, 95, axis=0)
    inside = (cryo_map >= p5) & (cryo_map <= p95)

    for region_name, region_mask in [("coastal", fixed_coastal), ("interior", fixed_interior)]:
        n = int(region_mask.sum())
        net_rmse = float(np.sqrt(np.mean((net_map[region_mask] - cryo_map[region_mask]) ** 2)))
        bilin_rmse = float(np.sqrt(np.mean((bilin_map[region_mask] - cryo_map[region_mask]) ** 2)))
        coverage = float(inside[region_mask].mean()) * 100
        rmse_rows.append({"month": month, "region": region_name, "n_fixed_cells": n,
                           "network_rmse": net_rmse, "bilinear_rmse": bilin_rmse,
                           "network_worse": net_rmse > bilin_rmse})
        cov_rows.append({"month": month, "region": region_name, "n_fixed_cells": n,
                          "coverage_5_95_pct": coverage})

rmse_df = pd.DataFrame(rmse_rows)
cov_df = pd.DataFrame(cov_rows)
rmse_df.to_csv(os.path.join(OUT_DIR, "fixed_domain_network_vs_bilinear_rmse.csv"), index=False)
cov_df.to_csv(os.path.join(OUT_DIR, "fixed_domain_ensemble_coverage.csv"), index=False)

print("\n=== RMSE on the FIXED common-cell domain (Jan-Apr, same cells every month) ===")
print(rmse_df.to_string(index=False))
print("\n=== Ensemble 5-95% coverage on the FIXED common-cell domain ===")
print(cov_df.to_string(index=False))

print("\n=== Summary ===")
for region in ["coastal", "interior"]:
    r = rmse_df[rmse_df.region == region]
    c = cov_df[cov_df.region == region]
    print(f"{region}: mean network RMSE={r['network_rmse'].mean():.4f}  "
          f"bilinear RMSE={r['bilinear_rmse'].mean():.4f}  "
          f"network worse {r['network_worse'].sum()}/{len(r)} months  "
          f"mean coverage={c['coverage_5_95_pct'].mean():.2f}% (ideal ~90%)")

# ---------------------------------------------------------------------------
# Figure: fixed-domain map showing exactly which cells are being scored
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))
display = np.where(fixed_coastal, 2, np.where(fixed_interior, 1, np.where(ocean_hw, 0.3, np.nan)))
im = ax.imshow(display, origin="lower", cmap="viridis", vmin=0, vmax=2)
ax.set_title(f"Fixed common-cell comparison domain\n({GROUP_NAME}: {int(fixed_coastal.sum())} coastal + "
             f"{int(fixed_interior.sum())} interior cells,\nvalid in CryoSat-2 every month, not shifting)")
ax.set_xlabel("grid x"); ax.set_ylabel("grid y")
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "fixed_domain_mask.png")
fig.savefig(fig_path, dpi=170, bbox_inches="tight")
print("\nSaved figure:", fig_path)
print("Done.")
