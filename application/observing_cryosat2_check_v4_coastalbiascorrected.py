"""
observing_cryosat2_check_v4_coastalbiascorrected.py

CryoSat-2 comparison for the spatially-informed coastal bias correction
(processing/build_X_PIOMAS_obs_2020_v4_coastalbiascorrected.py). Identical
mechanism to observing_cryosat2_check_v3_conservative.py, pointed at this
run's eval_data. The correction was fit on Jan-Feb 2020 only -- Mar-Apr and
Oct-Dec-Nov-Dec here are genuinely held out from the fit, so the per-month
breakdown (not just the 7-month average) is the real generalization check.
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/evaluation")
import functions_engressnet as fe

YEAR = 2020
MONTHS = [1, 2, 3, 4, 10, 11, 12]
FIT_MONTHS = [1, 2]
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

RUN_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_daily_v4_coastalbiascorrected_5719064.casper-pbs"
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
OUT_DIR = os.path.join(RUN_DIR, "cryosat2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

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
interior_ocean_hw = ocean_hw & ~coastal_hw

dst_grid = xr.Dataset({"lat": ("lat", target_lat), "lon": ("lon", target_lon)})
regridder = xe.Regridder(
    xr.Dataset({"lat": (["y", "x"], np.zeros((448, 304))), "lon": (["y", "x"], np.zeros((448, 304)))}),
    dst_grid, method="bilinear", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/cryosat2_rdeft4_to_medium_domain.nc", reuse_weights=True,
)

rows = []
for month in MONTHS:
    matches = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")]
    if not matches:
        continue
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, matches[0]))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)
    sit_da = sit.assign_coords(lat=(["y", "x"], ds["lat"].values), lon=(["y", "x"], ds["lon"].values % 360))
    cryo_map = regridder(sit_da, skipna=True).values
    ds.close()

    day_idx = np.where(sample_month == month)[0]
    net_map = Y_pred_phys[day_idx].mean(axis=0)
    bilin_map = Y_base_phys[day_idx].mean(axis=0)

    for region_name, region_mask in [("coastal", coastal_hw), ("interior", interior_ocean_hw)]:
        valid = region_mask & np.isfinite(cryo_map)
        n = int(valid.sum())
        if n < 10:
            continue
        net_bias = float(np.mean(net_map[valid] - cryo_map[valid]))
        bilin_bias = float(np.mean(bilin_map[valid] - cryo_map[valid]))
        net_rmse = float(np.sqrt(np.mean((net_map[valid] - cryo_map[valid]) ** 2)))
        bilin_rmse = float(np.sqrt(np.mean((bilin_map[valid] - cryo_map[valid]) ** 2)))
        rows.append({
            "month": month, "fit_or_heldout": "fit" if month in FIT_MONTHS else "held-out",
            "region": region_name, "n_valid": n,
            "network_bias_vs_cryosat": net_bias, "bilinear_bias_vs_cryosat": bilin_bias,
            "network_worse_bias": abs(net_bias) > abs(bilin_bias),
            "network_rmse_vs_cryosat": net_rmse, "bilinear_rmse_vs_cryosat": bilin_rmse,
            "network_worse_rmse": net_rmse > bilin_rmse,
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "network_vs_bilinear_coastal_bias.csv"), index=False)
print(df.to_string(index=False))

print("\n=== Summary: coastal band only (all 7 months) ===")
coastal = df[df["region"] == "coastal"]
print(f"Months where |network bias| > |bilinear bias| (network worse): "
      f"{coastal['network_worse_bias'].sum()}/{len(coastal)}")
print(f"Months where network RMSE > bilinear RMSE (network worse): "
      f"{coastal['network_worse_rmse'].sum()}/{len(coastal)}")
print(f"Mean RMSE    -- network: {coastal['network_rmse_vs_cryosat'].mean():.4f}  "
      f"bilinear: {coastal['bilinear_rmse_vs_cryosat'].mean():.4f}  "
      f"ratio: {coastal['network_rmse_vs_cryosat'].mean() / coastal['bilinear_rmse_vs_cryosat'].mean():.4f}")

print("\n=== Summary: coastal band, HELD-OUT months only (Mar/Apr/Oct/Nov/Dec) ===")
coastal_heldout = coastal[coastal["fit_or_heldout"] == "held-out"]
print(f"Months where |network bias| > |bilinear bias| (network worse): "
      f"{coastal_heldout['network_worse_bias'].sum()}/{len(coastal_heldout)}")
print(f"Months where network RMSE > bilinear RMSE (network worse): "
      f"{coastal_heldout['network_worse_rmse'].sum()}/{len(coastal_heldout)}")
print(f"Mean RMSE    -- network: {coastal_heldout['network_rmse_vs_cryosat'].mean():.4f}  "
      f"bilinear: {coastal_heldout['bilinear_rmse_vs_cryosat'].mean():.4f}  "
      f"ratio: {coastal_heldout['network_rmse_vs_cryosat'].mean() / coastal_heldout['bilinear_rmse_vs_cryosat'].mean():.4f}")

print("\n=== Summary: interior (non-coastal ocean), all 7 months ===")
interior = df[df["region"] == "interior"]
print(f"Mean RMSE    -- network: {interior['network_rmse_vs_cryosat'].mean():.4f}  "
      f"bilinear: {interior['bilinear_rmse_vs_cryosat'].mean():.4f}")

print("\nDone. Saved:", os.path.join(OUT_DIR, "network_vs_bilinear_coastal_bias.csv"))
