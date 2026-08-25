"""
observing_cryosat2_check_biascorrected.py

Test 1 result check: does bias-correcting PIOMAS's coastal hi_d input
(processing/build_X_PIOMAS_obs_2020_biascorrected.py) let the network
beat bilinear against independent CryoSat-2 truth in the coastal band,
where the uncorrected run showed the network making PIOMAS's coastal bias
WORSE (6/7 months, see cryosat2_validation/network_vs_bilinear_coastal_bias.csv)?
Same method as observing_cryosat2_network_vs_bilinear_coastal.py, pointed
at the bias-corrected run instead.
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
import functions_engressnet as fe

YEAR = 2020
MONTHS = [1, 2, 3, 4, 10, 11, 12]
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

RUN_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_biascorrected_5714517.casper-pbs"
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
            "month": month, "region": region_name, "n_valid": n,
            "network_bias_vs_cryosat": net_bias, "bilinear_bias_vs_cryosat": bilin_bias,
            "network_worse_bias": abs(net_bias) > abs(bilin_bias),
            "network_rmse_vs_cryosat": net_rmse, "bilinear_rmse_vs_cryosat": bilin_rmse,
            "network_worse_rmse": net_rmse > bilin_rmse,
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "network_vs_bilinear_coastal_bias_BIASCORRECTED.csv"), index=False)
print(df.to_string(index=False))

for region_name in ["coastal", "interior"]:
    sub = df[df["region"] == region_name]
    print(f"\n=== {region_name} (bias-corrected input) ===")
    print(f"Months network worse on bias: {sub['network_worse_bias'].sum()}/{len(sub)}")
    print(f"Months network worse on RMSE: {sub['network_worse_rmse'].sum()}/{len(sub)}")
    print(f"Mean |bias| -- network: {sub['network_bias_vs_cryosat'].abs().mean():.4f}  "
          f"bilinear: {sub['bilinear_bias_vs_cryosat'].abs().mean():.4f}")
    print(f"Mean RMSE   -- network: {sub['network_rmse_vs_cryosat'].mean():.4f}  "
          f"bilinear: {sub['bilinear_rmse_vs_cryosat'].mean():.4f}")

# ---------------------------------------------------------------------------
# Direct before/after comparison against the uncorrected run's result
# ---------------------------------------------------------------------------

uncorrected_path = ("/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/"
                     "FOSI_recommended_infer_2020_daily_5705277.casper-pbs/cryosat2_validation/"
                     "network_vs_bilinear_coastal_bias.csv")
if os.path.exists(uncorrected_path):
    before = pd.read_csv(uncorrected_path)
    before_coastal = before[before["region"] == "coastal"]
    after_coastal = df[df["region"] == "coastal"]
    print("\n=== BEFORE (uncorrected) vs AFTER (bias-corrected), coastal RMSE ===")
    comp = before_coastal[["month", "network_rmse_vs_cryosat", "bilinear_rmse_vs_cryosat"]].merge(
        after_coastal[["month", "network_rmse_vs_cryosat", "bilinear_rmse_vs_cryosat"]],
        on="month", suffixes=("_before", "_after")
    )
    print(comp.to_string(index=False))
    print(f"\nMean network coastal RMSE: before={comp['network_rmse_vs_cryosat_before'].mean():.4f}  "
          f"after={comp['network_rmse_vs_cryosat_after'].mean():.4f}")

print("\nDone.")
