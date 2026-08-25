"""
observing_test1_recalibration_v3conservative_sitbiasaug_noisesigma0p5.py

Same post-hoc spread-rescaling mechanism as observing_test1_recalibration.py
(fit a single scalar multiplier s on ensemble spread so the 5-95% band
covers ~90% of real CryoSat-2 truth, fit on Jan-Feb 2020, validated on
held-out Mar-Apr 2020), rerun against the noise_sigma=0.5 combined-fix run
(domain-randomization checkpoint + conservative-regridded PIOMAS input,
--noise-sigma 0.5 instead of the original 1.0 -- see
submit/evaluation/submit_infer_piomas_obs_2020_v3conservative_sitbiasaug_noisesigma0p5.sh).

Why rerun rather than reuse the old multiplier: the old s was fit against a
noise_sigma=1.0 ensemble's raw spread. Halving noise_sigma already shrinks
raw spread by a different amount before any rescaling, so the old
multiplier doesn't transfer -- a fresh fit is needed to know the real
calibration cost of the noise_sigma=0.5 speckle fix.
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
FIT_MONTHS = [1, 2]
VALIDATE_MONTHS = [3, 4]
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

RUN_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/FOSI_sitbiasaug_infer_2020_daily_v3_conservative_noisesigma0p5_5717974.casper-pbs"
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
OUT_DIR = os.path.join(RUN_DIR, "cryosat2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

fields = np.load(os.path.join(EVAL_DIR, "fields.npz"))
preds_all_phys = fields["preds_all_phys"][:, :, 0]   # (365, K=20, H, W)
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


def get_cryo_map(month):
    matches = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")]
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, matches[0]))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)
    sit_da = sit.assign_coords(lat=(["y", "x"], ds["lat"].values), lon=(["y", "x"], ds["lon"].values % 360))
    cryo_map = regridder(sit_da, skipna=True).values
    ds.close()
    return cryo_map


def coverage_for_scale(months, s, region_mask):
    hits, total = 0, 0
    for month in months:
        cryo_map = get_cryo_map(month)
        day_idx = np.where(sample_month == month)[0]
        member_month_mean = preds_all_phys[day_idx].mean(axis=0)  # (K, H, W)
        ens_mean = member_month_mean.mean(axis=0)
        scaled = ens_mean[None] + s * (member_month_mean - ens_mean[None])
        p5 = np.percentile(scaled, 5, axis=0)
        p95 = np.percentile(scaled, 95, axis=0)
        inside = (cryo_map >= p5) & (cryo_map <= p95)
        valid = region_mask & np.isfinite(cryo_map)
        hits += inside[valid].sum()
        total += valid.sum()
    return 100 * hits / total


results = []
for region_name, region_mask in [("coastal", coastal_hw), ("interior", interior_ocean_hw)]:
    print(f"\n=== Fitting scale factor on {region_name}, months {FIT_MONTHS} ===")
    best_s, best_diff = None, np.inf
    for s in [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 400]:
        cov = coverage_for_scale(FIT_MONTHS, s, region_mask)
        diff = abs(cov - 90)
        print(f"  s={s:>4}  fit-coverage={cov:6.2f}%")
        if diff < best_diff:
            best_diff, best_s = diff, s
    val_cov_raw = coverage_for_scale(VALIDATE_MONTHS, 1, region_mask)
    val_cov_scaled = coverage_for_scale(VALIDATE_MONTHS, best_s, region_mask)
    print(f"  >> best s={best_s}")
    print(f"  Held-out validation (months {VALIDATE_MONTHS}): raw coverage={val_cov_raw:.2f}%  "
          f"scaled (s={best_s}) coverage={val_cov_scaled:.2f}%")
    results.append({"region": region_name, "best_scale_s": best_s,
                     "validation_coverage_raw_pct": val_cov_raw,
                     "validation_coverage_scaled_pct": val_cov_scaled})

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUT_DIR, "test1_recalibration_scale_factors.csv"), index=False)
print("\n", df.to_string(index=False))
print("\nDone.")
