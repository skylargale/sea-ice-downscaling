"""
observing_cryosat2_ensemble_calibration.py

Test B: a fairer accuracy standard for a *probabilistic* model. Every check
so far graded the network's ensemble MEAN against CryoSat-2 with RMSE --
but a stochastic model isn't wrong just because its mean misses; it's wrong
if CryoSat's true value falls OUTSIDE its own claimed uncertainty band. This
checks ensemble coverage: for each available 2020 month, build each of the
K=20 members' own monthly-mean field (same day-averaging convention used
for the point comparisons elsewhere), take the 5th/95th percentile across
those 20 members per cell, and check whether the regridded CryoSat-2 value
actually falls inside [p5, p95] -- split coastal vs. interior, same
definitions as every other check in this chain.

A well-calibrated 5-95% band should contain truth roughly 90% of the time.
Much lower than that = confidently wrong (the real problem). Much higher =
uselessly wide/uninformative uncertainty, not a virtue either.
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
MONTHS = [1, 2, 3, 4, 10, 11, 12]
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

RUN_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs"
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

rows = []
example_band = None
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
    # Per-member monthly mean: (K=20, H, W)
    member_month_mean = preds_all_phys[day_idx].mean(axis=0)
    p5 = np.percentile(member_month_mean, 5, axis=0)
    p95 = np.percentile(member_month_mean, 95, axis=0)
    band_width = p95 - p5
    inside = (cryo_map >= p5) & (cryo_map <= p95)

    if month == MONTHS[0] or (example_band is None):
        example_band = {"p5": p5, "p95": p95, "cryo": cryo_map, "month": month}

    for region_name, region_mask in [("coastal", coastal_hw), ("interior", interior_ocean_hw)]:
        valid = region_mask & np.isfinite(cryo_map)
        n = int(valid.sum())
        if n < 10:
            continue
        coverage = float(inside[valid].mean())
        mean_band_width = float(band_width[valid].mean())
        rows.append({
            "month": month, "region": region_name, "n_valid": n,
            "coverage_5_95_pct": coverage * 100, "mean_band_width_m": mean_band_width,
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "ensemble_calibration_vs_cryosat.csv"), index=False)
print(df.to_string(index=False))

print("\n=== Summary: ensemble 5-95% coverage of CryoSat-2 truth (ideal ~90%) ===")
summary = df.groupby("region")[["coverage_5_95_pct", "mean_band_width_m"]].mean()
print(summary.to_string())

# ---------------------------------------------------------------------------
# Figure: coverage bar chart by month/region, with the 90% target line
# ---------------------------------------------------------------------------

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 10: "Oct", 11: "Nov", 12: "Dec"}
months_order = [m for m in MONTHS if m in df["month"].values]
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(months_order))
w = 0.35
coastal_cov = [df[(df.month == m) & (df.region == "coastal")]["coverage_5_95_pct"].values[0]
               if len(df[(df.month == m) & (df.region == "coastal")]) else np.nan for m in months_order]
interior_cov = [df[(df.month == m) & (df.region == "interior")]["coverage_5_95_pct"].values[0]
                if len(df[(df.month == m) & (df.region == "interior")]) else np.nan for m in months_order]
ax.bar(x - w / 2, coastal_cov, width=w, color="#c0392b", label="Coastal")
ax.bar(x + w / 2, interior_cov, width=w, color="#2a78d6", label="Interior")
ax.axhline(90, color="#333", linestyle="--", linewidth=1.5, label="Ideal (90%)")
ax.set_xticks(x)
ax.set_xticklabels([MONTH_NAMES[m] for m in months_order])
ax.set_ylabel("% of cells where CryoSat-2 falls\ninside the network's 5-95% ensemble band")
ax.set_ylim(0, 100)
ax.set_title("Is the network honestly uncertain, or confidently wrong?\nEnsemble coverage of independent CryoSat-2 truth")
ax.legend()
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "ensemble_calibration_coverage.png")
fig.savefig(fig_path, dpi=180, bbox_inches="tight")
print("\nSaved figure:", fig_path)
print("Done.")
