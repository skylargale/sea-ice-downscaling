"""
observing_cryosat2_correlation_diagnosis.py

Follow-up on observing_cryosat2_validation_2020.py: Jan-Apr 2020 (CryoSat-2's
most reliable, full-coverage months) still showed weak/negative correlation
between the PIOMAS-driven network output and independent CryoSat-2 RDEFT4
truth. This digs into *why*: (1) coastal-band cells vs. open-ocean interior
cells separately (coastal bilinear-bleed artifacts are a documented issue
elsewhere in this project, functions_engressnet.coastal_band_mask), and
(2) whether the network's own output field simply lacks enough spatial
structure/variance at this domain scale to correlate meaningfully with
CryoSat's finer real structure in the first place -- a coarse-predictor
ceiling, not a network failure.
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_engressnet as fe

YEAR = 2020
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"
MONTHS = [1, 2, 3, 4]

RUN_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs"
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
OUT_DIR = os.path.join(RUN_DIR, "cryosat2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

fields = np.load(os.path.join(EVAL_DIR, "fields.npz"))
Y_pred_phys = fields["Y_pred_phys"][:, 0]
Y_base_phys = fields["Y_base_phys"][:, 0]
mask_test = fields["mask_test"]  # (365, 1, H, W)

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
print(f"Ocean cells: {ocean_hw.sum()} total -- coastal-band: {coastal_hw.sum()}, interior: {interior_ocean_hw.sum()}")

# ---------------------------------------------------------------------------
# Regrid CryoSat for Jan-Apr (reuse the weights already built)
# ---------------------------------------------------------------------------

dst_grid = xr.Dataset({"lat": ("lat", target_lat), "lon": ("lon", target_lon)})
regridder = xe.Regridder(
    xr.Dataset({"lat": (["y", "x"], np.zeros((448, 304))), "lon": (["y", "x"], np.zeros((448, 304)))}),
    dst_grid, method="bilinear", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/cryosat2_rdeft4_to_medium_domain.nc", reuse_weights=True,
)

rows = []
scatter_data = {"coastal": [], "interior": []}

for month in MONTHS:
    fname = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")][0]
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, fname))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)
    sit_da = sit.assign_coords(lat=(["y", "x"], ds["lat"].values), lon=(["y", "x"], ds["lon"].values % 360))
    cryo_map = regridder(sit_da, skipna=True).values
    ds.close()

    day_idx = np.where(sample_month == month)[0]
    net_map = Y_pred_phys[day_idx].mean(axis=0)
    bilin_map = Y_base_phys[day_idx].mean(axis=0)

    for label, region_mask in [("coastal", coastal_hw), ("interior", interior_ocean_hw)]:
        valid = region_mask & ~np.isnan(cryo_map)
        n = int(valid.sum())
        if n < 10:
            continue
        net_std = float(net_map[valid].std())
        cryo_std = float(cryo_map[valid].std())
        corr = float(np.corrcoef(net_map[valid], cryo_map[valid])[0, 1])
        bilin_corr = float(np.corrcoef(bilin_map[valid], cryo_map[valid])[0, 1])
        rmse = float(np.sqrt(np.mean((net_map[valid] - cryo_map[valid]) ** 2)))
        rows.append({
            "month": month, "region": label, "n_valid": n,
            "network_spatial_std": net_std, "cryosat_spatial_std": cryo_std,
            "std_ratio_network_over_cryosat": net_std / max(cryo_std, 1e-6),
            "network_corr_vs_cryosat": corr, "bilinear_corr_vs_cryosat": bilin_corr,
            "network_rmse_vs_cryosat": rmse,
        })
        scatter_data[label].append((net_map[valid], cryo_map[valid], month))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "cryosat2_correlation_diagnosis_jan_apr.csv"), index=False)
print(df.to_string(index=False))

print("\n=== Aggregated by region (mean over Jan-Apr) ===")
print(df.groupby("region")[["network_spatial_std", "cryosat_spatial_std", "std_ratio_network_over_cryosat",
                              "network_corr_vs_cryosat", "bilinear_corr_vs_cryosat", "network_rmse_vs_cryosat"]].mean())

# ---------------------------------------------------------------------------
# Scatter: network vs CryoSat, coastal vs interior, all 4 months pooled
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, label in zip(axes, ["coastal", "interior"]):
    all_net = np.concatenate([d[0] for d in scatter_data[label]])
    all_cryo = np.concatenate([d[1] for d in scatter_data[label]])
    ax.scatter(all_cryo, all_net, s=4, alpha=0.15, color="#2a78d6" if label == "interior" else "#eb6834")
    lims = [0, max(all_cryo.max(), all_net.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1")
    corr = np.corrcoef(all_net, all_cryo)[0, 1]
    ax.set_xlabel("CryoSat-2 SIT (m)")
    ax.set_ylabel("Network SIT (m)")
    ax.set_title(f"{label} cells, Jan-Apr pooled (r={corr:.3f}, n={len(all_cryo)})")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend()
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "cryosat2_scatter_coastal_vs_interior.png")
fig.savefig(fig_path, dpi=180, bbox_inches="tight")
print("\nSaved scatter figure:", fig_path)
print("Done.")
