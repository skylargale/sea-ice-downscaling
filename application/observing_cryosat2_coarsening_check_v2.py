"""
observing_cryosat2_coarsening_check.py

Follow-up on observing_cryosat2_correlation_diagnosis.py: is the weak
within-month spatial correlation against CryoSat-2 because the network's
fine-scale (0.1-degree) pattern is genuinely wrong, or because CryoSat's own
~25 km footprint can't resolve detail that fine in the first place (so any
comparison at full resolution is partly apples-to-oranges)? Tests this
directly: block-average both the network output and the (already
0.1-degree-regridded) CryoSat field over the same NxN blocks on the common
grid, at a few block sizes bracketing CryoSat's ~25 km footprint, and
recompute correlation at each coarsened scale. If correlation climbs
substantially as blocks grow toward ~25 km, the fine detail is the
unverified part, not the whole spatial pattern.
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
MONTHS = [1, 2, 3, 4]
CRYOSAT_DIR = "/glade/derecho/scratch/skygale/CryoSat2_RDEFT4"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

RUN_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs"
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
OUT_DIR = os.path.join(RUN_DIR, "cryosat2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

fields = np.load(os.path.join(EVAL_DIR, "fields.npz"))
Y_pred_phys = fields["Y_pred_phys"][:, 0]
Y_base_phys = fields["Y_base_phys"][:, 0]
mask_test = fields["mask_test"]

with open(os.path.join(EVAL_DIR, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
target_lat = np.asarray(tile_geometry[0]["target_lat"])   # (150,), 0.1 deg spacing
target_lon = np.asarray(tile_geometry[0]["target_lon"])   # (310,), 0.1 deg spacing
H, W = target_lat.size, target_lon.size

sample_times = pd.read_csv(os.path.join(EVAL_DIR, "sample_times.csv"))
sample_times["time"] = pd.to_datetime(sample_times["time"])
sample_month = sample_times["time"].dt.month.values

ocean_hw = mask_test[0, 0] <= 0.5
land_mask_hw = mask_test[0, 0]
coastal_hw = fe.coastal_band_mask(land_mask_hw[None, None], coastal_width=5)[0, 0].numpy()
interior_ocean_hw = ocean_hw & ~coastal_hw

# ---------------------------------------------------------------------------
# Regrid CryoSat-2 for Jan-Apr onto the common 0.1deg grid (reuse weights)
# ---------------------------------------------------------------------------

dst_grid = xr.Dataset({"lat": ("lat", target_lat), "lon": ("lon", target_lon)})
regridder = xe.Regridder(
    xr.Dataset({"lat": (["y", "x"], np.zeros((448, 304))), "lon": (["y", "x"], np.zeros((448, 304)))}),
    dst_grid, method="bilinear", periodic=False,
    filename=f"{WEIGHTED_GRIDS_DIR}/cryosat2_rdeft4_to_medium_domain.nc", reuse_weights=True,
)

cryo_maps, net_maps, bilin_maps = {}, {}, {}
for month in MONTHS:
    fname = [f for f in os.listdir(CRYOSAT_DIR) if f.startswith(f"RDEFT4_{YEAR}{month:02d}")][0]
    ds = xr.open_dataset(os.path.join(CRYOSAT_DIR, fname))
    sit = ds["sea_ice_thickness"].where(ds["sea_ice_thickness"] > -100)
    sit_da = sit.assign_coords(lat=(["y", "x"], ds["lat"].values), lon=(["y", "x"], ds["lon"].values % 360))
    cryo_maps[month] = regridder(sit_da, skipna=True).values
    ds.close()

    day_idx = np.where(sample_month == month)[0]
    net_maps[month] = Y_pred_phys[day_idx].mean(axis=0)
    bilin_maps[month] = Y_base_phys[day_idx].mean(axis=0)

# ---------------------------------------------------------------------------
# Block-average helper: NxN blocks in (lat, lon) index space, nanmean,
# require >=50% valid coverage per block to keep it.
# ---------------------------------------------------------------------------

def block_average(field, block, min_frac=0.5):
    Hc, Wc = H // block, W // block
    out = np.full((Hc, Wc), np.nan)
    valid_frac = np.zeros((Hc, Wc))
    for i in range(Hc):
        for j in range(Wc):
            patch = field[i * block:(i + 1) * block, j * block:(j + 1) * block]
            vf = np.isfinite(patch).mean()
            valid_frac[i, j] = vf
            if vf >= min_frac:
                out[i, j] = np.nanmean(patch)
    return out, valid_frac


# Approximate degrees-per-25km at this domain's latitudes (~65-75N):
# lat: 25/111 ~= 0.225 deg -> ~2 cells (0.1deg spacing)
# lon: 25/(111*cos(68deg)) ~= 0.60 deg -> ~6 cells (0.1deg spacing)
# Blocks tested are square in index space (simplification -- not exactly
# square in km given lat/lon cell-size anisotropy at these latitudes, but
# consistent between network and CryoSat so the comparison is still fair).
BLOCK_SIZES = {1: "0.1deg (~5x11km, native)", 3: "0.3deg (~15x33km)",
               6: "0.6deg (~30x66km, ~CryoSat footprint)", 10: "1.0deg (~50x110km)"}

rows = []
for block, label in BLOCK_SIZES.items():
    for region_name, region_mask in [("coastal", coastal_hw), ("interior", interior_ocean_hw)]:
        region_mask_coarse, _ = block_average(region_mask.astype(float), block, min_frac=0.5)
        region_valid = region_mask_coarse > 0.5

        pooled_net, pooled_cryo, pooled_bilin = [], [], []
        for month in MONTHS:
            cryo_c, cryo_vf = block_average(np.where(region_mask, cryo_maps[month], np.nan), block)
            net_c, _ = block_average(np.where(region_mask, net_maps[month], np.nan), block)
            bilin_c, _ = block_average(np.where(region_mask, bilin_maps[month], np.nan), block)

            valid = region_valid & np.isfinite(cryo_c) & np.isfinite(net_c)
            n = int(valid.sum())
            if n < 5:
                continue
            corr = float(np.corrcoef(net_c[valid], cryo_c[valid])[0, 1]) if n > 2 else np.nan
            bilin_corr = float(np.corrcoef(bilin_c[valid], cryo_c[valid])[0, 1]) if n > 2 else np.nan
            rmse = float(np.sqrt(np.mean((net_c[valid] - cryo_c[valid]) ** 2)))
            rows.append({
                "block_size_deg": round(block * 0.1, 2), "block_label": label, "month": month,
                "region": region_name, "n_valid_blocks": n,
                "network_corr_vs_cryosat": corr, "bilinear_corr_vs_cryosat": bilin_corr,
                "network_rmse_vs_cryosat": rmse,
            })
            pooled_net.append(net_c[valid]); pooled_cryo.append(cryo_c[valid]); pooled_bilin.append(bilin_c[valid])

        if pooled_net:
            pn, pc, pb = np.concatenate(pooled_net), np.concatenate(pooled_cryo), np.concatenate(pooled_bilin)
            if len(pn) > 2:
                rows.append({
                    "block_size_deg": round(block * 0.1, 2), "block_label": label, "month": "pooled_Jan-Apr",
                    "region": region_name, "n_valid_blocks": len(pn),
                    "network_corr_vs_cryosat": float(np.corrcoef(pn, pc)[0, 1]),
                    "bilinear_corr_vs_cryosat": float(np.corrcoef(pb, pc)[0, 1]),
                    "network_rmse_vs_cryosat": float(np.sqrt(np.mean((pn - pc) ** 2))),
                })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "cryosat2_coarsening_check.csv"), index=False)

print("=== Per-month correlation vs. block size ===")
per_month = df[df["month"] != "pooled_Jan-Apr"]
print(per_month.pivot_table(index=["region", "block_label"], columns="month",
                              values="network_corr_vs_cryosat").to_string())

print("\n=== Pooled (Jan-Apr) correlation vs. block size ===")
pooled = df[df["month"] == "pooled_Jan-Apr"][["region", "block_label", "n_valid_blocks",
                                                "network_corr_vs_cryosat", "bilinear_corr_vs_cryosat"]]
print(pooled.to_string(index=False))

print("\n=== Mean within-month correlation vs. block size (the key diagnostic) ===")
mean_within_month = per_month.groupby(["region", "block_label"])["network_corr_vs_cryosat"].mean()
print(mean_within_month.to_string())

print("\nDone. Saved:", os.path.join(OUT_DIR, "cryosat2_coarsening_check.csv"))
