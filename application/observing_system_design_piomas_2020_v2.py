"""
observing_system_design_piomas_2020.py

Paragraph 8 (Observing-system-design application): uses the per-day K=20
stochastic ensemble already produced by the PIOMAS observational-application
run (Paragraph 7, results/PIOMAS_obs_2020/FOSI_recommended_infer_2020_*)
to identify *where* the FOSI-trained network's downscaled coastal SIT is
least constrained across the 2020 PIOMAS-driven domain -- i.e. where new
coastal observations would do the most to reduce predictive uncertainty in
this product, bridging directly from the observational application to an
observing-system-design use case.

Method: for every ocean grid cell, take the ensemble spread (std across the
K=20 stochastic members) on each of the 365 days, then average over the
year -- a time-mean uncertainty map. Grid cells and named coastal
communities are ranked by this time-mean spread (absolute, in meters of
SIT) and by a relative version (spread / mean predicted thickness, since 0.1
m of spread means something different on 0.2 m of ice than on 2 m of ice).
This is a descriptive uncertainty map, not a formal OSSE (no synthetic
observation is actually assimilated back into the network) -- it answers
"where is this product least trustworthy", which is the natural first cut
at "where would new observations help most", not a full observing-network
optimization.

Caveat this inherits from Paragraph 7 (see
processing/build_X_Y_PIOMAS_obs_2020.py): the input's ice-state channel
(hi_d) is monthly-PIOMAS held constant within each month, so day-to-day
spread changes mostly reflect the network's own stochastic sampling plus
daily CDR concentration/JRA55 wind variation, not daily PIOMAS thickness
variability. The *spatial pattern* of the uncertainty map (which coastal
segments are worst) is the meaningful output here, not fine day-to-day
detail.
"""

import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_engressnet as fe

RUN_DIR = ("/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/"
           "FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs")
EVAL_DIR = os.path.join(RUN_DIR, "eval_data")
OUT_DIR = os.path.join(RUN_DIR, "observing_system_design")
os.makedirs(OUT_DIR, exist_ok=True)
print("Using run:", RUN_DIR)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

fields = np.load(os.path.join(EVAL_DIR, "fields.npz"))
preds_all_phys = fields["preds_all_phys"]   # (N=365, K=20, 1, H=150, W=310)
Y_pred_phys = fields["Y_pred_phys"]          # (N, 1, H, W) -- ensemble mean
mask_test = fields["mask_test"]              # (N, 1, H, W) -- 1=land, 0=ocean

with open(os.path.join(EVAL_DIR, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
target_lat = np.asarray(tile_geometry[0]["target_lat"])   # (150,)
target_lon = np.asarray(tile_geometry[0]["target_lon"])   # (310,)

point_df = pd.read_csv(os.path.join(EVAL_DIR, "candidate_point_timeseries.csv"))
with open(os.path.join(EVAL_DIR, "meta.json")) as f:
    meta = json.load(f)

N, K, C, H, W = preds_all_phys.shape
print(f"N={N} days, K={K} ensemble members, grid {H}x{W}")

# ---------------------------------------------------------------------------
# Time-mean ensemble spread map (uncertainty), ocean cells only
# ---------------------------------------------------------------------------

spread_daily = preds_all_phys.std(axis=1)[:, 0]          # (N, H, W)
mean_spread_map = spread_daily.mean(axis=0)               # (H, W)
mean_thickness_map = Y_pred_phys[:, 0].mean(axis=0)        # (H, W)

ocean_hw = mask_test[0, 0] <= 0.5                          # static across time
mean_spread_map_masked = np.where(ocean_hw, mean_spread_map, np.nan)
rel_spread_map = np.where(
    ocean_hw & (mean_thickness_map > 0.05),
    mean_spread_map / np.maximum(mean_thickness_map, 1e-6),
    np.nan,
)

print("Ocean-cell mean spread: mean=%.4f m, p90=%.4f m, max=%.4f m" % (
    np.nanmean(mean_spread_map_masked), np.nanpercentile(mean_spread_map_masked, 90),
    np.nanmax(mean_spread_map_masked),
))

# ---------------------------------------------------------------------------
# Grid-cell hotspot ranking (top 20 ocean cells by absolute spread)
# ---------------------------------------------------------------------------

flat_idx = np.argsort(np.where(ocean_hw, mean_spread_map, -np.inf).ravel())[::-1][:20]
iy_top, ix_top = np.unravel_index(flat_idx, mean_spread_map.shape)
hotspot_rows = []
for iy, ix in zip(iy_top, ix_top):
    hotspot_rows.append({
        "grid_iy": int(iy), "grid_ix": int(ix),
        "lat": float(target_lat[iy]), "lon_360": float(target_lon[ix]),
        "lon_180": float(((target_lon[ix] + 180) % 360) - 180),
        "mean_spread_m": float(mean_spread_map[iy, ix]),
        "mean_thickness_m": float(mean_thickness_map[iy, ix]),
        "rel_spread": float(mean_spread_map[iy, ix] / max(mean_thickness_map[iy, ix], 1e-6)),
    })
hotspot_df = pd.DataFrame(hotspot_rows)
hotspot_df.to_csv(os.path.join(OUT_DIR, "grid_cell_hotspots_top20.csv"), index=False)
print("\nTop 20 grid-cell uncertainty hotspots (ocean cells, absolute spread):")
print(hotspot_df[["lat", "lon_180", "mean_spread_m", "mean_thickness_m", "rel_spread"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Candidate-point (named coastal community) ranking
# ---------------------------------------------------------------------------

point_rows = []
for point_name, grp in point_df.groupby("point"):
    iy = int(grp["grid_iy"].iloc[0])
    ix = int(grp["grid_ix"].iloc[0])
    spread_ts = spread_daily[:, iy, ix]
    mean_ts = Y_pred_phys[:, 0, iy, ix]
    point_rows.append({
        "point": point_name,
        "grid_iy": iy, "grid_ix": ix,
        "dist_km_to_nearest_cell": float(grp["dist_km"].iloc[0]),
        "mean_spread_m": float(spread_ts.mean()),
        "mean_predicted_thickness_m": float(mean_ts.mean()),
        "rel_spread": float(spread_ts.mean() / max(mean_ts.mean(), 1e-6)),
    })
point_rank_df = pd.DataFrame(point_rows).sort_values("mean_spread_m", ascending=False).reset_index(drop=True)
point_rank_df.to_csv(os.path.join(OUT_DIR, "candidate_point_uncertainty_ranking.csv"), index=False)
print("\nCoastal community ranking by 2020 annual-mean ensemble spread (most- to least-uncertain):")
print(point_rank_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Spatial figure: time-mean spread map + candidate points + top hotspots
# ---------------------------------------------------------------------------

plot_bbox = {"lon_min": -182, "lon_max": -151, "lat_min": 60, "lat_max": 75}
proj, boundary_path, central_lon = fe.make_polar_proj(plot_bbox)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj})
fe.style_polar_ax(ax, proj, boundary_path, plot_bbox)

lon2d, lat2d = np.meshgrid(target_lon, target_lat)
pc = ax.pcolormesh(
    lon2d, lat2d, mean_spread_map_masked, transform=ccrs.PlateCarree(),
    cmap="magma", shading="auto",
)
cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85)
cb.set_label("2020 annual-mean ensemble spread (m SIT)")

for point_name, pt in meta["candidate_points"].items():
    ax.plot(pt["lon"], pt["lat"], marker="*", markersize=16, color="cyan",
             markeredgecolor="black", transform=ccrs.PlateCarree(), zorder=5)
    ax.annotate(point_name, xy=(pt["lon"], pt["lat"]), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                xytext=(4, 4), textcoords="offset points", fontsize=9, color="white",
                path_effects=None, zorder=6)

ax.scatter(hotspot_df["lon_360"], hotspot_df["lat"], transform=ccrs.PlateCarree(),
           s=25, facecolor="none", edgecolor="lime", linewidth=1.2, zorder=4,
           label="Top 20 uncertainty hotspot cells")
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Paragraph 8: PIOMAS-driven 2020 SIT ensemble uncertainty\n(where new coastal obs would help most)")

fig_path = os.path.join(OUT_DIR, "uncertainty_map_2020.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print("\nSaved figure:", fig_path)

print("\nDone. Outputs in:", OUT_DIR)
