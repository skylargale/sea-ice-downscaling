"""
Paper Fig. 3 (example fields) from the 100-member EMA run (2026-10-18; perfect-model accuracy is reported with 100 members).
Generated from figures/all_figures_by_run.ipynb cells 2 (setup) and 15 (example fields) with RUN = MESA_final_ema_K100;
same samples (rng seed 0 on the truth), deterministic panel = det-MSE + EMA. Writes final_figures/03_example_fields_K100.png.
"""
import matplotlib
matplotlib.use("Agg")

import os
import glob
import json
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nicopal as ncp
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm

BASE = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(BASE, "training"))
sys.path.insert(0, os.path.join(BASE, "evaluation"))
import run_daily_eval_batch as rdeb
import functions_engressnet as fe

# ---- Choose run here ----
RUN    = "MESA_final_ema_K100"  # final EngressNet (2026-10-18: 2026-10-07 configuration + EMA; was MESA_final_avg) -- subdirectory under results/ and evaluation/saved_figs/
WINDOW = "2015-2020"       # training-window substring; only used by the raw-npz (map) cells


def resolve_run_window_dir(base_subdir, run, window):
    """Find the single results/ or evaluation/saved_figs/ subfolder for (run, window). Folder names
    embed a PBS job id (e.g. MESA_noisecasc_avg_2015-2020_2021_5956149.casper-pbs), so this
    globs on the window substring rather than requiring the exact folder name."""
    pattern = os.path.join(BASE, base_subdir, run, f"*{window}*")
    matches = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern!r}, found {len(matches)}: {matches}")
    return matches[0]


def agg_dir(run):
    return os.path.join(BASE, "evaluation", "saved_figs", run, "_4window_aggregate")



RESULTS_DIR = resolve_run_window_dir("results", RUN, WINDOW)
print('RESULTS_DIR', RESULTS_DIR)
edir = os.path.join(RESULTS_DIR, "eval_data")
with open(os.path.join(edir, "meta.json")) as f:
    meta = json.load(f)
bbox = meta["bbox"]
candidate_points = {**meta["candidate_points"], "Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
with open(os.path.join(edir, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
fields = np.load(os.path.join(edir, "fields.npz"))
X_test_sit_phys = fields["X_test_sit_phys"]
Y_base_phys = fields["Y_base_phys"]
Y_pred_det_phys = fields["Y_pred_det_phys"]
preds_all_phys = fields["preds_all_phys"]
Y_pred_phys = fields["Y_pred_phys"]
Y_test_phys = fields["Y_test_phys"]
test_tile_ids = fields["test_tile_ids"]

# "Deterministic" panel = the separately MSE-trained U-Net (same backbone, window and test set)
DET_RUN = ("MESA" if RUN.startswith("MESA") else "FOSI") + "_detmse_ema"  # det-MSE + EMA baseline for this dataset (2026-10-18)
if DET_RUN != RUN and os.path.isdir(os.path.join(BASE, "results", DET_RUN)):
    det_fields = np.load(os.path.join(resolve_run_window_dir("results", DET_RUN, WINDOW), "eval_data", "fields.npz"))
    assert np.array_equal(det_fields["Y_test_phys"], Y_test_phys), "MSE run's test set differs"
    Y_pred_det_phys = det_fields["Y_pred_det_phys"]
else:
    print(f"No MSE deterministic batch for {RUN}; 'Deterministic' panel is the zero-noise pass.")

proj, boundary_path, central_lon = rdeb.make_polar_proj(bbox)


def _style(ax, lon_=None, lat_=None):
    rdeb.style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_, lat_)


N_SAMPLES = min(3, Y_test_phys.shape[0])
MIN_ICE_THICKNESS = 0.5
MIN_ICE_FRAC = 0.25
ice_frac = (Y_test_phys[:, 0] > MIN_ICE_THICKNESS).mean(axis=(1, 2))
valid_idxs = np.where(ice_frac > MIN_ICE_FRAC)[0]
SAMPLE_IDXS = np.random.default_rng(0).choice(valid_idxs, N_SAMPLES, replace=False)
MEMBER_IDX = min(4, preds_all_phys.shape[1] - 1)
VMIN, VMAX = 0, 3
CMAP = "Blues"

panel_titles = ["Low-Res Input", "Bilinear", "Deterministic", "One Member", "Ensemble Mean", "High-Res Truth"]
fig, axs = plt.subplots(len(SAMPLE_IDXS), 6, figsize=(18, 3.2 * len(SAMPLE_IDXS)), constrained_layout=True, subplot_kw={"projection": proj})
if len(SAMPLE_IDXS) == 1:
    axs = axs[None, :]
for row, idx in enumerate(SAMPLE_IDXS):
    geo = tile_geometry[int(test_tile_ids[idx])]
    ctx_lon, ctx_lat = geo["context_lon"], geo["context_lat"]
    tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]
    fields_row = [
        X_test_sit_phys[idx, 0], Y_base_phys[idx, 0], Y_pred_det_phys[idx, 0],
        preds_all_phys[idx, MEMBER_IDX, 0], Y_pred_phys[idx, 0], Y_test_phys[idx, 0],
    ]
    lons = [ctx_lon, tgt_lon, tgt_lon, tgt_lon, tgt_lon, tgt_lon]
    lats = [ctx_lat, tgt_lat, tgt_lat, tgt_lat, tgt_lat, tgt_lat]
    for col, (field, lon_, lat_) in enumerate(zip(fields_row, lons, lats)):
        ax = axs[row, col]
        im = ax.pcolormesh(lon_, lat_, field, transform=ccrs.PlateCarree(), cmap=CMAP, vmin=VMIN, vmax=VMAX, shading="auto")
        _style(ax, lon_, lat_)
        if row == 0:
            ax.set_title(panel_titles[col], fontsize=18)
    axs[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=13)
cbar = fig.colorbar(im, ax=axs, aspect=30, shrink=0.9, pad=0.02)
cbar.set_label("Sea ice thickness (m)", fontsize=18)
cbar.ax.tick_params(labelsize=14)
plt.savefig(os.path.join(BASE, 'figures', 'final_figures', '03_example_fields_K100.png'), dpi=300, bbox_inches='tight')
