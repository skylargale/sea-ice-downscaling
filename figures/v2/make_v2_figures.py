# %% converter cell (not part of the notebook)
"""
make_v2_figures.py: every figures/v2 figure (paper figures, table charts, supplementary set) in one file.

The code is split into cells marked "# %%". Three ways to use it:
  python make_v2_figures.py                     run every cell top to bottom (what submit_make_v2_figures.sh does)
  python make_v2_figures.py --write-notebook    write make_v2_figures.ipynb from this file (for JupyterHub)
  python make_v2_figures.py --from-notebook     copy edits made in make_v2_figures.ipynb back into this file
In VS Code the "# %%" cells also run one at a time ("Run Cell"). Map figures need a compute node (about 64 GB).
This file is the source of truth: --write-notebook overwrites the notebook, --from-notebook overwrites this file
(keeping this first cell).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PY, _NB = os.path.join(_HERE, "make_v2_figures.py"), os.path.join(_HERE, "make_v2_figures.ipynb")
_MARK = "# %%"


def _py_to_cells(text):
    blocks, cur = [], None
    for line in text.splitlines():
        if line.startswith(_MARK):
            cur = {"md": "[markdown]" in line, "lines": []}
            blocks.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    out = []
    for b in blocks[1:]:  # block 0 = this converter cell, not part of the notebook
        src = "\n".join(b["lines"]).strip("\n")
        if b["md"]:
            src = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in src.splitlines())
        out.append({"cell_type": "markdown" if b["md"] else "code", "metadata": {}, "source": src,
                    **({} if b["md"] else {"execution_count": None, "outputs": []})})
    for i, c in enumerate(out):
        c["id"] = f"cell-{i:02d}"
    return out


if __name__ == "__main__" and "--write-notebook" in sys.argv:
    nb = {"cells": _py_to_cells(open(_PY).read()), "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "seaice-downscaling", "language": "python", "name": "seaice-downscaling"},
                       "language_info": {"name": "python"}}}
    json.dump(nb, open(_NB, "w"), indent=1)
    print(f"wrote {_NB} ({len(nb['cells'])} cells)")
    sys.exit(0)
if __name__ == "__main__" and "--from-notebook" in sys.argv:
    text = open(_PY).read()
    first = text[:text.index("\n" + _MARK, text.index(_MARK) + 1) + 1]  # keep this converter cell
    parts = []
    for c in json.load(open(_NB))["cells"]:
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        if c["cell_type"] == "markdown":
            parts.append(_MARK + " [markdown]\n" + "\n".join("# " + l if l else "#" for l in src.splitlines()))
        else:
            parts.append(_MARK + "\n" + src)
    open(_PY, "w").write(first + "\n\n".join(parts) + "\n")
    print(f"updated {_PY} from {_NB}")
    sys.exit(0)


# %% [markdown]
# # figures/v2: all revision figures in one notebook
#
# **How to use**
# 1. Run this on a Casper compute node (JupyterHub "Casper PBS batch", about 64 GB of memory). The map figures load large
#    `fields.npz` files, so the login node runs out of memory.
# 2. Kernel: `seaice-downscaling`.
# 3. Run **Setup** and then **Data**, once. After that, every figure cell runs on its own, in any order.
# 4. To change a figure, edit its cell and re-run it. The cell saves the PNG into `figures/v2/` (same file name as now, which
#    overwrites the current file) and shows it below the cell.
#
# **Model:** `FINAL_SET = "final2"` in Setup is the revised final model (current paper figures, written to `figures/v2/`).
# `"final1"` is the submitted-paper model; its figures are written to `figures/v2/superseded_final1/` instead.
#
# **"Deterministic" in the supplementary figures** is the deterministic U-Net (det-MSE) trained on the same days, taken from
# its own run (the Data cell swaps it in). The old `figures/final_figures/` versions used EngressNet's zero-noise pass under
# that label.
#
# Colours for method-comparison charts come from `METHOD_COLORS` (Setup); map figures keep their own colormaps.
#
# This notebook is written from `make_v2_figures.py` (`python make_v2_figures.py --write-notebook`). To keep edits made
# here, copy them back with `python make_v2_figures.py --from-notebook` before regenerating.

# %% [markdown]
# ## Setup

# %%
try:  # notebook / VS Code: inline figures; plain python: no display
    get_ipython().run_line_magic("matplotlib", "inline")
    _IN_NOTEBOOK = True
except NameError:
    import matplotlib
    matplotlib.use("Agg")
    _IN_NOTEBOOK = False
# ---- choose the model -------------------------------------------------------------------------------------------
FINAL_SET = "final2"   # "final2" = revised final model (paper); "final1" = submitted model -> figures/v2/superseded_final1/

import importlib
import os
import shutil
import sys
os.environ["FINAL_SET"] = FINAL_SET
V2 = "/glade/work/skygale/projects/SeaIceDownscaling/figures/v2"
if "/glade/work/skygale/projects/SeaIceDownscaling/evaluation/revisions" not in sys.path:
    sys.path.insert(0, "/glade/work/skygale/projects/SeaIceDownscaling/evaluation/revisions")
import final_set  # noqa: E402  (evaluation/revisions/final_set.py: batch names of each final model)
importlib.reload(final_set)  # pick up a changed FINAL_SET
FS = final_set.SETS[FINAL_SET]

# one method palette for every comparison chart (validated categorical order: adjacent-pair CVD dE >= 8);
# new baselines take the next free slot. Truth and observations are drawn in ink. Map figures keep their own colormaps.
INK, MUTED = "#0b0b0b", "#8a8984"
TRUTH_COLOR = INK
METHOD_COLORS = {
    "EngressNet": "#2a78d6",                   # 1 blue
    "Deterministic U-Net": "#eb6834",          # 2 orange
    "Bilinear": "#1baf7a",                     # 3 aqua
    "Seed/window ensemble (det.)": "#eda100",  # 4 yellow
    "MC-dropout U-Net": "#e87ba4",             # 5 magenta (final1 tables only)
    "EngressNet (mult noise)": "#008300",      # 6 green
    "EngressNet (add_feat noise)": "#6250d6",  # 7 violet
}
OUT_DIR = V2 if FINAL_SET == "final2" else os.path.join(V2, "superseded_final1")  # final1 never overwrites the paper set


def out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)
def show(*names, width=900):
    """Display saved figures under the cell (names relative to the output folder, or absolute paths)."""
    for n in names:
        p = n if os.path.isabs(n) else out(n)
        if not os.path.exists(p):
            print("not found:", p)
        elif _IN_NOTEBOOK:
            from IPython.display import Image, display
            display(Image(filename=p, width=width))
        else:
            print("wrote", p)


print("FINAL_SET:", FINAL_SET, "| figures ->", OUT_DIR)

# ---- paper-notebook setup (figures/all_figures_by_run.ipynb), pointed at the chosen model -----------------------
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
RUN    = FS["MESA"]["eng"]   # EngressNet batch for the chosen FINAL_SET
WINDOW = FS["map_window"]    # training-window substring of the run folder


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
SAVED_FIGS_WINDOW_DIR = resolve_run_window_dir("evaluation/saved_figs", RUN, WINDOW)
AGG_DIR = agg_dir(RUN)

print("RUN            :", RUN)
print("WINDOW         :", WINDOW)
print("RESULTS_DIR    :", RESULTS_DIR)
print("SAVED_FIGS_DIR :", SAVED_FIGS_WINDOW_DIR)
print("AGG_DIR        :", AGG_DIR, "exists:", os.path.isdir(AGG_DIR))

# %% [markdown]
# ## Data: put the real deterministic U-Net into the aggregate files
#
# The EngressNet run's aggregate CSVs (`evaluation/saved_figs/<run>/_4window_aggregate/`) have a "deterministic" column,
# but it is EngressNet's own zero-noise pass. This cell copies the files the figures use to `figures/v2/data_<FINAL_SET>/`,
# replaces that column with the det-MSE U-Net run on the same days (`FS["MESA"]["det"]`), using the same pooling as
# `evaluation/build_4window_aggregate_figures.py` (mean over members and windows), and checks that the truth matches. The
# map cells read the det U-Net from `DET_FIELDS`.

# %%
import glob
import numpy as np
import pandas as pd

DET_RUN = FS["MESA"]["det"]
DATA_DIR = os.path.join(V2, f"data_{FINAL_SET}", "MESA_aggregate")
os.makedirs(DATA_DIR, exist_ok=True)
ENG_AGG = agg_dir(RUN)
det_wins = sorted(d for d in glob.glob(os.path.join(BASE, "evaluation", "saved_figs", DET_RUN, "*")) if os.path.isdir(d)
                  and not d.endswith("_4window_aggregate"))
print("EngressNet aggregate:", ENG_AGG)
print("det U-Net windows   :", [os.path.basename(d) for d in det_wins])


def det_cat(fname, **kw):
    return pd.concat([pd.read_csv(os.path.join(d, fname), index_col=0, **kw) for d in det_wins], ignore_index=True)


# files used unchanged (EngressNet-only quantities)
for f in ("13b_psd_per_member_spread_data_4window.csv", "14_rank_histogram_data_4window.csv",
          "14b_rank_histogram_by_regime_data_4window.csv", "15_spread_skill_by_regime_data_4window.csv",
          "16_crps_by_regime_data_4window.csv"):
    shutil.copy2(os.path.join(ENG_AGG, f), os.path.join(DATA_DIR, f))

# 01 domain-mean SIT and 01b pattern correlation time series: det column = det run, pooled by date
for agg_f, win_f, has_truth in (("01_domain_mean_sit_timeseries_data_4window.csv", "01_domain_mean_sit_timeseries_data.csv", True),
                                ("01b_domain_mean_pattern_corr_timeseries_data_4window.csv", "01b_domain_mean_pattern_corr_timeseries_data.csv", False)):
    a = pd.read_csv(os.path.join(ENG_AGG, agg_f), index_col=0, parse_dates=["time"])
    g = det_cat(win_f, parse_dates=["time"]).groupby("time").mean(numeric_only=True).reindex(a["time"])
    if has_truth:
        assert np.allclose(g["truth"].values, a["truth"].values, atol=1e-5), "truth differs between runs"
    a["deterministic_unet"] = g["deterministic_unet"].values
    a.to_csv(os.path.join(DATA_DIR, agg_f))

# 02 community nearest-cell series
a = pd.read_csv(os.path.join(ENG_AGG, "02_candidate_point_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])
g = det_cat("02_candidate_point_timeseries_data.csv", parse_dates=["time"]).groupby(["point", "method", "time"])["value"].mean()
key = pd.MultiIndex.from_frame(a[["point", "method", "time"]])
t = a["method"] == "truth"
assert np.allclose(g.reindex(key)[t.values].values, a.loc[t, "value"].values, atol=1e-5), "truth differs between runs"
d = a["method"] == "deterministic_unet"
a.loc[d, "value"] = g.reindex(key)[d.values].values
a.to_csv(os.path.join(DATA_DIR, "02_candidate_point_timeseries_data_4window.csv"))

# 11 Taylor statistics (mean over windows) and 13 PSD (mean over windows)
a = pd.read_csv(os.path.join(ENG_AGG, "11_taylor_stats_4window.csv"), index_col=0)
dt = det_cat("11_taylor_stats.csv").groupby("Method").mean(numeric_only=True)
for c in dt.columns:
    a.loc[a["Method"] == "Deterministic UNet", c] = dt.loc["Deterministic UNet", c]
a.to_csv(os.path.join(DATA_DIR, "11_taylor_stats_4window.csv"))
a = pd.read_csv(os.path.join(ENG_AGG, "13_psd_data_4window.csv"), index_col=0)
dp = det_cat("13_psd_data.csv").groupby("wavenumber_cycles_per_km", sort=False).mean(numeric_only=True)
assert np.allclose(dp["Truth"].values, a["Truth"].values, rtol=1e-4), "truth PSD differs between runs"
a["Deterministic UNet"] = dp["Deterministic UNet"].values
a.to_csv(os.path.join(DATA_DIR, "13_psd_data_4window.csv"))

AGG_DIR = DATA_DIR  # every figure below reads the aggregate files from here

# det U-Net fields for the map cells (same window, same test dates)
DET_FIELDS = np.load(os.path.join(resolve_run_window_dir("results", DET_RUN, WINDOW), "eval_data", "fields.npz"))
_eng = np.load(os.path.join(RESULTS_DIR, "eval_data", "fields.npz"))
assert np.allclose(DET_FIELDS["Y_test_phys"][:20], _eng["Y_test_phys"][:20]), "test dates differ between runs"
del _eng
print("Data ready:", DATA_DIR)

# %% [markdown]
# # Paper figures

# %% [markdown]
# ## Fig. 2: domain overview map

# %% [markdown]
# Paper Fig. 2 (domain overview), revised 2026-11-13. Same frame as the original figure (all_figures_by_run.ipynb cell 4):
# polar stereographic with the rounded domain boundary (rdeb.make_polar_proj on the run bbox), rdeb.style_polar_ax for
# coastlines / gridlines / community stars, and the 0.1 deg training sub-domain outlined in tab:blue. Added: a muted ETOPO2
# relief on land (NOAA, GDEX d759003), light-blue ocean, sea and region names, an Alaska locator inset and a scale bar.
# Communities at the exact coordinates the evaluation uses (eval_data/meta.json + rdeb's Point Hope).
# Output: figures/v2/fig02_domain_overview.png (+ .pdf).

# %%
import matplotlib
_rc_saved = matplotlib.rcParams.copy()

import glob
import json
import os
import pickle
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "evaluation"))
import run_daily_eval_batch as rdeb  # noqa: E402

ETOPO = "/glade/derecho/scratch/skygale/downscaling/ancillary/maps/etopo2_2006apr.nc"  # gunzip of d759003/etopo2_2006apr/etopo2_2006apr.nc.gz
CACHE = "/glade/derecho/scratch/skygale/downscaling/ancillary/maps/etopo2_chukchi_subset.npz"

edir = os.path.join(glob.glob(f"{BASE}/results/MESA_final_ema/*2015-2020*")[0], "eval_data")
meta = json.load(open(os.path.join(edir, "meta.json")))
bbox = {**meta["bbox"], "lat_min": 54.0}  # southern edge extended 60 -> 54 N (2026-11-13) to show more of the Bering Sea
candidate_points = {**meta["candidate_points"], "Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
geo0 = pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb"))[0]
for k, v in candidate_points.items():
    print(f"{k:11s} {v['lat']:.4f} N  {360 - v['lon']:.4f} W")

# ---- ETOPO2 subset (cached), longitudes 0-360 so the dateline is interior
if not os.path.exists(CACHE):
    d = xr.open_dataset(ETOPO)
    sub = d.z.sel(y=slice(48, 86))
    lon = sub.x.values % 360
    keep = (lon >= 145) & (lon <= 250)
    o = np.argsort(lon[keep])
    np.savez_compressed(CACHE, z=sub.values[:, keep][:, o].astype(np.float32), lon=lon[keep][o], lat=sub.y.values)
c = np.load(CACHE)
lon360, ui = np.unique(c["lon"], return_index=True)
z, lat = c["z"][:, ui], c["lat"]

land_cmap = LinearSegmentedColormap.from_list("land", ["#c5d6a4", "#d9e0bb", "#e8e6d6", "#f5f4ef"])
ls = LightSource(azdeg=315, altdeg=45)
dy = 111e3 * 2 / 60
rgb_land = ls.shade(np.clip(z, 0, 1800), cmap=land_cmap, vmin=0, vmax=1800, blend_mode="soft", vert_exag=6,
                    dx=dy * np.cos(np.deg2rad(68)), dy=dy)[..., :3]
sea = np.array([0.80, 0.90, 0.96])
rgb = np.where((z > 0)[..., None], rgb_land, sea)

# ---- map (same projection and frame as the original figure)
proj, boundary_path, central_lon = rdeb.make_polar_proj(bbox)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.imshow(rgb, origin="lower", extent=[lon360.min() - 180, lon360.max() - 180, lat.min(), lat.max()],
          transform=ccrs.PlateCarree(central_longitude=180), regrid_shape=1500, interpolation="bilinear", zorder=0.5)
# clip the 2' relief to the 10 m coastline (otherwise land colour bleeds past the coast)
ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "10m"), facecolor=sea, edgecolor="none", zorder=0.55)
ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m"), facecolor="#cce5f5", edgecolor="none", zorder=0.6)
ax.add_feature(cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "10m"), facecolor="none",
               edgecolor="#9cc8e6", lw=0.4, zorder=0.6)
rdeb.style_polar_ax(ax, proj, boundary_path, bbox, candidate_points)

lon_t, lat_t = geo0["target_lon"], geo0["target_lat"]
x0, x1, y0, y1 = float(np.min(lon_t)), float(np.max(lon_t)), float(np.min(lat_t)), float(np.max(lat_t))
t = np.linspace(0, 1, 100)
box_lon = np.r_[x0 + (x1 - x0) * t, np.full(100, x1), x1 - (x1 - x0) * t, np.full(100, x0)]
box_lat = np.r_[np.full(100, y0), y0 + (y1 - y0) * t, np.full(100, y1), y1 - (y1 - y0) * t]
ax.plot(box_lon, box_lat, color="tab:blue", linewidth=1.5, transform=ccrs.PlateCarree())
ax.text(-166.5, 73.6, "Model training domain", color="tab:blue", fontsize=9, ha="center", va="center",
        transform=ccrs.PlateCarree())

for txt, lo, la, kw in [("Chukchi Sea", -169.5, 71.0, dict(style="italic", color="#2b5d8c", fontsize=10)),
                        ("Bering Sea", -175.0, 58.5, dict(style="italic", color="#2b5d8c", fontsize=10)),
                        ("Arctic Ocean", -160.0, 76.5, dict(style="italic", color="#2b5d8c", fontsize=10)),
                        ("Alaska", -153.0, 66.0, dict(color="#4d4d4d", fontsize=11)),
                        ("Russia", -186.0, 66.2, dict(color="#4d4d4d", fontsize=11))]:
    ax.text(lo, la, txt, ha="center", va="center", transform=ccrs.PlateCarree(), **kw)

# simple 200 km scale bar near the bottom of the domain
xa, ya = proj.transform_point(-187.0, 56.5, ccrs.PlateCarree())
ax.plot([xa, xa + 200e3], [ya, ya], color="k", lw=1.5, transform=proj)
ax.text(xa + 100e3, ya + 25e3, "200 km", ha="center", va="bottom", fontsize=8, transform=proj)

# Alaska locator inset in the empty upper-left corner of the rounded frame
iax = fig.add_axes([0.06, 0.70, 0.24, 0.20], projection=ccrs.LambertConformal(central_longitude=-166.5, standard_parallels=(60, 75)))
iax.set_extent([-206, -127, 52, 84], crs=ccrs.PlateCarree())
iax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "110m"), facecolor="white", edgecolor="none")
iax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), facecolor="lightgray", edgecolor="0.4", lw=0.4)
iax.plot(box_lon, box_lat, color="k", lw=1.1, transform=ccrs.PlateCarree())

plt.savefig(out("fig02_domain_overview.png"), dpi=300, bbox_inches="tight")
plt.savefig(out("fig02_domain_overview.pdf"), bbox_inches="tight")
print("wrote", out("fig02_domain_overview.png"))


matplotlib.rcParams.update(_rc_saved)  # undo this cell's style changes
show('fig02_domain_overview.png')

# %% [markdown]
# ## Fig. 3: domain and resolution (shared colourbar vmax 3 m)
#
# Output: `fig03_domain_resolution.png`

# %%
edir = os.path.join(RESULTS_DIR, "eval_data")
with open(os.path.join(edir, "meta.json")) as f:
    meta = json.load(f)
bbox = meta["bbox"]
with open(os.path.join(edir, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
geo0 = tile_geometry[0]
ctx_lon, ctx_lat = geo0["context_lon"], geo0["context_lat"]
tgt_lon, tgt_lat = geo0["target_lon"], geo0["target_lat"]

fields = np.load(os.path.join(edir, "fields.npz"))
X_test_sit_phys = fields["X_test_sit_phys"]
Y_test_phys = fields["Y_test_phys"]

MIN_ICE_THICKNESS = 0.5
MIN_ICE_FRAC = 0.25
ice_frac = (Y_test_phys[:, 0] > MIN_ICE_THICKNESS).mean(axis=(1, 2))
SAMPLE_IDX = 150  # int(np.where(ice_frac > MIN_ICE_FRAC)[0][0])

x_field = X_test_sit_phys[SAMPLE_IDX, 0]
y_field = Y_test_phys[SAMPLE_IDX, 0]
panels = [("Low Resolution (X)", x_field, ctx_lon, ctx_lat), ("High Resolution (Y)", y_field, tgt_lon, tgt_lat)]

proj, boundary_path, central_lon = rdeb.make_polar_proj(bbox)
fig, axs = plt.subplots(1, 2, figsize=(10, 6), subplot_kw={"projection": proj})
axes = []
for ax, (title, field, lon_, lat_) in zip(axs, panels):
    axes.append(ax)
    im = ax.pcolormesh(lon_, lat_, field, transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=3, shading="auto")
    rdeb.style_polar_ax(ax, proj, boundary_path, bbox, {}, lon_, lat_, points={})
    ax.set_title(title, fontsize=14)

fig.colorbar(im, ax=axes, shrink=0.7, aspect=15, pad=0.02, label="SIT (m)")
plt.savefig(out("fig03_domain_resolution.png"), dpi=300, bbox_inches='tight')
plt.show()

show('fig03_domain_resolution.png')

# %% [markdown]
# ## Fig. 5: example fields with difference rows

# %% [markdown]
# Revision Fig. 5 (example fields), from figures/build_fig3_example_fields_k100.py (same run, same three samples:
# rng seed 0 on days with > 25 % of cells above 0.5 m; member 4; deterministic = det-MSE + EMA).
#
# Outputs in figures/v2/:
#   fig05_example_fields.png                      PAPER Fig. 5 (chosen 2026-11-13): cmocean ice_r, vmax 2 m, each sample's field row
#                                                 followed by its method-minus-truth row; rows labelled with date + MESACLIP member
#   fig05_example_fields_alt_{blues,ice_r,viridis}_nodiff.png   fields only, in each candidate colormap
#   fig05_example_fields_cmap_compare.png         sample 1 in Blues / ice_r / viridis
#   fig05_example_diffs_alt_separate.png          differences as a separate 3 x 4 figure
# Diff colour limit: 95th percentile of |method - truth| over ocean cells of every diff panel shown, rounded up to 0.05 m.

# %%
import matplotlib
_rc_saved = matplotlib.rcParams.copy()

import glob
import json
import os
import pickle
import sys

import matplotlib
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import cmocean  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "evaluation"))
import run_daily_eval_batch as rdeb  # noqa: E402

RUN, DET_RUN, WINDOW = FS["MESA"]["eng"], FS["MESA"]["det"], FS["map_window"]  # FINAL_SET (final1 = MESA_final_ema, 2015-2020)
VMIN, VMAX = 0, 2


def run_dir(batch):
    m = [d for d in glob.glob(os.path.join(BASE, "results", batch, f"*{WINDOW}*")) if os.path.isdir(d)]
    assert len(m) == 1, m
    return m[0]


edir = os.path.join(run_dir(RUN), "eval_data")
meta = json.load(open(os.path.join(edir, "meta.json")))
bbox = meta["bbox"]
candidate_points = {**meta["candidate_points"], "Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
tile_geometry = pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb"))
z = np.load(os.path.join(edir, "fields.npz"))
Y_test = z["Y_test_phys"]
ice_frac = (Y_test[:, 0] > 0.5).mean(axis=(1, 2))
SAMPLE_IDXS = np.random.default_rng(0).choice(np.where(ice_frac > 0.25)[0], 3, replace=False)  # identical to the original
MEMBER_IDX = 4
F = {"X": z["X_test_sit_phys"][SAMPLE_IDXS], "bil": z["Y_base_phys"][SAMPLE_IDXS], "mem": z["preds_all_phys"][SAMPLE_IDXS, MEMBER_IDX],
     "ens": z["Y_pred_phys"][SAMPLE_IDXS], "truth": Y_test[SAMPLE_IDXS], "ocean": z["mask_test"][SAMPLE_IDXS, 0] <= 0.5,
     "tile": z["test_tile_ids"][SAMPLE_IDXS]}
del z
zd = np.load(os.path.join(run_dir(DET_RUN), "eval_data", "fields.npz"))
assert np.array_equal(zd["Y_test_phys"][SAMPLE_IDXS], F["truth"]), "det-MSE test set differs"
F["det"] = zd["Y_pred_det_phys"][SAMPLE_IDXS]
del zd

st = pd.read_csv(os.path.join(edir, "sample_times.csv"))
n_days = st["time"].nunique()
labels = []
for idx in SAMPLE_IDXS:
    d = pd.Timestamp(st.loc[idx, "time"])
    member = int(idx) // n_days  # samples are stored member-block by member-block (6 x 365 for MESACLIP)
    labels.append((d, member))
    print(f"Fig. 5 sample: index {idx}, date {d:%Y-%m-%d}, MESACLIP member index {member}, ice frac>0.5m {ice_frac[idx]:.2f}")

proj, boundary_path, _ = rdeb.make_polar_proj(bbox)
COLS = [("Low-Res Input", "X", True), ("Bilinear", "bil", False), ("Deterministic", "det", False),
        ("One Member", "mem", False), ("Ensemble Mean", "ens", False), ("High-Res Truth", "truth", False)]
DIFF_KEYS = ["bil", "det", "mem", "ens"]

diffs = np.concatenate([np.abs(F[k][:, 0] - F["truth"][:, 0])[F["ocean"]] for k in DIFF_KEYS])
DLIM = float(np.ceil(np.percentile(diffs, 95) / 0.05) * 0.05)
print(f"diff colour limit: 95th pct |diff| = {np.percentile(diffs, 95):.3f} m -> +/-{DLIM:.2f} m (max |diff| {diffs.max():.2f} m)")


def geo(r):
    g = tile_geometry[int(F["tile"][r])]
    return g["context_lon"], g["context_lat"], g["target_lon"], g["target_lat"]


def draw(ax, field, lon_, lat_, cmap, vmin, vmax, land_on_top):
    im = ax.pcolormesh(lon_, lat_, field, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    rdeb.style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_, lat_)
    if land_on_top:
        # above the field (pcolormesh zorder 1) but below the community stars/labels (zorder 2)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1.5)
        ax.coastlines(resolution="50m", zorder=1.6)
    return im


def row_label(ax, r):
    d, m = labels[r]
    ax.text(-0.08, 0.5, f"{d:%-d %b %Y}\nmember {m}", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=13)


def fields_figure(cmap, name, land_on_top):
    fig, axs = plt.subplots(3, 6, figsize=(18, 9.6), constrained_layout=True, subplot_kw={"projection": proj})
    for r in range(3):
        clon, clat, tlon, tlat = geo(r)
        for c, (title, k, coarse) in enumerate(COLS):
            im = draw(axs[r, c], F[k][r, 0], clon if coarse else tlon, clat if coarse else tlat, cmap, VMIN, VMAX, land_on_top)
            if r == 0:
                axs[r, c].set_title(title, fontsize=18)
        row_label(axs[r, 0], r)
    cb = fig.colorbar(im, ax=axs, aspect=30, shrink=0.9, pad=0.02, extend="max")
    cb.set_label("Sea ice thickness (m)", fontsize=18)
    cb.ax.tick_params(labelsize=14)
    fig.savefig(out(name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out(name))


fields_figure("Blues", "fig05_example_fields_alt_blues_nodiff.png", land_on_top=False)
fields_figure(cmocean.cm.ice_r, "fig05_example_fields_alt_ice_r_nodiff.png", land_on_top=True)
fields_figure("viridis", "fig05_example_fields_alt_viridis_nodiff.png", land_on_top=True)

# colormap side-by-side: sample 1, one row per colormap
fig, axs = plt.subplots(3, 6, figsize=(18, 9.6), constrained_layout=True, subplot_kw={"projection": proj})
clon, clat, tlon, tlat = geo(0)
for r, (cname, cmap, top) in enumerate((("Blues", "Blues", False), ("cmocean ice_r", cmocean.cm.ice_r, True), ("viridis", "viridis", True))):
    for c, (title, k, coarse) in enumerate(COLS):
        im = draw(axs[r, c], F[k][0, 0], clon if coarse else tlon, clat if coarse else tlat, cmap, VMIN, VMAX, top)
        if r == 0:
            axs[r, c].set_title(title, fontsize=18)
    axs[r, 0].text(-0.08, 0.5, cname, transform=axs[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=14)
    cb = fig.colorbar(im, ax=axs[r, :], aspect=12, shrink=0.9, pad=0.01, extend="max")
    cb.set_label("SIT (m)", fontsize=12)
fig.suptitle(f"Colormap comparison, {labels[0][0]:%-d %b %Y} (member {labels[0][1]}), 0-{VMAX} m", fontsize=16)
fig.savefig(out("fig05_example_fields_cmap_compare.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", out("fig05_example_fields_cmap_compare.png"))

# layout A: separate difference figure
DIFF_TITLES = {"bil": "Bilinear − Truth", "det": "Deterministic − Truth", "mem": "One Member − Truth", "ens": "Ensemble Mean − Truth"}
fig, axs = plt.subplots(3, 4, figsize=(12.5, 9.6), constrained_layout=True, subplot_kw={"projection": proj})
for r in range(3):
    _, _, tlon, tlat = geo(r)
    for c, k in enumerate(DIFF_KEYS):
        im = draw(axs[r, c], np.where(F["ocean"][r], F[k][r, 0] - F["truth"][r, 0], np.nan), tlon, tlat, "RdBu_r", -DLIM, DLIM, False)
        if r == 0:
            axs[r, c].set_title(DIFF_TITLES[k], fontsize=16)
    row_label(axs[r, 0], r)
cb = fig.colorbar(im, ax=axs, aspect=30, shrink=0.9, pad=0.02, extend="both")
cb.set_label("Difference from high-res truth (m)", fontsize=16)
cb.ax.tick_params(labelsize=13)
fig.savefig(out("fig05_example_diffs_alt_separate.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", out("fig05_example_diffs_alt_separate.png"))

# layout B: field row + diff row per sample (diffs sit under their method; input/truth columns blank in diff rows)
fig = plt.figure(figsize=(18, 19.8), constrained_layout=True)
gs = fig.add_gridspec(7, 6, height_ratios=[1] * 6 + [0.07])  # last row: the two horizontal colourbars
field_axes, diff_axes = [], []
for r in range(3):
    clon, clat, tlon, tlat = geo(r)
    for c, (title, k, coarse) in enumerate(COLS):
        ax = fig.add_subplot(gs[2 * r, c], projection=proj)
        im_f = draw(ax, F[k][r, 0], clon if coarse else tlon, clat if coarse else tlat, cmocean.cm.ice_r, VMIN, VMAX, True)
        field_axes.append(ax)
        if r == 0:
            ax.set_title(title, fontsize=18)
        if c == 0:
            row_label(ax, r)
        if k in DIFF_KEYS:
            axd = fig.add_subplot(gs[2 * r + 1, c], projection=proj)
            im_d = draw(axd, np.where(F["ocean"][r], F[k][r, 0] - F["truth"][r, 0], np.nan), tlon, tlat, "RdBu_r", -DLIM, DLIM, False)
            axd.set_title("− Truth", fontsize=12)
            diff_axes.append(axd)
cb = fig.colorbar(im_f, cax=fig.add_subplot(gs[6, 0:3]), orientation="horizontal", extend="max")
cb.set_label("Sea ice thickness (m)", fontsize=16)
cb = fig.colorbar(im_d, cax=fig.add_subplot(gs[6, 3:6]), orientation="horizontal", extend="both")
cb.set_label("Method − truth (m)", fontsize=16)
fig.savefig(out("fig05_example_fields.png"), dpi=250, bbox_inches="tight")
plt.close(fig)
print("wrote", out("fig05_example_fields.png"))


matplotlib.rcParams.update(_rc_saved)  # undo this cell's style changes
show('fig05_example_fields.png', 'fig05_example_fields_cmap_compare.png', 'fig05_example_diffs_alt_separate.png')

# %% [markdown]
# ## Fig. 6: rank histogram
#
# Output: `fig06_rank_histogram.png`

# %%
rank_hist_df = pd.read_csv(os.path.join(AGG_DIR, "14_rank_histogram_data_4window.csv"), index_col=0)
rank_regime_df = pd.read_csv(os.path.join(AGG_DIR, "14b_rank_histogram_by_regime_data_4window.csv"), index_col=0)
K = len(rank_hist_df) - 1
expected_freq = rank_hist_df["expected_frequency"].iloc[0]

sample = ncp.pal_sample("Sodium", 8)
regime_colors = [sample[0], sample[3], sample[5], sample[6]]
REGIME_SHORT = {"Open water/frazil (0-0.15 m)": "Open Water/Frazil", "Thin ice (0.15-0.5 m)": "Thin Ice",
                "Moderate ice (0.5-1.0 m)": "Moderate Ice", "Thick ice (>1.0 m)": "Thick Ice"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.bar(rank_hist_df["rank"], rank_hist_df["frequency"], width=0.85, color=sample[3], alpha=0.85, label="Observed Frequency")
ax1.axhline(expected_freq, color="black", linestyle="--", linewidth=2, label="Ideal Uniform Frequency")
ax1.set_xlabel(f"Rank of Truth Among {K} Ensemble Members", fontsize=14)
ax1.set_ylabel("Frequency", fontsize=14)
ax1.set_title("(A) Rank Histogram, All Ocean Cells", fontsize=16)
ax1.tick_params(axis='both', labelsize=14)
ax1.legend(fontsize=11, frameon=False)

total = rank_regime_df["count"].sum()
for (label, grp), color in zip(rank_regime_df.groupby("SIT regime", sort=False), regime_colors):
    share = grp["count"].sum() / total
    ax2.plot(grp["rank"], grp["relative_frequency"], marker="o", markersize=6, linewidth=2, color=color,
             label=f"{REGIME_SHORT.get(label, label)} ({share:.0%} of cells)")
ax2.axhline(1.0, color="black", linestyle="--", linewidth=2, label="Ideal Uniform Frequency")
ax2.set_xlabel(f"Rank of Truth Among {K} Ensemble Members", fontsize=14)
ax2.set_ylabel("Frequency / Ideal Frequency", fontsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_title("(B) Rank Histogram by SIT Regime", fontsize=16)
ax2.legend(fontsize=11, frameon=False, loc='upper center')  # 2026-10-24: was 'lower center', which overlapped the curves
plt.tight_layout()
plt.savefig(out("fig06_rank_histogram.png"), dpi=300, bbox_inches="tight")

plt.show()

show('fig06_rank_histogram.png')

# %% [markdown]
# ## Fig. 7: spread vs error by SIT regime
#
# Output: `fig07_spread_skill_regime.png`

# %%
spread_skill_df = pd.read_csv(os.path.join(AGG_DIR, "15_spread_skill_by_regime_data_4window.csv"), index_col=0)
spread_skill_df = spread_skill_df[spread_skill_df["SIT regime"] != "All ocean cells"]
SIT_BIN_LABELS = spread_skill_df["SIT regime"].tolist()

sample = ncp.pal_sample("Sodium", 8)
x = np.arange(len(SIT_BIN_LABELS))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
width = 0.35
ax1.bar(x - width / 2, spread_skill_df["RMS spread, (K+1)/K-corrected (m)"], width, color=sample[3], label=r"$\sqrt{\frac{K+1}{K}\,\overline{s^2}}$")
ax1.bar(x + width / 2, spread_skill_df["RMSE (m)"], width, color=sample[0], label="Ensemble-Mean RMSE")
ax1.set_xticks(x)
ax1.set_xticklabels(SIT_BIN_LABELS, rotation=45, ha="right")
ax1.tick_params(axis='both', labelsize=12)
ax1.set_ylabel("SIT (m)", fontsize=14)
ax1.set_title("(A) Spread vs. Error by SIT Regime", fontsize=16)
ax1.legend(fontsize=14, frameon=False, loc='upper left')

ax2.bar(x, spread_skill_df["Spread-skill ratio (1.0 = calibrated)"], color=sample[5], alpha=0.85)
ax2.axhline(1.0, color="black", linestyle="--", linewidth=2, label="Perfect Calibration")
ax2.set_xticks(x)
ax2.set_xticklabels(SIT_BIN_LABELS, rotation=45, ha="right")
ax2.tick_params(axis='both', labelsize=12)
ax2.set_ylabel("Spread-Skill Ratio\n(<1 under-dispersed, >1 over-dispersed)", fontsize=14)
ax2.set_title("(B) Spread-Skill Ratio by SIT Regime", fontsize=16)
ax2.legend(fontsize=14, frameon=False, loc='lower left')
plt.tight_layout()
plt.savefig(out("fig07_spread_skill_regime.png"), dpi=300, bbox_inches="tight")

plt.show()

show('fig07_spread_skill_regime.png')

# %% [markdown]
# ## Fig. 9: PIOMAS application (and supplement versions)

# %% [markdown]
# Revision Fig. 9 (PIOMAS application), from figures/build_fig7_final_piomas_application.py (same run: PIOMAS 2021 input,
# EngressNet + EMA, window 2015-2020, results/PIOMAS_2010-2023_cdrv6_emaFOSI_4win).
#
# Outputs in figures/v2/:
#   fig09_piomas_application_supp_original_tracks.png  (supplement) original top row (ICESat-2 tracks on the Blues scale); bottom row: the +-1 std band is
#                                    replaced by the 16-84th percentile band of the 20 members (respects SIT >= 0)
#   fig09_piomas_application.png     PAPER Fig. 9 (chosen 2026-11-13): top row: ICESat-2 pass-cell mean minus EngressNet ensemble mean at the same cell and day
#                                    (diverging), over a light grey ensemble-mean background; same bottom row
#   fig09b_supp_is2_scatter.png      (supplement) ICESat-2 pass-cell mean vs. bilinear PIOMAS / deterministic U-Net / EngressNet ensemble
#                                    mean, all QC'd pass-cells Oct 2018-May 2022, 4 training windows (EngressNet 4 x 20
#                                    members; deterministic = mean of the 4 det-MSE + EMA windows), 1:1 line, bias/RMSE/r
# Pass-cell model values come from the matched files observations' evaluation already uses (ema_piomas_cells.npz,
# detmse_ema_piomas_cells.npz; written 2026-10-24, after the coarse-zero rule); same QC (extent >= 2 km, >= 20 segments,
# median segment uncertainty <= 1 m). These reproduce scoreboard.csv's EngressNet (final) row exactly.

# %%
import matplotlib
_rc_saved = matplotlib.rcParams.copy()

import glob
import os
import pickle
import sys

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import nicopal as ncp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "training"))
sys.path.insert(0, os.path.join(BASE, "evaluation"))
import run_daily_eval_batch as rdeb  # noqa: E402
import functions_engressnet as fe  # noqa: E402

IS2C = "/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells"
YEAR, WINDOW = 2021, FS["piomas_window"]  # FINAL_SET
P_LO, P_HI = 16, 84

# ---------------- model fields for the maps / point series (unchanged source) ----------------
m = [d for d in sorted(glob.glob(os.path.join(BASE, "results", FS["piomas2010"], f"*_{WINDOW}_{YEAR}_*")))
     if os.path.exists(os.path.join(d, "eval_data", "fields.npz"))]
assert len(m) == 1, m
EDIR = os.path.join(m[0], "eval_data")
z = np.load(os.path.join(EDIR, "fields.npz"))
Y_pred = z["Y_pred_phys"][:, 0]
members = z["preds_all_phys"][:, :, 0]
ocean_hw = z["mask_test"][0, 0] <= 0.5
del z
geo = pickle.load(open(os.path.join(EDIR, "tile_geometry.pkl"), "rb"))[0]
tlat, tlon = np.asarray(geo["target_lat"]), np.asarray(geo["target_lon"])
times = pd.to_datetime(pd.read_csv(os.path.join(EDIR, "sample_times.csv"))["time"])
months = times.dt.month.values
points = pd.read_csv(os.path.join(EDIR, "candidate_point_timeseries.csv"))

# ---------------- ICESat-2 pass-cells + matched model values ----------------
pcd = pd.read_parquet(f"{IS2C}/pass_cells.parquet").query("in_domain").reset_index(drop=True)
qc = ((pcd.extent_km >= 2.0) & (pcd.n_seg >= 20) & (pcd.unc_med <= 1.0)).values
E = np.load(f"{IS2C}/eval/{FS['cells_eng']}_cells.npz")
D = np.load(f"{IS2C}/eval/{FS['cells_det']}_cells.npz")
assert list(E["windows"]) == list(D["windows"]) and len(E["members"]) == len(pcd)
Mm, Dm, Bl = E["members"].astype(float), D["members"].astype(float), E["bilinear"].astype(float)
w_idx = [i for i, w in enumerate(E["windows"]) if str(w).startswith(WINDOW)][0]
ens_win = E["ensmean"][:, w_idx].astype(float)  # the map's window, for the difference panels
y = pcd.h_mean.values
dates = pd.DatetimeIndex(pcd.date)

MONTHS = [1, 4, 7, 10]
MONTH_NAMES = {1: "January", 4: "April", 7: "July", 10: "October"}
plot_bbox = {"lon_min": -182, "lon_max": -151, "lat_min": 60, "lat_max": 75}
proj, boundary, _ = fe.make_polar_proj(plot_bbox)
lon2d, lat2d = np.meshgrid(tlon, tlat)
sample = ncp.pal_sample("Sodium", 8)
in_year = qc & (dates.year == YEAR) & np.isfinite(ens_win)
diff = y - ens_win
DLIM = float(np.ceil(np.nanpercentile(np.abs(diff[in_year & np.isin(dates.month, MONTHS)]), 95) / 0.25) * 0.25)


def point_hope(ax):
    lon = ((rdeb.POINT_HOPE_LON_360 + 180) % 360) - 180
    ax.plot(lon, rdeb.POINT_HOPE_LAT, marker="*", color="red", markersize=10, transform=ccrs.PlateCarree())
    ax.text(lon + 1, rdeb.POINT_HOPE_LAT + 0.35, "Point Hope", color="red", fontsize=7, transform=ccrs.PlateCarree())


def bottom_row(fig):
    band_rows = []
    for j, name in enumerate(["Kivalina", "Point Hope"]):
        ax = fig.add_subplot(2, 2, 3 + j)
        row = points[(points.point == name) & (points.method == "stochastic_unet_mean")]
        iy, ix = int(row["grid_iy"].iloc[0]), int(row["grid_ix"].iloc[0])
        mean = Y_pred[:, iy, ix]
        lo, hi = np.percentile(members[:, :, iy, ix], [P_LO, P_HI], axis=1)
        sd = members[:, :, iy, ix].std(axis=1)
        band_rows.append((name, (mean - sd < 0).sum(), (lo < 0).sum()))
        ax.fill_between(times.values, lo, hi, alpha=0.3, color=sample[1], label=f"{P_LO}–{P_HI}th Percentile of Members")
        ax.plot(times.values, mean, color=sample[0], linewidth=2, label="Ensemble Mean")
        ax.set_title(f"{name} Daily Downscaled PIOMAS SIT ({YEAR})", fontsize=16)
        ax.set_ylabel("SIT (m)", fontsize=14)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=14)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="both", labelsize=14)
        ax.tick_params(axis="x", rotation=30)
    return band_rows


# ---------------- (1) original top row + percentile band ----------------
fig = plt.figure(figsize=(16, 12))
axes = []
for i, mo in enumerate(MONTHS):
    ax = fig.add_subplot(2, 4, i + 1, projection=proj)
    axes.append(ax)
    fe.style_polar_ax(ax, proj, boundary, plot_bbox)
    point_hope(ax)
    mm = np.where(ocean_hw, Y_pred[months == mo].mean(axis=0), np.nan)
    pc = ax.pcolormesh(lon2d, lat2d, mm, transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=2.5, shading="auto")
    s = in_year & (dates.month == mo)
    if s.any():
        ax.scatter(((pcd.lon.values[s] + 180) % 360) - 180, pcd.lat.values[s], c=y[s], cmap="Blues", vmin=0, vmax=2.5,
                   s=2.5, linewidths=0.15, edgecolors="#3a3a3a", transform=ccrs.PlateCarree(), zorder=5, rasterized=True)
    ax.set_title(f"{MONTH_NAMES[mo]} {YEAR}", fontsize=14)
cb = fig.colorbar(pc, ax=axes, orientation="horizontal", aspect=30, shrink=0.9, pad=0.02)
cb.set_label("Ensemble Mean SIT (m)", fontsize=16)
cb.ax.tick_params(labelsize=14)
band_rows = bottom_row(fig)
fig.savefig(out("fig09_piomas_application_supp_original_tracks.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", out("fig09_piomas_application_supp_original_tracks.png"))
for name, n_sd, n_pct in band_rows:
    print(f"  {name}: days with mean-1std < 0: {n_sd}; days with {P_LO}th pct < 0: {n_pct}")

# ---------------- (2) ICESat-2 minus EngressNet on a light background ----------------
fig = plt.figure(figsize=(16, 12.6))
axes = []
for i, mo in enumerate(MONTHS):
    ax = fig.add_subplot(2, 4, i + 1, projection=proj)
    axes.append(ax)
    fe.style_polar_ax(ax, proj, boundary, plot_bbox)
    point_hope(ax)
    mm = np.where(ocean_hw, Y_pred[months == mo].mean(axis=0), np.nan)
    bg = ax.pcolormesh(lon2d, lat2d, mm, transform=ccrs.PlateCarree(), cmap="Greys", vmin=0, vmax=5, shading="auto", alpha=0.6)
    s = in_year & (dates.month == mo)
    if s.any():
        sc = ax.scatter(((pcd.lon.values[s] + 180) % 360) - 180, pcd.lat.values[s], c=diff[s], cmap="RdBu_r", vmin=-DLIM,
                        vmax=DLIM, s=3, linewidths=0, transform=ccrs.PlateCarree(), zorder=5, rasterized=True)
        ax.set_title(f"{MONTH_NAMES[mo]} {YEAR}\nmean {np.mean(diff[s]):+.2f} m, N = {s.sum():,}", fontsize=13)
    else:
        ax.set_title(f"{MONTH_NAMES[mo]} {YEAR}\n(no ICESat-2 data)", fontsize=13)
cb = fig.colorbar(sc, ax=axes, orientation="horizontal", aspect=30, shrink=0.9, pad=0.02, extend="both")
cb.set_label("ICESat-2 pass-cell mean − EngressNet ensemble mean (m)", fontsize=15)
cb.ax.tick_params(labelsize=13)
cb2 = fig.colorbar(bg, ax=axes, orientation="horizontal", aspect=60, shrink=0.45, pad=0.01)
cb2.set_label("Background: EngressNet ensemble-mean SIT (m)", fontsize=11)
cb2.ax.tick_params(labelsize=10)
bottom_row(fig)
fig.savefig(out("fig09_piomas_application.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", out("fig09_piomas_application.png"), f"| diff limit +/-{DLIM} m (95th pct |diff|, shown months)")

# ---------------- (3) hexbin: ICESat-2 vs downscaled SIT ----------------
ok = qc & np.isfinite(Mm).all(1) & np.isfinite(Dm).all(1) & np.isfinite(Bl)
est = {"Bilinear": Bl, "Deterministic U-Net": Dm.mean(1), "EngressNet": Mm.mean(1)}
NU = len(E["windows"])  # training units (4 windows for final1, 1 model for final2)
LABEL = {"Bilinear": "Bilinear PIOMAS",
         "Deterministic U-Net": "Deterministic U-Net" + (f" ({NU}-window mean)" if NU > 1 else ""),
         "EngressNet": f"EngressNet ensemble mean ({NU} × {int(E['K'])})" if NU > 1 else f"EngressNet ensemble mean ({int(E['K'])} members)"}


def crps_fair(X, yy):
    K = X.shape[1]
    Xs = np.sort(X, 1)
    w = 2 * np.arange(1, K + 1) - K - 1
    return (np.abs(X - yy[:, None]).mean(1) - (Xs * w).sum(1) / (K * (K - 1))).mean()


yy = y[ok]
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True, constrained_layout=True)
norm = LogNorm(vmin=1, vmax=2000)
stats = {}
for ax, (k, x) in zip(axs, est.items()):
    xx = x[ok]
    r = xx - yy
    stats[k] = dict(bias=r.mean(), rmse=np.sqrt((r ** 2).mean()), mae=np.abs(r).mean(), r=np.corrcoef(xx, yy)[0, 1])
    hb = ax.hexbin(xx, yy, gridsize=50, extent=(0, 5, 0, 5), mincnt=1, norm=norm, cmap="Greys")
    ax.plot([0, 5], [0, 5], color=METHOD_COLORS[k], lw=1.4, label="1:1")
    ax.set_title(LABEL[k], fontsize=12, color=INK)
    ax.text(0.03, 0.97, f"bias {stats[k]['bias']:+.3f} m\nRMSE {stats[k]['rmse']:.3f} m\nr {stats[k]['r']:.3f}",
            transform=ax.transAxes, va="top", fontsize=11, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlabel("Downscaled SIT (m)", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
axs[0].set_ylabel("ICESat-2 pass-cell mean (m)", fontsize=12)
fig.colorbar(hb, ax=axs, shrink=0.85, pad=0.01).set_label("Pass-cells per bin", fontsize=12)
fig.suptitle(f"All QC'd ICESat-2 0.1° pass-cells, Oct 2018 – May 2022 (N = {ok.sum():,}), PIOMAS input", fontsize=13, y=1.06)
fig.savefig(out("fig09b_supp_is2_scatter.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", out("fig09b_supp_is2_scatter.png"))

crps_eng = crps_fair(Mm[ok], yy)
KD = int(D["K"])
crps_det4 = crps_fair(Dm[ok][:, 0::KD], yy) if NU > 1 else float("nan")  # distinct windows as an ensemble (final1 only)
print(f"N = {ok.sum()}")
for k, s_ in stats.items():
    print(f"  {k:22s} bias {s_['bias']:+.4f}  RMSE {s_['rmse']:.4f}  MAE {s_['mae']:.4f}  r {s_['r']:.4f}")
print(f"  EngressNet fair CRPS ({Mm.shape[1]} members) {crps_eng:.4f}; det-MSE as a {NU}-window ensemble, fair CRPS {crps_det4:.4f}; "
      f"det MAE {stats['Deterministic U-Net']['mae']:.4f}")


matplotlib.rcParams.update(_rc_saved)  # undo this cell's style changes
show('fig09_piomas_application.png', 'fig09_piomas_application_supp_original_tracks.png', 'fig09b_supp_is2_scatter.png')

# %% [markdown]
# ## Fig. A1: MESACLIP sensitivity heatmap ("cascade" shown as "repeated per-stage noise")
#
# Output: `figA1_mesa_sensitivity.png`

# %%
mesa_df = pd.read_csv(os.path.join(BASE, "sensitivity", "analysis", "sensitivity_data_mesa.csv"), index_col=0)
CORRECTED_SSR_MESA = {
    "avg, pre-fix (old refiner)": 0.707,
    "interp, pre-fix (old refiner)": 0.655,
    "avg, shared-bias fix (old refiner)": 0.628,
    "avg, deep mask head (old refiner)": 0.652,
    "avg, noise-mix none (old refiner)": 0.683,
    "avg, noise-mix gaussian_fixed (old refiner)": 0.690,
    "no correlated noise": 0.803,
    'single global noise field ("global z")': 0.768,
    "cascade (used in paper), 4-window avg": 0.715,
    "cascade, 2000-2005 window, seed 0": 0.713,
    "cascade, 2000-2005 window, seed 1": 0.701,
    "cascade, 2000-2005 window, seed 2": 0.737,
    "cascade, 2000-2005 window, seed 3": 0.689,
    "cascade, 2000-2005 window, seed 4": 0.704,
    "cascade, late mask fusion": 0.726,
    "cascade, noise channels 1→2": 0.719,
    "cascade, kernel size 5→3": 0.693,
    "cascade, kernel size 5→9": 0.707,
}
missing = set(mesa_df.index) - set(CORRECTED_SSR_MESA)
assert not missing, missing
mesa_df["Spread/Error"] = mesa_df.index.map(CORRECTED_SSR_MESA)
mesa_df = mesa_df.rename(columns={"Spread/Error": "Spread-skill ratio"})
mesa_df = mesa_df.drop(mesa_df.index[[0, 1, 2, 3, 4, 5, 7]])

METRIC_DIRECTION = {
    "MAE": "lower", "RMSE": "lower", "Bias": "zero", "Grad MAE": "lower",
    "Pattern Corr": "higher", "SSIM": "higher", "IIEE": "lower",
    "Coastal MAE": "lower", "Coastal RMSE": "lower", "Spread-skill ratio": "one",
}


def badness(series, direction):
    if direction == "lower":
        return series
    if direction == "higher":
        return -series
    if direction == "zero":
        return series.abs()
    if direction == "one":
        return (series - 1).abs()
    raise ValueError(direction)


def normalize_0_1(series):
    lo, hi = series.min(), series.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return series * 0.0
    return (series - lo) / (hi - lo)


def plot_badness_heatmap(value_df, title, figsize=None, cmap="Reds", fmt="{:.3f}"):
    badness_df = pd.DataFrame(index=value_df.index, columns=value_df.columns, dtype=float)
    for col in value_df.columns:
        direction = METRIC_DIRECTION.get(col, "lower")
        badness_df[col] = normalize_0_1(badness(value_df[col], direction))
    if figsize is None:
        figsize = (len(value_df.columns), 0.5 * len(value_df.index))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(badness_df.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(value_df.columns)))
    ax.set_xticklabels(value_df.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(value_df.index)))
    ax.set_yticklabels(value_df.index)
    for i in range(value_df.shape[0]):
        for j in range(value_df.shape[1]):
            v = value_df.values[i, j]
            if pd.isna(v):
                continue
            b = badness_df.values[i, j]
            text_color = "white" if b > 0.6 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=text_color)
    ax.set_title(title, fontsize=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(value_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(value_df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    return fig, ax


mesa_df = mesa_df.rename(index=lambda s: s.replace("cascade", "repeated per-stage noise"))
fig1, ax1 = plot_badness_heatmap(
    mesa_df,
    "Sensitivity Tests for MESACLIP (blank = not measured)",
)

plt.savefig(out("figA1_mesa_sensitivity.png"), dpi=300, bbox_inches='tight')
plt.show()

show("figA1_mesa_sensitivity.png")

# %% [markdown]
# ## Fig. A2: FOSI sensitivity heatmap
#
# Output: `figA2_fosi_sensitivity.png`

# %%
fosi_df = pd.read_csv(os.path.join(BASE, "sensitivity", "analysis", "sensitivity_data_fosi.csv"), index_col=0)
fosi_df = fosi_df.drop(index="cascade, MESA-warm-started")
CORRECTED_SSR_FOSI = {
    "avg, smooth-noise-fixed refiner (historical)": 0.613,
    "interp, smooth-noise-fixed refiner (historical)": 0.547,
    "trained full-year (historical refiner)": 0.641,
    "trained season-only (historical refiner)": 0.763,
    "cascade (used in paper), 4-window avg": 0.642,
}
missing = set(fosi_df.index) - set(CORRECTED_SSR_FOSI)
assert not missing, missing
fosi_df["Spread/Error"] = fosi_df.index.map(CORRECTED_SSR_FOSI)
fosi_df = fosi_df.rename(columns={"Spread/Error": "Spread-skill ratio"})

METRIC_DIRECTION = {
    "MAE": "lower", "RMSE": "lower", "Bias": "zero", "Grad MAE": "lower",
    "Pattern Corr": "higher", "SSIM": "higher", "IIEE": "lower",
    "Coastal MAE": "lower", "Coastal RMSE": "lower", "Spread-skill ratio": "one",
}


def badness(series, direction):
    if direction == "lower":
        return series
    if direction == "higher":
        return -series
    if direction == "zero":
        return series.abs()
    if direction == "one":
        return (series - 1).abs()
    raise ValueError(direction)


def normalize_0_1(series):
    lo, hi = series.min(), series.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return series * 0.0
    return (series - lo) / (hi - lo)


def plot_badness_heatmap(value_df, title, figsize=None, cmap="Reds", fmt="{:.3f}"):
    badness_df = pd.DataFrame(index=value_df.index, columns=value_df.columns, dtype=float)
    for col in value_df.columns:
        direction = METRIC_DIRECTION.get(col, "lower")
        badness_df[col] = normalize_0_1(badness(value_df[col], direction))
    if figsize is None:
        figsize = (len(value_df.columns), 0.5 * len(value_df.index))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(badness_df.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(value_df.columns))); ax.set_xticklabels(value_df.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(value_df.index))); ax.set_yticklabels(value_df.index)
    for i in range(value_df.shape[0]):
        for j in range(value_df.shape[1]):
            v = value_df.values[i, j]
            if pd.isna(v):
                continue
            b = badness_df.values[i, j]
            text_color = "white" if b > 0.6 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=text_color)
    ax.set_title(title, fontsize=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(value_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(value_df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    return fig, ax


fosi_df = fosi_df.rename(index=lambda s: s.replace("cascade", "repeated per-stage noise"))
fig2, ax2 = plot_badness_heatmap(
    fosi_df,
    "Sensitivity tests for FOSI (blank = not measured)",
)

plt.savefig(out("figA2_fosi_sensitivity.png"), dpi=300, bbox_inches='tight')

plt.show()

show("figA2_fosi_sensitivity.png")

# %% [markdown]
# # Tables as figures

# %% [markdown]
# ## Tables 2-5 (summary figure and single panels) and Table 2b (probabilistic baseline)
#
# Revisions 2026-11, section B: the paper's Tables 2-5 as compact charts (tables kept). One method palette (METHOD_COLORS, set in Setup).
# Values are read from the saved results (final configuration: 4-window mean, seed 0, K = 20, coarse-zero rule, ocean cells):
#   Table 2  evaluation/corrected_calibration/{MESA,FOSI}_{final,detmse}_ema{,_ocean_metrics}.csv  (RMSE, r, CRPS; det/bilinear CRPS = MAE)
#   Table 3  same files, MESACLIP, EngressNet relative to det-MSE (% change; IIEE, CRPS/MAE, SSIM, r, Grad MAE)
#   Table 4  results/revisions_2026-11/table4_community.csv (C.1) + domain-mean ensemble-mean bias from the ocean-metric files
#   Table 5  observations/data/alongtrack_eval/scoreboard.csv (CRPS; det and bilinear: MAE) and
#            cice_ceiling_bar.csv (pooled within-1 deg std relative to ICESat-2; CICE truth markers)
# Outputs figures/v2/tab02-05_summary.png (all panels) and tab02_headline, tab03_relative, tab04_community, tab04a_bias,
# tab05_icesat2 (each alone). Table 5 deterministic = single model, CRPS = its MAE (decision 2026-11-13, as in Table 2).

# %%
import matplotlib
_rc_saved = matplotlib.rcParams.copy()

import os
import sys

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"

C = f"{BASE}/evaluation/corrected_calibration"
A = f"{BASE}/observations/data/alongtrack_eval"
METHODS = ["EngressNet", "Deterministic U-Net", "Bilinear"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False})


def perfect(ds):
    pre = {"MESACLIP": "MESA", "FOSI": "FOSI"}[ds]
    eng, det = FS[pre]["eng"], FS[pre]["det"]  # FINAL_SET (final_set.py)
    om_e = pd.read_csv(f"{C}/{eng}_ocean_metrics.csv"); om_d = pd.read_csv(f"{C}/{det}_ocean_metrics.csv")
    cal = pd.read_csv(f"{C}/{eng}.csv"); cal = cal[cal.run != "POOLED"]
    g = lambda df, m: df[(df.run == "MEAN over windows") & (df.method == m)].iloc[0]
    e, d, b = g(om_e, "EngressNet ensemble mean"), g(om_d, "Deterministic (zero-noise pass)"), g(om_e, "Bilinear")
    rows = {"EngressNet": {**e[["RMSE", "MAE", "Pattern Corr", "SSIM", "Grad MAE", "IIEE", "Bias"]].to_dict(), "CRPS": cal["CRPS (ocean)"].mean()},
            "Deterministic U-Net": {**d[["RMSE", "MAE", "Pattern Corr", "SSIM", "Grad MAE", "IIEE", "Bias"]].to_dict(), "CRPS": d["MAE"]},
            "Bilinear": {**b[["RMSE", "MAE", "Pattern Corr", "SSIM", "Grad MAE", "IIEE", "Bias"]].to_dict(), "CRPS": b["MAE"]}}
    return pd.DataFrame(rows).T


P = {ds: perfect(ds) for ds in ("MESACLIP", "FOSI")}

# 2026-11-14: MC-dropout baseline (E2; window-matched like EngressNet: 4-window mean of seed 0) from the D-metric outputs.
DM = f"{BASE}/results/revisions_2026-11/D_metrics"
METHODS_T2 = list(METHODS)
D_KEYS = {"RMSE": "Ensemble-mean RMSE", "Pattern Corr": "Pattern Corr (ensemble mean / point value)", "CRPS": "CRPS (point mass: MAE)"}
for _ds, _pre in ((("MESACLIP", "MESA"), ("FOSI", "FOSI")) if FINAL_SET == "final1" else ()):  # MC-dropout: final1 only
    _cols = [pd.read_csv(f"{DM}/{_pre}_{w}_extra_E2.csv", index_col=0)["MC-dropout U-Net"] for w in ("2000-2005", "2005-2010", "2010-2015")]
    _cols.append(pd.read_csv(f"{DM}/{_pre}_2015-2020_extra_E.csv", index_col=0)["MC-dropout U-Net s0"])
    _mc = pd.concat(_cols, axis=1).apply(pd.to_numeric, errors="coerce").mean(axis=1)
    for _k, _dk in D_KEYS.items():
        P[_ds].loc["MC-dropout U-Net", _k] = _mc[_dk]
if FINAL_SET == "final1":
    METHODS_T2 = METHODS + ["MC-dropout U-Net"]


def bar_group(ax, labels, series, ylabel, fmt="{:.3f}", annotate=True):
    """series: {method: [values per label]} -> grouped bars, method colours, 2 px surface gaps."""
    n = len(series); x = np.arange(len(labels)); w = 0.8 / n
    for i, (m, v) in enumerate(series.items()):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, color=METHOD_COLORS[m], label=m, edgecolor="white", linewidth=1)
        if annotate:
            for xi, vi in zip(x + (i - (n - 1) / 2) * w, v):
                ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)


def panel_t2(axs):
    for ax, (metric, lab) in zip(axs, (("RMSE", "RMSE (m)"), ("Pattern Corr", "Pattern correlation r"),
                                       ("CRPS", "CRPS (m)\n(deterministic, bilinear: MAE)"))):
        bar_group(ax, list(P), {m: [P[ds].loc[m, metric] for ds in P] for m in METHODS_T2}, lab,
                  fmt="{:.3f}" if metric != "Pattern Corr" else "{:.2f}")
        if metric == "Pattern Corr":
            ax.set_ylim(0.8, 1.0)
        ax.set_title({"RMSE": "Table 2: RMSE", "Pattern Corr": "Table 2: pattern r", "CRPS": "Table 2: CRPS"}[metric], loc="left")
    axs[0].legend(fontsize=8, loc="upper left")


def panel_t3(ax):
    ds = P["MESACLIP"]
    spec = [("IIEE", "IIEE", "lower"), ("CRPS", "CRPS (det: MAE)", "lower"), ("SSIM", "SSIM", "higher"),
            ("Pattern Corr", "Pattern r", "higher"), ("Grad MAE", "Gradient MAE", "lower")]
    y = np.arange(len(spec))[::-1]
    for yi, (k, lab, better) in zip(y, spec):
        pct = 100 * (ds.loc["EngressNet", k] / ds.loc["Deterministic U-Net", k] - 1)
        good = (pct < 0) if better == "lower" else (pct > 0)
        ax.plot([0, pct], [yi, yi], color=METHOD_COLORS["EngressNet"], lw=2)
        ax.plot(pct, yi, "o", ms=9, color=METHOD_COLORS["EngressNet"], mfc=METHOD_COLORS["EngressNet"] if good else "white", mew=2)
        ax.text(pct + (1.2 if pct >= 0 else -1.2), yi + 0.22, f"{pct:+.1f}%", ha="left" if pct >= 0 else "right", fontsize=9)
        ax.text(1.02, yi, "better: ←" if better == "lower" else "better: →", transform=ax.get_yaxis_transform(), fontsize=8,
                color=MUTED, va="center")
    ax.axvline(0, color=INK, lw=1)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.8)
    ax.set_yticks(y); ax.set_yticklabels([s[1] for s in spec]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("EngressNet relative to deterministic U-Net (%)")
    ax.set_title("Table 3 (MESACLIP): filled = EngressNet better", loc="left")


def panel_t4(ax_bias, ax_vals):
    t = pd.read_csv(f"{BASE}/results/revisions_2026-11/table4_community{SFX}.csv")
    comms = ["Point Hope", "Kivalina", "Kotzebue", "Shishmaref", "Nome"]
    x = np.arange(len(comms)); w = 0.2
    styles = [("MESACLIP", "EngressNet", "EngressNet bias yr"), ("MESACLIP", "Deterministic U-Net", "Det bias yr"),
              ("FOSI", "EngressNet", "EngressNet bias yr"), ("FOSI", "Deterministic U-Net", "Det bias yr")]
    for i, (ds, m, col) in enumerate(styles):
        v = [t[(t.Dataset == ds) & (t.Community == c)][col].item() for c in comms]
        ax_bias.bar(x + (i - 1.5) * w, v, w * 0.92, color=METHOD_COLORS[m], edgecolor="white", linewidth=1,
                    hatch="" if ds == "MESACLIP" else "///", label=f"{m}, {ds}")
    for ds, ls in (("MESACLIP", "--"), ("FOSI", ":")):
        ax_bias.axhline(P[ds].loc["EngressNet", "Bias"], color=INK, ls=ls, lw=1.2,
                        label=f"domain-mean EngressNet bias, {ds} ({P[ds].loc['EngressNet', 'Bias']:+.4f} m)")
    ax_bias.axhline(0, color=MUTED, lw=0.8)
    ax_bias.set_xticks(x); ax_bias.set_xticklabels(comms, rotation=20, ha="right"); ax_bias.set_ylabel("Nearest-cell bias, 2021 (m)")
    ax_bias.set_title("Table 4: community bias (hatched = FOSI)", loc="left"); ax_bias.set_ylim(-0.1, 0.04); ax_bias.legend(fontsize=7, ncol=2, loc="lower left")
    ax_bias.grid(axis="x", visible=False)
    # actual values: truth vs predictions, full year
    for i, (ds, mk) in enumerate((("MESACLIP", "o"), ("FOSI", "s"))):
        sub = t[t.Dataset == ds].set_index("Community").loc[comms]
        off = (i - 0.5) * 0.3
        ax_vals.scatter(x + off, sub["Truth mean yr"], marker=mk, s=60, facecolor="white", edgecolor=INK, lw=1.5, zorder=3,
                        label=f"truth, {ds}")
        ax_vals.scatter(x + off, sub["EngressNet mean yr"], marker=mk, s=28, color=METHOD_COLORS["EngressNet"], zorder=4,
                        label=f"EngressNet, {ds}")
        ax_vals.scatter(x + off, sub["Det mean yr"], marker=mk, s=28, color=METHOD_COLORS["Deterministic U-Net"], zorder=4,
                        label=f"deterministic, {ds}")
    ax_vals.set_xticks(x); ax_vals.set_xticklabels(comms, rotation=20, ha="right"); ax_vals.set_ylabel("Mean SIT at nearest cell, 2021 (m)")
    ax_vals.set_title("Table 4: actual values (open = truth)", loc="left"); ax_vals.legend(fontsize=7, ncol=3, loc="upper center")
    ax_vals.set_ylim(0, 0.95); ax_vals.grid(axis="x", visible=False)


def panel_t5(ax_crps, ax_var):
    if FINAL_SET == "final1":
        s = pd.read_csv(f"{A}/scoreboard.csv")
        s = s[(s.reference == "mean") & (s.subset == "all")].set_index("estimate")
        crps = {"EngressNet": s.loc["EngressNet (final)", "CRPS"], "Deterministic U-Net": s.loc["det-MSE", "MAE of mean"],
                "Bilinear": s.loc["bilinear", "CRPS"]}
    else:  # revised final model: evaluation/revisions/is2_crps_basis.py (FINAL_SET=final2), pass-cell means
        s = pd.read_csv(f"{BASE}/results/revisions_2026-11/is2_crps_basis{SFX}.csv", index_col=0).loc["pass-cell mean"]
        crps = {"EngressNet": s["EngressNet CRPS K=20 (per window, mean of 4)"], "Deterministic U-Net": s["Deterministic MAE (per window, mean of 4)"],
                "Bilinear": s["Bilinear MAE"]}
    for i, m in enumerate(METHODS):
        ax_crps.bar(i, crps[m], 0.6, color=METHOD_COLORS[m], edgecolor="white")
        ax_crps.text(i, crps[m], f"{crps[m]:.3f}", ha="center", va="bottom", fontsize=8)
    ax_crps.set_xticks(range(3)); ax_crps.set_xticklabels(["EngressNet", "Deterministic\n(MAE)", "Bilinear\n(MAE)"])
    ax_crps.set_ylabel("CRPS vs ICESat-2 (m)"); ax_crps.set_title("Table 5: ICESat-2 CRPS", loc="left"); ax_crps.grid(axis="x", visible=False)
    b = pd.read_csv(f"{A}/cice_ceiling_bar{SFX}.csv", index_col=0).iloc[:, 0]
    is2 = b["ICESat-2 (observed)"]
    var = {"EngressNet": b["EngressNet (final) on PIOMAS"] / is2, "Deterministic U-Net": b["Deterministic U-Net (det-MSE) on PIOMAS"] / is2,
           "Bilinear": b["PIOMAS (bilinear)"] / is2}
    for i, m in enumerate(METHODS):
        ax_var.bar(i, var[m], 0.6, color=METHOD_COLORS[m], edgecolor="white")
        ax_var.text(i, var[m], f"{var[m]:.2f}", ha="center", va="bottom", fontsize=8)
    ax_var.axhline(1.0, color=INK, lw=1.2, label="ICESat-2 = 1.0")
    for v, lab, ls in ((b["CICE (FOSI) truth"] / is2, "FOSI truth", "--"), (b["CICE (MESACLIP) truth"] / is2, "MESACLIP truth", ":")):
        ax_var.axhline(v, color=MUTED, lw=1.2, ls=ls, label=f"{lab} ({v:.2f})")
    ax_var.set_xticks(range(3)); ax_var.set_xticklabels(["EngressNet", "Deterministic", "Bilinear"])
    ax_var.set_ylabel("Within-1° std / ICESat-2"); ax_var.set_ylim(0, 1.1)
    ax_var.set_title("Table 5: sub-grid variability", loc="left"); ax_var.legend(fontsize=7, loc="upper right"); ax_var.grid(axis="x", visible=False)
    return crps, var


def panel_pooled(axs):
    """Like-for-like with the seed/window ensemble (12 det-MSE models: 3 seeds x 4 windows), which pools training
    information from all 4 windows: EngressNet pooled over the same 4 windows (80 members)."""
    pw = pd.read_csv(f"{BASE}/results/revisions_2026-11/pooled_window_ensembles.csv")
    sw = {ds: pd.read_csv(f"{DM}/{pre}_2015-2020_extra_E.csv", index_col=0)["Seed/window ensemble (det.)"].apply(pd.to_numeric, errors="coerce")
          for ds, pre in (("MESACLIP", "MESA"), ("FOSI", "FOSI"))}
    lab_e, lab_s = "EngressNet, 4 windows pooled (80 members)", "Seed/window ensemble (det.)"
    vals = {lab_e: {}, lab_s: {}}
    for ds in ("MESACLIP", "FOSI"):
        r = pw[(pw.dataset == ds) & pw.estimate.str.startswith("EngressNet")].iloc[0]
        vals[lab_e][ds] = {"CRPS": r["CRPS"], "RMSE": r["ens-mean RMSE"], "SSR": r["spread-skill"]}
        vals[lab_s][ds] = {"CRPS": sw[ds]["CRPS (point mass: MAE)"], "RMSE": sw[ds]["Ensemble-mean RMSE"], "SSR": sw[ds]["Spread-skill (corrected)"]}
    cols = {lab_e: METHOD_COLORS["EngressNet"], lab_s: METHOD_COLORS["Seed/window ensemble (det.)"]}
    for ax, (k, ylab) in zip(axs, (("CRPS", "CRPS (m)"), ("RMSE", "Ensemble-mean RMSE (m)"), ("SSR", "Spread-skill (1 = calibrated)"))):
        x = np.arange(2); w = 0.38
        for i, (m, v) in enumerate(vals.items()):
            yy = [v[ds][k] for ds in ("MESACLIP", "FOSI")]
            ax.bar(x + (i - 0.5) * w, yy, w * 0.92, color=cols[m], edgecolor="white", label=m)
            for xi, vi in zip(x + (i - 0.5) * w, yy):
                ax.text(xi, vi, f"{vi:.3f}" if k != "SSR" else f"{vi:.2f}", ha="center", va="bottom", fontsize=7)
        if k == "SSR":
            ax.axhline(1.0, color=INK, lw=1, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(["MESACLIP", "FOSI"]); ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    axs[0].set_title("Pooled over the 4 training windows", loc="left")
    axs[0].legend(fontsize=7, loc="upper left")


def panel_seedens(axs):
    """Revised final model: EngressNet (one model, 20 members) vs the seed ensemble (10 det-MSE models trained on the same
    1825 days, seeds 0-9), from spatial_prob_metrics.py --seedwin (FINAL_SET=final2)."""
    lab_e, lab_s = "EngressNet (1 model, 20 members)", "Seed ensemble (10 det. models)"
    vals = {lab_e: {}, lab_s: {}}
    for ds, pre in (("MESACLIP", "MESA"), ("FOSI", "FOSI")):
        d = pd.read_csv(f"{DM}{SFX}/{pre}_{FS[pre]['windows'][0]}.csv", index_col=0).apply(pd.to_numeric, errors="coerce")
        for lab, col in ((lab_e, "EngressNet"), (lab_s, "Seed/window ensemble (det.)")):
            vals[lab][ds] = {"CRPS": d.loc["CRPS (point mass: MAE)", col], "RMSE": d.loc["Ensemble-mean RMSE", col],
                             "SSR": d.loc["Spread-skill (corrected)", col]}
    cols = {lab_e: METHOD_COLORS["EngressNet"], lab_s: METHOD_COLORS["Seed/window ensemble (det.)"]}
    for ax, (k, ylab) in zip(axs, (("CRPS", "CRPS (m)"), ("RMSE", "Ensemble-mean RMSE (m)"), ("SSR", "Spread-skill (1 = calibrated)"))):
        x = np.arange(2); w = 0.38
        for i, (m, v) in enumerate(vals.items()):
            yy = [v[ds][k] for ds in ("MESACLIP", "FOSI")]
            ax.bar(x + (i - 0.5) * w, yy, w * 0.92, color=cols[m], edgecolor="white", label=m)
            for xi, vi in zip(x + (i - 0.5) * w, yy):
                ax.text(xi, vi, f"{vi:.3f}" if k != "SSR" else f"{vi:.2f}", ha="center", va="bottom", fontsize=7)
        if k == "SSR":
            ax.axhline(1.0, color=INK, lw=1, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(["MESACLIP", "FOSI"]); ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    axs[0].set_title("Probabilistic baseline: seed ensemble", loc="left")
    axs[0].legend(fontsize=7, loc="upper left")


f, a3 = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
if FINAL_SET == "final1":
    panel_pooled(a3)
    f.savefig(out("tab02b_pooled_window_baselines.png"), dpi=250)
else:
    panel_seedens(a3)
    f.savefig(out("tab02b_seed_ensemble_baseline.png"), dpi=250)
plt.close(f)

# ---- combined
fig = plt.figure(figsize=(17, 13), constrained_layout=True)
gs = fig.add_gridspec(3, 6)
panel_t2([fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])])
panel_t3(fig.add_subplot(gs[1, 0:2]))
panel_t4(fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6]))
crps, var = panel_t5(fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:6]))
fig.savefig(out("tab02-05_summary.png"), dpi=250)
plt.close(fig)
# ---- each panel alone
singles = {"tab02_headline.png": (lambda f: panel_t2(f.subplots(1, 3)), (13, 3.8)),
           "tab03_relative.png": (lambda f: panel_t3(f.subplots()), (6.5, 3.8)),
           "tab04a_bias.png": (lambda f: panel_t4(f.subplots(), plt.figure().add_subplot()), (7, 4)),
           "tab05_icesat2.png": (lambda f: panel_t5(*f.subplots(1, 2)), (10, 3.8))}
for name, (fn, size) in singles.items():
    f = plt.figure(figsize=size, constrained_layout=True)
    fn(f)
    f.savefig(out(name), dpi=250)
    plt.close("all")
f, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
panel_t4(a1, a2)
f.savefig(out("tab04_community.png"), dpi=250)
plt.close(f)
print("wrote tab02-05_summary, tab02_headline, tab03_relative, tab04_community, tab04a_bias, tab05_icesat2")
for ds in P:
    print(ds); print(P[ds][["RMSE", "Pattern Corr", "CRPS", "IIEE", "SSIM", "Grad MAE", "Bias"]].round(4).to_string())
print("Table 5 CRPS", {k: round(v, 4) for k, v in crps.items()}, "| sub-grid ratio", {k: round(v, 3) for k, v in var.items()})


matplotlib.rcParams.update(_rc_saved)  # undo this cell's style changes
show('tab02-05_summary.png', 'tab02_headline.png', 'tab03_relative.png', 'tab04_community.png', 'tab04a_bias.png', 'tab05_icesat2.png')
show("tab02b_seed_ensemble_baseline.png" if FINAL_SET == "final2" else "tab02b_pooled_window_baselines.png")

# %% [markdown]
# ## Sub-grid variability bar (ICESat-2 vs CICE truths vs downscaled PIOMAS)
#
# CICE sea-ice-thickness variability ceiling, as a simple bar chart (2026-10-08, advisor meeting; revised same day).
#
# Quantity: sub-grid variability = how much 0.1 deg thickness varies inside one 1 deg cell on one day. 1 deg is EngressNet's input
# grid (the coarse PIOMAS / CICE cells it downscales), so this is exactly the variability the downscaling has to add: the input itself
# carries none of it (bilinear PIOMAS ~ 0).
#
# Method (like-for-like): QC'd ICESat-2 pass-cell means (same QC as the evaluation notebooks), Oct 2018 - Dec 2021 (the years the CICE
# truth covers), grouped by (day, 1 deg input cell) with >= 6 pass-cells. Every source is sampled at exactly those pass-cells on the
# same day. Variability = pooled within-cell standard deviation, the standard within-group statistic:
#     s_pooled^2 = sum_i (n_i - 1) s_i^2 / sum_i (n_i - 1)
# over all groups i (n_i pass-cells, sample variance s_i^2). Ensembles / multi-window models (EngressNet 80 members, det-MSE 4 training
# windows, MESACLIP 6 members): pooled variance of each member, averaged over members, then sqrt (the variability of one realization).
# ICESat-2: its random measurement-noise variance (per thickness bin, from the along-track structure function) is pooled the same way
# and subtracted, so the bar is real thickness variability. Percentages = ratio to ICESat-2.
# Output: figures/final_figures/cice_ceiling_bar.png (copy: observations/figures/15b_cice_ceiling_bar.png).

# %%
import matplotlib
_rc_saved = matplotlib.rcParams.copy()

import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, f"{V6}/observations/processing")
import calibration_options_compare as c  # noqa: E402  (shared ICESat-2 pass-cell tools: QC masks, noise model)
sys.path.insert(0, f"{V6}/evaluation/revisions")
from final_set import FS, NAME as FINAL_SET  # noqa: E402  (2026-11: FINAL_SET=final2 -> revised final model)
SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"

OUT = f"{V6}/figures/final_figures/cice_ceiling_bar.png" if FINAL_SET == "final1" else f"{V6}/figures/v2/cice_ceiling_bar{SFX}.png"
NMIN = 6


def pooled_std(F, groups, dof):
    """sqrt of the dof-weighted mean within-group variance; F (n,) or (n, K) -> averaged over the K columns."""
    X = F[:, None] if F.ndim == 1 else F
    v = np.zeros(X.shape[1])
    for g in groups:
        v += (len(g) - 1) * X[g].var(axis=0, ddof=1)
    return float(np.sqrt(np.mean(v / dof)))


def main():
    ev = c.EV2
    ct, cb = np.load(f"{ev}/cice_truth_cells.npz"), np.load(f"{ev}/cice_truth_bins.npz")
    eng, engb = np.load(f"{ev}/{FS['cells_eng']}_cells.npz"), np.load(f"{ev}/{FS['cells_eng']}_bins.npz")  # final model (FINAL_SET)
    det = np.load(f"{ev}/{FS['cells_det']}_cells.npz")  # matched det-MSE
    F = {"CICE (FOSI)\ntruth": ct["fosi"].astype(float), "CICE (MESACLIP)\ntruth": ct["mesa"].astype(float),
         "EngressNet (final)\non PIOMAS": eng["members"].astype(float),
         "Deterministic U-Net\n(det-MSE) on PIOMAS": det["ensmean"].astype(float),
         "PIOMAS\n(bilinear)": eng["bilinear"].astype(float)}
    fin = lambda X: np.isfinite(X).all(1) if X.ndim == 2 else np.isfinite(X)
    y = c.PCD.h_mean.values.astype(float)
    ok = c.qcC & (c.PCD.date.values <= np.datetime64("2021-12-31")) & np.isfinite(y) & np.all([fin(X) for X in F.values()], 0)
    # ICESat-2 measurement-noise model sigma(h), from the along-track 10 km bins (as in the evaluation notebooks)
    FB = [cb["fosi"], cb["mesa"], engb["members"]]
    okb = c.qcB & (c.BIN.date.values <= np.datetime64("2021-12-31")) & np.all([fin(X.astype(float)) for X in FB], 0)
    sig_fn = c.noise_fn(c.BIN.h_mean.values.astype(float), c.contiguous_windows(okb))

    d = c.PCD.loc[ok, ["date", "clat", "clon"]].copy()
    d["row"] = np.where(ok)[0]
    groups = [g["row"].values for _, g in d.groupby(["date", "clat", "clon"]) if len(g) >= NMIN]
    dof = sum(len(g) - 1 for g in groups)
    obs_var = sum((len(g) - 1) * y[g].var(ddof=1) for g in groups) / dof
    noise_var = sum((len(g) - 1) * float(sig_fn(y[g].mean())) ** 2 for g in groups) / dof
    vals = {"ICESat-2\n(observed)": float(np.sqrt(max(obs_var - noise_var, 0)))}
    vals.update({k: pooled_std(X, groups, dof) for k, X in F.items()})
    ref = vals["ICESat-2\n(observed)"]
    print(f"{len(groups)} (day, 1° cell) groups, {sum(map(len, groups))} pass-cells; ICESat-2 raw {np.sqrt(obs_var):.3f} m, "
          f"noise {np.sqrt(noise_var):.3f} m")
    print({k.replace("\n", " "): f"{v:.3f} m ({v / ref:.0%})" for k, v in vals.items()})
    pd.Series({k.replace("\n", " "): v for k, v in vals.items()}, name="pooled within-1° std (m)").to_csv(
        f"{V6}/observations/data/alongtrack_eval/cice_ceiling_bar{SFX}.csv")

    INK, MUTED = "#14202b", "#57667a"
    col = {"ICESat-2\n(observed)": INK, "CICE (FOSI)\ntruth": "#7fa7c9", "CICE (MESACLIP)\ntruth": "#a9c3da",
           "EngressNet (final)\non PIOMAS": "#1d5d8f", "Deterministic U-Net\n(det-MSE) on PIOMAS": "#d08a4a",
           "PIOMAS\n(bilinear)": "#c9cfd6"}
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for i, (k, v) in enumerate(vals.items()):
        ax.bar(i, v, width=0.62, color=col[k])
        ax.text(i, v + ref * 0.025, f"{v:.2f} m" if i == 0 else f"{v:.2f} m\n{v / ref:.0%} of ICESat-2", ha="center",
                va="bottom", fontsize=9, color=INK, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    fosi = vals["CICE (FOSI)\ntruth"]
    ax.axhline(fosi, color=MUTED, ls="--", lw=1, zorder=1)
    ax.text(len(vals) - 0.5, fosi + ref * 0.012, "FOSI CICE: EngressNet's training target", ha="right", va="bottom",
            fontsize=8.5, color=MUTED)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals.keys(), fontsize=9)
    ax.set_ylabel("sub-grid variability (m)\nstd of 0.1° thickness within a 1° input cell", fontsize=9.5)
    ax.set_ylim(0, ref * 1.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Sea ice thickness variability inside EngressNet's 1° input cells, at ICESat-2 tracks (2018–2021)",
                 loc="left", fontsize=10)
    fig.text(0.01, -0.03, "Pooled within-cell standard deviation, every source sampled at the same ICESat-2 pass-cells on the same "
             "day. ICESat-2: measurement noise removed.\n"
             + ("EngressNet: average over its 80 ensemble members; det-MSE: over its 4 training windows; "
                if FINAL_SET == "final1" else "EngressNet: average over its 20 ensemble members; det-MSE: one model; ")
             + "MESACLIP: over its 6 members.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT, dpi=250, bbox_inches="tight")
    if FINAL_SET == "final1":
        shutil.copy(OUT, f"{V6}/observations/figures/15b_cice_ceiling_bar.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()


matplotlib.rcParams.update(_rc_saved)  # undo this cell's style changes
show()

# %% [markdown]
# # Supplementary figures (the unnumbered `figures/final_figures/` set)
#
# MESACLIP, test year 2021. "Deterministic" = det-MSE U-Net (see the Data cell).

# %% [markdown]
# ## Domain-mean SIT time series
#
# Output: `domain_mean_timeseries.png`

# %%
ts_df = pd.read_csv(os.path.join(AGG_DIR, "01_domain_mean_sit_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])

ts_plot_cols = ["truth", "stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"]
ts_plot_df = ts_df.copy()
if rdeb.ROLLING_DAYS_TIMESERIES:
    win = rdeb.rolling_window_samples(ts_plot_df["time"], rdeb.ROLLING_DAYS_TIMESERIES)
    ts_plot_df[ts_plot_cols] = ts_plot_df[ts_plot_cols].rolling(win, center=True, min_periods=1).mean()

sample = ncp.pal_sample("Manganese_r", 5)
fig, ax = plt.subplots(figsize=(10, 4))
for label, col, color, ls in [
    ("Truth", "truth", sample[0], "-"), ("EngressNet Ensemble Mean", "stochastic_unet_mean", sample[1], "-"),
    ("Deterministic", "deterministic_unet", sample[2], "-"), ("Bilinear", "bilinear", sample[3], "-"),
    ("PIOMAS", "piomas", sample[4], "--"),
]:
    if ts_plot_df[col].notna().sum() == 0:
        continue
    ax.plot(ts_plot_df["time"], ts_plot_df[col], label=label, color=color, ls=ls, lw=2 if col == "truth" else 1.5)
ax.set_ylabel("Domain-mean SIT (m)")
ax.legend(frameon=True, fontsize=10, ncols=2)
fig.autofmt_xdate()
ax.set_xlim(ts_plot_df["time"].min(), ts_plot_df["time"].max())
plt.tight_layout()
ax.grid(alpha=0.2)
plt.savefig(out("domain_mean_timeseries.png"), dpi=300, bbox_inches='tight')
plt.show()

show('domain_mean_timeseries.png')

# %% [markdown]
# ## Domain-mean pattern correlation time series
#
# Output: `domain_mean_pattern_corr.png`

# %%
pc_df = pd.read_csv(os.path.join(AGG_DIR, "01b_domain_mean_pattern_corr_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])

pc_plot_cols = ["bilinear", "piomas", "deterministic_unet", "stochastic_unet_mean"]
pc_plot_df = pc_df.copy()
if rdeb.ROLLING_DAYS_TIMESERIES:
    win = rdeb.rolling_window_samples(pc_plot_df["time"], rdeb.ROLLING_DAYS_TIMESERIES)
    pc_plot_df[pc_plot_cols] = pc_plot_df[pc_plot_cols].rolling(win, center=True, min_periods=1).mean()

sample = ncp.pal_sample("Manganese_r", 5)
fig, ax = plt.subplots(figsize=(10, 4))
for label, col, color, ls in [
    ("EngressNet Ensemble Mean", "stochastic_unet_mean", sample[1], "-"),
    ("Deterministic", "deterministic_unet", sample[2], "-"),
    ("Bilinear", "bilinear", sample[3], "-"),
    ("PIOMAS", "piomas", sample[4], "--"),
]:
    if pc_plot_df[col].notna().sum() == 0:
        continue
    ax.plot(pc_plot_df["time"], pc_plot_df[col], label=label, color=color, ls=ls, linewidth=1.5)
ax.set_ylabel("Domain-mean Pattern Correlation to Truth")
ax.set_xlim(pc_plot_df["time"].min(), pc_plot_df["time"].max())
ax.set_ylim(top=1)
ax.legend(frameon=True, fontsize=10, ncols=2)
ax.grid(alpha=0.2)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(out("domain_mean_pattern_corr.png"), dpi=300, bbox_inches='tight')
plt.show()

show('domain_mean_pattern_corr.png')

# %% [markdown]
# ## Community nearest-cell time series
#
# Output: `candidate_point_timeseries.png`

# %%
point_df = pd.read_csv(os.path.join(AGG_DIR, "02_candidate_point_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])
points_to_plot = ["Point Hope", "Kivalina", "Kotzebue", "Shishmaref", "Nome"]

sample = ncp.pal_sample("Manganese_r", 5)
fig, axs = plt.subplots(len(points_to_plot), 1, figsize=(10, 2.5 * len(points_to_plot)), sharex=True)
if len(points_to_plot) == 1:
    axs = [axs]
for ax, point_name in zip(axs, points_to_plot):
    sub = point_df[point_df["point"] == point_name]
    dist_km = sub["dist_km"].iloc[0] if len(sub) else np.nan
    piv = sub.pivot_table(index="time", columns="method", values="value").sort_index()
    ax.axhline(0, color="black", linewidth=1, linestyle=':')
    for label, col, color, ls, lw in [
        ("EngressNet Ensemble Mean", "stochastic_unet_mean", sample[1], "-", 1.2),
        ("Deterministic", "deterministic_unet", sample[2], "-", 1.2),
        ("Bilinear", "bilinear", sample[3], "-", 1.2),
        ("PIOMAS", "piomas", sample[4], "--", 1.2),
    ]:
        if col not in piv.columns or "truth" not in piv.columns:
            continue
        y = (piv[col] - piv["truth"]).dropna()
        if y.empty:
            continue
        if rdeb.ROLLING_DAYS_TIMESERIES:
            y = y.rolling(rdeb.rolling_window_samples(y.index.to_series(), rdeb.ROLLING_DAYS_TIMESERIES), center=True, min_periods=1).mean()
        ax.plot(y.index, y, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.set_title(f"{point_name} (nearest valid ocean grid cell ~{dist_km:.0f} km)", fontsize=12)
    ax.set_xlim(sub["time"].min(), sub["time"].max())
    ax.set_ylim(-0.3, 1.3)
    ax.grid(alpha=0.2)
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], labels=['-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
    ax.set_ylabel("Pred - Truth SIT (m)")
    if point_name == "Point Hope":
        ax.legend(fontsize=10, loc="upper right", ncols=2, frameon=True)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(out("candidate_point_timeseries.png"), dpi=300, bbox_inches='tight')
plt.show()

show('candidate_point_timeseries.png')

# %% [markdown]
# ## Absolute-error maps
#
# Output: `error_figure.png`

# %%
edir = os.path.join(RESULTS_DIR, "eval_data")
with open(os.path.join(edir, "meta.json")) as f:
    meta = json.load(f)
bbox = meta["bbox"]
candidate_points = {**meta["candidate_points"], "Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
with open(os.path.join(edir, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
fields = np.load(os.path.join(edir, "fields.npz"))
Y_base_phys = fields["Y_base_phys"]
Y_pred_det_phys = DET_FIELDS["Y_pred_det_phys"]  # det-MSE U-Net, not the zero-noise pass
Y_pred_phys = fields["Y_pred_phys"]
Y_test_phys = fields["Y_test_phys"]
test_tile_ids = fields["test_tile_ids"]

proj, boundary_path, central_lon = rdeb.make_polar_proj(bbox)


def _style(ax, lon_=None, lat_=None):
    rdeb.style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_, lat_)


N_SAMPLES = min(3, Y_test_phys.shape[0])
MIN_ICE_THICKNESS = 0.5
MIN_ICE_FRAC = 0.25
ice_frac = (Y_test_phys[:, 0] > MIN_ICE_THICKNESS).mean(axis=(1, 2))
valid_idxs = np.where(ice_frac > MIN_ICE_FRAC)[0]
SAMPLE_IDXS = np.random.default_rng(0).choice(valid_idxs, N_SAMPLES, replace=False)

ERR_VMIN, ERR_VMAX = 0, 2
ERR_CMAP = "viridis"
panel_titles_err = ["Bilinear", "Deterministic", "Ensemble Mean"]
fig, axs = plt.subplots(len(SAMPLE_IDXS), 3, figsize=(9, 2.5 * len(SAMPLE_IDXS)), constrained_layout=True, subplot_kw={"projection": proj})
if len(SAMPLE_IDXS) == 1:
    axs = axs[None, :]
for row, idx in enumerate(SAMPLE_IDXS):
    geo = tile_geometry[int(test_tile_ids[idx])]
    tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]
    truth = Y_test_phys[idx, 0]
    bilinear_ae = np.abs(Y_base_phys[idx, 0] - truth)
    det_ae = np.abs(Y_pred_det_phys[idx, 0] - truth)
    ens_ae = np.abs(Y_pred_phys[idx, 0] - truth)
    for col, field in enumerate([bilinear_ae, det_ae, ens_ae]):
        ax = axs[row, col]
        im = ax.pcolormesh(tgt_lon, tgt_lat, field, transform=ccrs.PlateCarree(), cmap=ERR_CMAP, vmin=ERR_VMIN, vmax=ERR_VMAX, shading="auto")
        _style(ax, tgt_lon, tgt_lat)
        if row == 0:
            ax.set_title(panel_titles_err[col], fontsize=18)
    axs[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=13)
cbar = fig.colorbar(im, ax=axs, aspect=20, shrink=0.9, pad=0.02)
cbar.set_label("Absolute Error to Truth (m)", fontsize=18)
cbar.ax.tick_params(labelsize=14)
plt.savefig(out("error_figure.png"), dpi=300, bbox_inches='tight')
plt.show()

show('error_figure.png')

# %% [markdown]
# ## Temporal pattern-correlation maps
#
# Output: `temporal_pattern_corr.png`

# %%
edir = os.path.join(RESULTS_DIR, "eval_data")
with open(os.path.join(edir, "meta.json")) as f:
    meta = json.load(f)
bbox = meta["bbox"]
candidate_points = {**meta["candidate_points"], "Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
with open(os.path.join(edir, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
fields = np.load(os.path.join(edir, "fields.npz"))
Y_base_phys = fields["Y_base_phys"]
Y_pred_det_phys = DET_FIELDS["Y_pred_det_phys"]  # det-MSE U-Net, not the zero-noise pass
Y_pred_phys = fields["Y_pred_phys"]
Y_test_phys = fields["Y_test_phys"]
mask_test = fields["mask_test"]
sample_times_df = pd.read_csv(os.path.join(edir, "sample_times.csv"), parse_dates=["time"])

proj, boundary_path, central_lon = rdeb.make_polar_proj(bbox)


def _style(ax, lon_=None, lat_=None):
    rdeb.style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_, lat_)


MONTH = 3
geo = tile_geometry[0]
tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]
land_hw = mask_test[0, 0] > 0.5
month_idx = np.where(sample_times_df["time"].dt.month.values == MONTH)[0]

methods_maps = {}
for label, field in [("Bilinear", Y_base_phys), ("Deterministic", Y_pred_det_phys), ("EngressNet Ensemble Mean", Y_pred_phys)]:
    corr_map = rdeb.temporal_pixel_corr(field, Y_test_phys, month_idx)
    methods_maps[label] = np.where(land_hw, np.nan, corr_map)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
for ax, (label, corr_map) in zip(axs, methods_maps.items()):
    im = ax.pcolormesh(tgt_lon, tgt_lat, corr_map, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
    _style(ax, tgt_lon, tgt_lat)
    ax.set_title(label, fontsize=18)
cbar = fig.colorbar(im, ax=axs, aspect=20, shrink=0.9, pad=0.02)
cbar.set_label("Temporal Correlation to Truth (March)", fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.savefig(out("temporal_pattern_corr.png"), dpi=300, bbox_inches='tight')
plt.show()

show('temporal_pattern_corr.png')

# %% [markdown]
# ## Bias and std-ratio maps
#
# Output: `bias_std_ratio_maps.png`

# %%
edir = os.path.join(RESULTS_DIR, "eval_data")
with open(os.path.join(edir, "meta.json")) as f:
    meta = json.load(f)
bbox = meta["bbox"]
candidate_points = {**meta["candidate_points"], "Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
with open(os.path.join(edir, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
fields = np.load(os.path.join(edir, "fields.npz"))
Y_base_phys = fields["Y_base_phys"]
Y_pred_det_phys = DET_FIELDS["Y_pred_det_phys"]  # det-MSE U-Net, not the zero-noise pass
Y_pred_phys = fields["Y_pred_phys"]
Y_test_phys = fields["Y_test_phys"]
mask_test = fields["mask_test"]

proj, boundary_path, central_lon = rdeb.make_polar_proj(bbox)


def _style(ax, lon_=None, lat_=None):
    rdeb.style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_, lat_)


geo = tile_geometry[0]
tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]
land_hw = mask_test[0, 0] > 0.5
BIAS_LIM = 0.5
STD_RATIO_LIM = (0, 2)
method_fields = {"Bilinear": Y_base_phys, "Deterministic": Y_pred_det_phys, "EngressNet Ensemble Mean": Y_pred_phys}
truth_mean_hw = Y_test_phys[:, 0].mean(axis=0)
truth_std_hw = Y_test_phys[:, 0].std(axis=0)
bias_maps, std_ratio_maps = {}, {}
for label, field in method_fields.items():
    pred_mean_hw = field[:, 0].mean(axis=0)
    pred_std_hw = field[:, 0].std(axis=0)
    bias_maps[label] = np.where(land_hw, np.nan, pred_mean_hw - truth_mean_hw)
    safe_truth_std = np.where(truth_std_hw > 1e-8, truth_std_hw, np.nan)
    std_ratio_maps[label] = np.where(land_hw, np.nan, pred_std_hw / safe_truth_std)

fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
for col, (label, bmap) in enumerate(bias_maps.items()):
    ax = axs[0, col]
    im0 = ax.pcolormesh(tgt_lon, tgt_lat, bmap, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-BIAS_LIM, vmax=BIAS_LIM, shading="auto")
    _style(ax, tgt_lon, tgt_lat)
    ax.set_title(label, fontsize=18)
for col, (label, rmap) in enumerate(std_ratio_maps.items()):
    ax = axs[1, col]
    im1 = ax.pcolormesh(tgt_lon, tgt_lat, rmap, transform=ccrs.PlateCarree(), cmap="RdBu_r",
                        norm=TwoSlopeNorm(vcenter=1, vmin=STD_RATIO_LIM[0], vmax=STD_RATIO_LIM[1]), shading="auto")
    _style(ax, tgt_lon, tgt_lat)

cbar1 = fig.colorbar(im0, ax=axs[0, :], aspect=20, shrink=0.8, pad=0.02)
cbar1.set_label("Time-mean Bias (m)", fontsize=14)
cbar1.ax.tick_params(labelsize=14)

cbar2 = fig.colorbar(im1, ax=axs[1, :], aspect=20, shrink=0.8, pad=0.02)
cbar2.set_label("Pred/Truth", fontsize=14)
cbar2.ax.tick_params(labelsize=14)

plt.savefig(out("bias_std_ratio_maps.png"), dpi=300, bbox_inches='tight')
plt.show()

show('bias_std_ratio_maps.png')

# %% [markdown]
# ## Domain-mean bias time series
#
# Output: `domain_mean_bias_timseries.png`

# %%
ts_df = pd.read_csv(os.path.join(AGG_DIR, "01_domain_mean_sit_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])
bias_df = ts_df.copy()
bias_df["bias_stochastic_unet_mean"] = bias_df["stochastic_unet_mean"] - bias_df["truth"]
bias_df["bias_deterministic_unet"] = bias_df["deterministic_unet"] - bias_df["truth"]
bias_df["bias_bilinear"] = bias_df["bilinear"] - bias_df["truth"]
bias_df["bias_piomas"] = bias_df["piomas"] - bias_df["truth"]
bias_cols = ["bias_stochastic_unet_mean", "bias_deterministic_unet", "bias_bilinear", "bias_piomas"]

bias_plot_df = bias_df.copy()
if rdeb.ROLLING_DAYS_BATCH:
    win = rdeb.rolling_window_samples(bias_plot_df["time"], rdeb.ROLLING_DAYS_BATCH)
    bias_plot_df[bias_cols] = bias_plot_df[bias_cols].rolling(win, center=True, min_periods=1).mean()

sample = ncp.pal_sample("Manganese_r", 5)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axhline(0, color="black", linewidth=1, linestyle=":")
for label, col, ls, color in [
    ("EngressNet Ensemble Mean", "bias_stochastic_unet_mean", "-", sample[1]),
    ("Deterministic", "bias_deterministic_unet", "-", sample[2]),
    ("Bilinear", "bias_bilinear", "-", sample[3]),
    ("PIOMAS", "bias_piomas", "--", sample[4]),
]:
    if bias_plot_df[col].notna().sum() == 0:
        continue
    ax.plot(bias_plot_df["time"], bias_plot_df[col], label=label, color=color, linewidth=1.5, linestyle=ls)
ax.set_ylabel("Domain-mean Pred - Truth SIT (m)")
ax.legend(fontsize=12, ncols=2, loc="upper right")
ax.set_xlim(bias_plot_df["time"].min(), bias_plot_df["time"].max())
ax.grid(alpha=0.2)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(out("domain_mean_bias_timseries.png"), dpi=300, bbox_inches='tight')
plt.show()

show('domain_mean_bias_timseries.png')

# %% [markdown]
# ## Anomaly-correlation maps
#
# Output: `anomaly_correlation.png`

# %%
edir = os.path.join(RESULTS_DIR, "eval_data")
with open(os.path.join(edir, "meta.json")) as f:
    meta = json.load(f)
bbox = meta["bbox"]
candidate_points = {**meta["candidate_points"], "Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
with open(os.path.join(edir, "tile_geometry.pkl"), "rb") as f:
    tile_geometry = pickle.load(f)
fields = np.load(os.path.join(edir, "fields.npz"))
Y_base_phys = fields["Y_base_phys"]
Y_pred_det_phys = DET_FIELDS["Y_pred_det_phys"]  # det-MSE U-Net, not the zero-noise pass
Y_pred_phys = fields["Y_pred_phys"]
Y_test_phys = fields["Y_test_phys"]
mask_test = fields["mask_test"]
sample_times_df = pd.read_csv(os.path.join(edir, "sample_times.csv"), parse_dates=["time"])

proj, boundary_path, central_lon = rdeb.make_polar_proj(bbox)


def _style(ax, lon_=None, lat_=None):
    rdeb.style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_, lat_)


geo = tile_geometry[0]
tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]
land_hw = mask_test[0, 0] > 0.5
method_fields = {"Bilinear": Y_base_phys, "Deterministic UNet": Y_pred_det_phys, "Stochastic UNet Mean": Y_pred_phys}

months = sample_times_df["time"].dt.month.values
all_idx = np.arange(len(months))
truth_anom = rdeb.remove_monthly_climatology(Y_test_phys, months)
acc_maps = {}
for label, field in method_fields.items():
    pred_anom = rdeb.remove_monthly_climatology(field, months)
    acc_maps[label] = np.where(land_hw, np.nan, rdeb.temporal_pixel_corr(pred_anom, truth_anom, all_idx))

fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
for ax, (label, amap) in zip(axs, acc_maps.items()):
    im = ax.pcolormesh(tgt_lon, tgt_lat, amap, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    _style(ax, tgt_lon, tgt_lat)
    ax.set_title(label, fontsize=18)
cbar = fig.colorbar(im, ax=axs, aspect=20, shrink=0.9, pad=0.02)
cbar.set_label("Anomaly Correlation", fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.savefig(out("anomaly_correlation.png"), dpi=300, bbox_inches='tight')

plt.show()

show('anomaly_correlation.png')

# %% [markdown]
# ## Taylor diagram
#
# Output: `taylor_diagram.png`

# %%
taylor_df = pd.read_csv(os.path.join(AGG_DIR, "11_taylor_stats_4window.csv"), index_col=0)
taylor_stats = {row["Method"]: (row["Correlation (R)"], row["Std Ratio"], row["Centered RMSE (m)"])
                for _, row in taylor_df.iterrows()}


def rms_circle(center, radius, theta_lim=(0, np.pi / 2), n=200):
    t = np.linspace(0, 2 * np.pi, n)
    x, y = center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)
    theta, r = np.arctan2(y, x), np.sqrt(x ** 2 + y ** 2)
    valid = (theta >= theta_lim[0]) & (theta <= theta_lim[1])
    return theta[valid], r[valid]


max_std = max(sr for _, sr, _ in taylor_stats.values())
r_lim = max(1.3, max_std * 1.2)

sample = ncp.pal_sample("Manganese_r", 5)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, polar=True)
ax.set_thetamin(0)
ax.set_thetamax(90)
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_ylim(0, r_lim)
corr_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
ax.set_xticks(np.arccos(corr_ticks))
ax.set_xticklabels([str(c) for c in corr_ticks])
ax.set_xlabel("Correlation", labelpad=15)
ax.set_ylabel("Normalized Standard Deviation", labelpad=10)
for rms in np.arange(0.5, r_lim, 0.5):
    th, r = rms_circle((1, 0), rms)
    ax.plot(th, r, color="gray", linestyle=":", linewidth=0.8)
std_circle_theta = np.linspace(0, np.pi / 2, 100)
ax.plot(std_circle_theta, np.ones_like(std_circle_theta), color="black", linestyle="--", linewidth=0.8)
ax.plot([0], [1], marker="*", color=sample[0], markersize=16, linestyle="none", label="Truth")
colors = {"Bilinear": sample[3], "Deterministic UNet": sample[2], "Stochastic UNet Mean": sample[1]}
display_name = {"Bilinear": "Bilinear", "Deterministic UNet": "Deterministic", "Stochastic UNet Mean": "EngressNet Ensemble Mean"}
for method, (R, std_ratio, crmse) in taylor_stats.items():
    ax.plot([np.arccos(np.clip(R, -1, 1))], [std_ratio], marker="o", markersize=10,
            color=colors.get(method, method), label=display_name.get(method, method), linestyle="none")
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.25), fontsize=11, ncols=2, frameon=False)
plt.tight_layout()
plt.savefig(out("taylor_diagram.png"), dpi=300, bbox_inches='tight')
plt.show()

show('taylor_diagram.png')

# %% [markdown]
# ## Domain-mean power spectra
#
# Output: `domain_mean_psd.png`

# %%
psd_df = pd.read_csv(os.path.join(AGG_DIR, "13_psd_data_4window.csv"), index_col=0)
method_cols = [c for c in psd_df.columns if c not in ("wavenumber_cycles_per_km", "wavelength_km")]

sample = ncp.pal_sample("Manganese_r", 5)
LABELS = {
    "Truth": "Truth",
    "Stochastic UNet Mean": "EngressNet Ensemble Mean",
    "Deterministic UNet": "Deterministic",
    "Bilinear": "Bilinear",
    "PIOMAS (obs)": "PIOMAS",
}
LINESTYLES = {
    "Truth": "-",
    "Stochastic UNet Mean": "-",
    "Deterministic UNet": "-",
    "Bilinear": "-",
    "PIOMAS (obs)": "--",
}

fig, ax = plt.subplots(figsize=(6, 4))
wavenumber = psd_df["wavenumber_cycles_per_km"].values
for i, c in enumerate(method_cols):
    valid = (wavenumber > 0) & (psd_df[c] > 0)
    label = LABELS.get(c, c)
    ax.plot(1/wavenumber[valid], 1/psd_df[c][valid],
            label=label, color=sample[i],
            ls=LINESTYLES.get(c, c), lw=2 if label == "Truth" else 1.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Wavelength (km)")
ax.set_ylabel("PSD")
ax.legend(fontsize=10, ncols=2, loc='upper right')
ax.grid(True, which="both", alpha=0.2)
plt.tight_layout()
plt.title("2021 Annual-Mean")
plt.savefig(out("domain_mean_psd.png"), dpi=300, bbox_inches='tight')

plt.show()

show('domain_mean_psd.png')

# %% [markdown]
# ## Per-member power-spectrum spread
#
# Output: `per_member_psd_spread.png`

# %%
psdb_df = pd.read_csv(os.path.join(AGG_DIR, "13b_psd_per_member_spread_data_4window.csv"), index_col=0)

sample = ncp.pal_sample("Manganese_r", 5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
wavenumber = psdb_df["wavenumber_cycles_per_km"].values
valid = (wavenumber > 0) & (psdb_df["Truth"] > 0)

ax1.plot(1/wavenumber[valid], 1/psdb_df["Truth"][valid], label="Truth", color=sample[0], linewidth=2)
ax1.fill_between(1/wavenumber[valid], 1/psdb_df["Member p10"][valid], 1/psdb_df["Member p90"][valid],
                 color=sample[1], alpha=0.25, label="EngressNet Members (10th–90th pct.)")
ax1.plot(1/wavenumber[valid], 1/psdb_df["Member mean"][valid], color=sample[1], linewidth=1.5, label="EngressNet Ensemble Mean")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Wavelength (km)")
ax1.set_ylabel("PSD")
ax1.set_title("Per-member PSD Spread")
ax1.legend(fontsize=10)
ax1.grid(True, which="both", alpha=0.2)

ax2.plot(1/wavenumber[valid], psdb_df["Relative member spread pct"][valid], color=sample[1], linewidth=1.5)
ax2.set_xscale("log")
ax2.set_xlabel("Wavelength (km)")
ax2.set_ylabel("Relative member spread (%)")
ax2.set_title("Inter-member Spectral-Texture Spread")
ax2.grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.savefig(out("per_member_psd_spread.png"), dpi=300, bbox_inches="tight")

plt.show()

show('per_member_psd_spread.png')

# %% [markdown]
# ## CRPS by SIT regime
#
# Output: `crps_regime.png`

# %%
crps_df = pd.read_csv(os.path.join(AGG_DIR, "16_crps_by_regime_data_4window.csv"), index_col=0)

sample = ncp.pal_sample("Sodium", 8)

fig, ax = plt.subplots(figsize=(8, 6))
plot_rows = crps_df.iloc[1:]
ax.bar(np.arange(len(plot_rows)), plot_rows["CRPS (m)"], color=sample[3], alpha=0.85)
ax.axhline(crps_df.iloc[0]["CRPS (m)"], color="black", linestyle="--", linewidth=2, 
           label=f"Domain-mean CRPS ({crps_df.iloc[0]['CRPS (m)']:.2f} m)")
ax.set_xticks(np.arange(len(plot_rows)))
ax.set_xticklabels(plot_rows["SIT regime"], fontsize=12, rotation=45, ha="right")
ax.set_ylabel("CRPS (m)", fontsize=14)
ax.set_title("CRPS by SIT Regime (test=2021)", fontsize=14)
ax.legend(fontsize=12, frameon=False)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig(out("crps_regime.png"), dpi=300, bbox_inches="tight")

plt.show()

show('crps_regime.png')

# %% [markdown]
# ## Fig. 8: observed seasonal cycle (PIOMAS vs FOSI / MESACLIP truth; model-independent)
#
# Output: `fig08_obs_seasonal_cycle.png`

# %%
seasonal_csv = os.path.join(BASE, "figures", "piomas_fosi_mesa_seasonal_cycle_2021.csv")
seasonal_df = pd.read_csv(seasonal_csv, index_col=0)

sample1 = ncp.pal_sample("Manganese_r", 5)
sample2 = ncp.pal_sample("Magnesium", 8)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(seasonal_df.index, seasonal_df["PIOMAS (obs)"], color=sample1[4], linewidth=2, linestyle='--', label="PIOMAS")
ax.plot(seasonal_df.index, seasonal_df["FOSI truth"], color=sample2[6], linewidth=2, label="FOSI")
ax.plot(seasonal_df.index, seasonal_df["MESA truth"], color=sample2[4], linewidth=2, label="MESACLIP")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_ylabel("Domain-mean SIT (m)")
ax.set_xlim(1, 12)
ax.set_ylim(0, 1.91)
ax.grid(alpha=0.2)
ax.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.savefig(out("fig08_obs_seasonal_cycle.png"), dpi=300, bbox_inches='tight')

plt.show()

show('fig08_obs_seasonal_cycle.png')

# %% [markdown]
# ## Seasonal cycle: Truth, EngressNet mean ± std, Deterministic, Bilinear
#
# Output: `seasonal_cycle.png`. Monthly means of the domain-mean SIT (2021, ocean cells; same pooling as the time-series
# figures). Two choices for the EngressNet band (`BAND` at the top of the cell):
# - `"cell"` (default): ±1 domain-mean per-cell ensemble std, i.e. the typical spread of the members at a grid cell
#   (about 0.1 m in winter; the quantity in Fig. 7).
# - `"domain"`: ±1 std across members of each member's domain mean. This is under 1 mm, because member differences
#   cancel when averaged over ~46,000 ocean cells, so the band is invisible.
#
# The first run computes both from `fields.npz` (about 8 GB, a few minutes) and caches them in
# `data_<FINAL_SET>/MESA_aggregate/seasonal_cycle_monthly.csv`; later runs only replot. Delete that CSV to recompute.
# %%
BAND = "cell"  # "cell": mean per-cell ensemble std (visible); "domain": std of the members' domain means (< 1 mm)
SEAS_CSV = os.path.join(DATA_DIR, "seasonal_cycle_monthly.csv")
if not os.path.exists(SEAS_CSV) or "engressnet_cell_std" not in pd.read_csv(SEAS_CSV, nrows=1).columns:
    ts = pd.read_csv(os.path.join(AGG_DIR, "01_domain_mean_sit_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])
    dom_std, cell_std = [], []
    for rd in sorted(d for d in glob.glob(os.path.join(BASE, "results", RUN, "*")) if os.path.isdir(d)):
        fz = np.load(os.path.join(rd, "eval_data", "fields.npz"))
        ocean = (1.0 - fz["mask_test"][0, 0]).clip(0, 1)  # same weights as run_daily_eval_batch.domain_mean
        dmean = lambda a: (a * ocean).sum(axis=(-2, -1)) / ocean.sum()
        win = pd.read_csv(os.path.join(BASE, "evaluation", "saved_figs", RUN, os.path.basename(rd), "01_domain_mean_sit_timeseries_data.csv"),
                          index_col=0, parse_dates=["time"])
        times = win.set_index("sample_idx")["time"].reindex(range(fz["Y_test_phys"].shape[0])).values
        truth = pd.Series(dmean(fz["Y_test_phys"][:, 0]), index=times).groupby(level=0).mean()
        assert np.allclose(truth.reindex(ts["time"]).values, ts["truth"].values, atol=1e-4), "sample/date mapping mismatch"
        P = fz["preds_all_phys"]  # (samples, members, 1, H, W)
        mem = pd.DataFrame(np.stack([dmean(P[:, k, 0]) for k in range(P.shape[1])], axis=1), index=times).groupby(level=0).mean()
        dom_std.append(mem.groupby(mem.index.month).mean().std(axis=1, ddof=1))  # spread of the members' monthly domain means
        cs = np.concatenate([dmean(P[i:i + 100, :, 0].std(axis=1, ddof=1)) for i in range(0, P.shape[0], 100)])  # per-cell spread
        cs = pd.Series(cs, index=times).groupby(level=0).mean()
        cell_std.append(cs.groupby(cs.index.month).mean())
        del P, fz
    clim = ts.assign(month=ts["time"].dt.month).groupby("month")[["truth", "stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"]].mean()
    clim["engressnet_member_std"] = pd.concat(dom_std, axis=1).mean(axis=1)
    clim["engressnet_cell_std"] = pd.concat(cell_std, axis=1).mean(axis=1)
    clim.to_csv(SEAS_CSV)
clim = pd.read_csv(SEAS_CSV, index_col=0)

sample = ncp.pal_sample("Manganese_r", 5)
fig, ax = plt.subplots(figsize=(8, 4))
sd = clim["engressnet_cell_std"] if BAND == "cell" else clim["engressnet_member_std"]
ax.fill_between(clim.index, (clim["stochastic_unet_mean"] - sd).clip(lower=0), clim["stochastic_unet_mean"] + sd,
                color=sample[1], alpha=0.35, lw=0, label="EngressNet ±1 std")
for col, label, ls, lw, color in [
    ("truth", "Truth", "-", 2, sample[0]),
    ("stochastic_unet_mean", "EngressNet Ensemble Mean", "-", 1.5, sample[1]),
    ("deterministic_unet", "Deterministic", "-", 1.5, sample[2]),
    ("bilinear", "Bilinear", "-", 1.5, sample[3]),
    # ("piomas", "PIOMAS", "--", 1.5, sample[4]),  # left out so the methods' differences are visible
]:
    if clim[col].notna().sum() == 0:
        continue
    ax.plot(clim.index, clim[col], label=label, color=color, linestyle=ls, linewidth=lw)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_ylabel("Domain-mean SIT (m)")
ax.set_xlim(1, 12)
ax.set_ylim(0, 1.2)
ax.legend(fontsize=11, ncols=1, loc="upper right")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(out("seasonal_cycle.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(clim.round(4).to_string())
show("seasonal_cycle.png")

# %% [markdown]
# ## Final results table (`final_results_table.csv` / `.md`)

# %%
# Perfect model (2021, ocean cells) and ICESat-2 (pass-cell means, PIOMAS input, Oct 2018 - May 2022), from the saved results.
R = os.path.join(BASE, "results", "revisions_2026-11")
SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"
t23 = pd.read_csv(os.path.join(R, f"tables23_three_methods{SFX}.csv"))
rows = []
for ds, key in (("FOSI", "FOSI"), ("MESACLIP", "MESA")):
    cal = pd.read_csv(os.path.join(BASE, "evaluation", "corrected_calibration", f"{FS[key]['eng']}.csv"))
    cal = cal[cal.run != "POOLED"]
    for method, lab in (("EngressNet ensemble mean", "EngressNet"), ("Deterministic U-Net", "deterministic U-Net"), ("Bilinear", "bilinear")):
        r = t23[(t23.dataset == ds) & (t23.method == method)].iloc[0]
        rows.append({"test": f"perfect model, {ds} 2021", "estimate": lab, "RMSE": r["RMSE"], "MAE": r["MAE"],
                     "CRPS": r["CRPS (single-valued: MAE)"], "spread-skill": cal["Spread-skill ratio"].mean() if lab == "EngressNet" else np.nan,
                     "bias": r["Bias"], "r": r["Pattern Corr"], "IIEE (1e-4 m)": r["IIEE"], "IIEE (0.15 m)": r["IIEE thr=0.15"]})
is2 = pd.read_csv(os.path.join(R, f"is2_crps_basis{SFX}.csv"), index_col=0).loc["pass-cell mean"]
sub = pd.read_csv(os.path.join(BASE, "observations", "data", "alongtrack_eval", f"cice_ceiling_bar{SFX}.csv"), index_col=0).iloc[:, 0]
is2_std = sub.loc["ICESat-2 (observed)"]
for lab, crps_col, rmse_col, sub_key in (
        ("EngressNet", [c for c in is2.index if c.startswith("EngressNet CRPS K=20")][0], "EngressNet RMSE of mean", "EngressNet (final) on PIOMAS"),
        ("deterministic U-Net", [c for c in is2.index if c.startswith("Deterministic MAE (per window")][0], "Deterministic RMSE", "Deterministic U-Net (det-MSE) on PIOMAS"),
        ("bilinear", "Bilinear MAE", "Bilinear RMSE", "PIOMAS (bilinear)")):
    rows.append({"test": "ICESat-2 pass-cell mean, PIOMAS input, 2018-22", "estimate": lab, "RMSE": is2[rmse_col],
                 "CRPS": is2[crps_col], "bias": is2["EngressNet bias"] if lab == "EngressNet" else np.nan,
                 "within-1 deg std / ICESat-2": sub.loc[sub_key] / is2_std})
table = pd.DataFrame(rows).set_index(["test", "estimate"]).astype(float).round(4)
table.to_csv(out("final_results_table.csv"))
with open(out("final_results_table.md"), "w") as f:
    r_ = table.reset_index().astype(object).where(table.reset_index().notna(), "")
    f.write("| " + " | ".join(r_.columns) + " |\n|" + "---|" * len(r_.columns) + "\n")
    for _, row in r_.iterrows():
        f.write("| " + " | ".join(str(v) for v in row) + " |\n")
table

# %% [markdown]
# ## Not in this notebook
#
# `engressnet_on_products_2021-03-10.png` (EngressNet run on other SIT products) needs new inference runs of the revised model
# on those products; ask before adding it.
