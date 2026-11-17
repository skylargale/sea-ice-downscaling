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
# # figures/v2
#
# **How to use**
# 1. Open on a Casper compute node (JupyterHub "Casper PBS batch", ~64 GB; kernel `seaice-downscaling`).
# 2. Run every cell from the top down to **Paper figures** once (click the first cell under *Paper figures*, then
#    *Run All Above*). That is setup, shared colours, data preparation and the data loaders.
# 3. Each cell below that makes one figure with ordinary matplotlib code: change titles, labels, colours, limits, sizes
#    directly in the cell and run it. It saves the PNG into `figures/v2/` (overwriting the current file) and shows it.
#    Figure cells run in any order.
#
# Table charts: the bars of each table are drawn by a `panel_*` function (section *Table panels*), because the summary page
# redraws all of them; change a table's styling there, its size and file name in its figure cell.
#
# `FINAL_SET` in Setup picks the model (`"final2"` = revised final model, the paper; `"final1"` = submitted model, written to
# `figures/v2/superseded_final1/`). "Deterministic" is the det-MSE U-Net trained on the same days.
# This notebook is written from `make_v2_figures.py` (`python make_v2_figures.py --write-notebook`); copy notebook edits back
# with `python make_v2_figures.py --from-notebook` before regenerating. `python make_v2_figures.py` makes every figure.

# %% [markdown]
# ## Setup

# %%
FINAL_SET = "final2"   # "final2" = revised final model (paper); "final1" = submitted model -> figures/v2/superseded_final1/

try:  # notebook / VS Code: inline figures; plain python: no display
    get_ipython().run_line_magic("matplotlib", "inline")
    _IN_NOTEBOOK = True
except NameError:
    import matplotlib
    matplotlib.use("Agg")
    _IN_NOTEBOOK = False

import glob
import importlib
import json
import os
import pickle
import shutil
import sys
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib
import matplotlib.pyplot as plt
import nicopal as ncp
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LightSource, LinearSegmentedColormap, LogNorm, TwoSlopeNorm

BASE = "/glade/work/skygale/projects/SeaIceDownscaling"
V2 = f"{BASE}/figures/v2"
for _p in (f"{BASE}/evaluation/revisions", f"{BASE}/training", f"{BASE}/evaluation", f"{BASE}/observations/processing"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["FINAL_SET"] = FINAL_SET
import final_set  # noqa: E402  (evaluation/revisions/final_set.py: batch names of each final model)
importlib.reload(final_set)  # pick up a changed FINAL_SET
import functions_engressnet as fe  # noqa: E402
import run_daily_eval_batch as rdeb  # noqa: E402

FS = final_set.SETS[FINAL_SET]
SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"
OUT_DIR = V2 if FINAL_SET == "final2" else os.path.join(V2, "superseded_final1")  # final1 never overwrites the paper set
RUN = FS["MESA"]["eng"]       # EngressNet batch (MESACLIP) used by the map and supplementary figures
WINDOW = FS["map_window"]     # training-window substring of the run folder


def out(name):
    """Path of an output file in the output folder."""
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


def resolve_run_window_dir(base_subdir, run, window):
    """The single results/ or evaluation/saved_figs/ subfolder for (run, window); folder names embed a PBS job id."""
    matches = sorted(d for d in glob.glob(os.path.join(BASE, base_subdir, run, f"*{window}*")) if os.path.isdir(d))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {run}/*{window}*, found {len(matches)}: {matches}")
    return matches[0]


RESULTS_DIR = resolve_run_window_dir("results", RUN, WINDOW)
print("FINAL_SET:", FINAL_SET, "| run:", RUN, "| figures ->", OUT_DIR)

# %% [markdown]
# ## Shared colours
#
# Used by several figures. Everything else (titles, labels, sizes, other colours) is written in each figure's cell.

# %%
INK, MUTED = "#0b0b0b", "#8a8984"   # text / reference lines; secondary text
# method colours of the table charts and the PIOMAS scatter (validated categorical order: adjacent-pair colour-blind dE >= 8)
METHOD_COLORS = {
    "EngressNet": "#2a78d6",                   # 1 blue
    "Deterministic U-Net": "#eb6834",          # 2 orange
    "Bilinear": "#1baf7a",                     # 3 aqua
    "Seed/window ensemble (det.)": "#eda100",  # 4 yellow
    "MC-dropout U-Net": "#e87ba4",             # 5 magenta (final1 tables only)
}
MANGANESE = ncp.pal_sample("Manganese_r", 5)   # supplementary line charts: 0 truth, 1 EngressNet, 2 deterministic, 3 bilinear, 4 PIOMAS
SODIUM = ncp.pal_sample("Sodium", 8)           # Figs. 6, 7, 9 and CRPS by regime
MAGNESIUM = ncp.pal_sample("Magnesium", 8)     # Fig. 8
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# %% [markdown]
# ## Data: put the real deterministic U-Net into the aggregate files
#
# The EngressNet run's aggregate CSVs (`evaluation/saved_figs/<run>/_4window_aggregate/`) have a "deterministic" column,
# but it is EngressNet's own zero-noise pass. This cell copies the files the figures use to `figures/v2/data_<FINAL_SET>/`,
# replaces that column with the det-MSE U-Net run on the same days (`FS["MESA"]["det"]`), using the same pooling as
# `evaluation/build_4window_aggregate_figures.py` (mean over members and windows), and checks that the truth matches. The
# map cells read the det U-Net from `DET_FIELDS`.

# %%
DET_RUN = FS["MESA"]["det"]
DATA_DIR = os.path.join(V2, f"data_{FINAL_SET}", "MESA_aggregate")
os.makedirs(DATA_DIR, exist_ok=True)
ENG_AGG = os.path.join(BASE, "evaluation", "saved_figs", RUN, "_4window_aggregate")
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
# # Data loaders
#
# Functions that read the data the figures need (no plotting). Big inputs are read once and reused. Run these cells once.

# %%
POINT_HOPE = {"Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
_CACHE = {}


def cached(key, fn):
    """fn() on the first call for (FINAL_SET, key), the stored result afterwards."""
    if (FINAL_SET, key) not in _CACHE:
        _CACHE[(FINAL_SET, key)] = fn()
    return _CACHE[(FINAL_SET, key)]


# %%
def etopo_relief():
    """ETOPO2 subset around the domain (cached on disk), longitudes 0-360 so the dateline is interior -> (z, lon360, lat)."""
    etopo = "/glade/derecho/scratch/skygale/downscaling/ancillary/maps/etopo2_2006apr.nc"  # gunzip of d759003/etopo2_2006apr/etopo2_2006apr.nc.gz
    cache = "/glade/derecho/scratch/skygale/downscaling/ancillary/maps/etopo2_chukchi_subset.npz"
    if not os.path.exists(cache):
        d = xr.open_dataset(etopo)
        sub = d.z.sel(y=slice(48, 86))
        lon = sub.x.values % 360
        keep = (lon >= 145) & (lon <= 250)
        o = np.argsort(lon[keep])
        np.savez_compressed(cache, z=sub.values[:, keep][:, o].astype(np.float32), lon=lon[keep][o], lat=sub.y.values)
    c = np.load(cache)
    lon360, ui = np.unique(c["lon"], return_index=True)
    return c["z"][:, ui], lon360, c["lat"]


# %%
def fig05_data():
    """Fig. 5 samples: three test days (rng seed 0 on days with > 25 % of cells above 0.5 m, identical to the original), the
    fields of every method on those days, member 4, the det-MSE U-Net, and the difference colour limit (95th percentile of
    |method - truth| over ocean cells of every difference panel, rounded up to 0.05 m)."""
    def load():
        def run_dir(batch):
            m = [d for d in glob.glob(os.path.join(BASE, "results", batch, f"*{WINDOW}*")) if os.path.isdir(d)]
            assert len(m) == 1, m
            return m[0]
        edir = os.path.join(run_dir(FS["MESA"]["eng"]), "eval_data")
        meta = json.load(open(os.path.join(edir, "meta.json")))
        D = SimpleNamespace(bbox=meta["bbox"], points={**meta["candidate_points"], **POINT_HOPE},
                            tile_geometry=pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb")))
        z = np.load(os.path.join(edir, "fields.npz"))
        Y_test = z["Y_test_phys"]
        ice_frac = (Y_test[:, 0] > 0.5).mean(axis=(1, 2))
        idxs = np.random.default_rng(0).choice(np.where(ice_frac > 0.25)[0], 3, replace=False)
        D.F = {"X": z["X_test_sit_phys"][idxs], "bil": z["Y_base_phys"][idxs], "mem": z["preds_all_phys"][idxs, 4],
               "ens": z["Y_pred_phys"][idxs], "truth": Y_test[idxs], "ocean": z["mask_test"][idxs, 0] <= 0.5, "tile": z["test_tile_ids"][idxs]}
        del z
        zd = np.load(os.path.join(run_dir(FS["MESA"]["det"]), "eval_data", "fields.npz"))
        assert np.array_equal(zd["Y_test_phys"][idxs], D.F["truth"]), "det-MSE test set differs"
        D.F["det"] = zd["Y_pred_det_phys"][idxs]
        del zd
        st = pd.read_csv(os.path.join(edir, "sample_times.csv"))
        n_days = st["time"].nunique()
        D.labels = []
        for idx in idxs:
            d = pd.Timestamp(st.loc[idx, "time"])
            member = int(idx) // n_days  # samples are stored member-block by member-block (6 x 365 for MESACLIP)
            D.labels.append((d, member))
            print(f"Fig. 5 sample: index {idx}, date {d:%Y-%m-%d}, MESACLIP member index {member}, ice frac>0.5m {ice_frac[idx]:.2f}")
        D.proj, D.boundary, _ = rdeb.make_polar_proj(D.bbox)
        diffs = np.concatenate([np.abs(D.F[k][:, 0] - D.F["truth"][:, 0])[D.F["ocean"]] for k in ("bil", "det", "mem", "ens")])
        D.dlim = float(np.ceil(np.percentile(diffs, 95) / 0.05) * 0.05)
        print(f"diff colour limit: 95th pct |diff| = {np.percentile(diffs, 95):.3f} m -> +/-{D.dlim:.2f} m (max |diff| {diffs.max():.2f} m)")
        return D
    return cached("fig05", load)


def fig05_geo(D, r):
    """(coarse lon, coarse lat, target lon, target lat) of Fig. 5 sample r."""
    g = D.tile_geometry[int(D.F["tile"][r])]
    return g["context_lon"], g["context_lat"], g["target_lon"], g["target_lat"]


def fig05_map(D, ax, field, lon_, lat_, cmap, vmin, vmax, land_on_top):
    """One Fig. 5 map panel: field, polar frame, communities; land drawn over the field if land_on_top."""
    im = ax.pcolormesh(lon_, lat_, field, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    rdeb.style_polar_ax(ax, D.proj, D.boundary, D.bbox, D.points, lon_, lat_)
    if land_on_top:  # above the field (pcolormesh zorder 1) but below the community stars/labels (zorder 2)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1.5)
        ax.coastlines(resolution="50m", zorder=1.6)
    return im


# %%
def fig09_data():
    """Fig. 9: model fields for the maps and point series (PIOMAS 2021 input) and the ICESat-2 pass-cells with the matched
    model values (same QC as Table 5: extent >= 2 km, >= 20 segments, median segment uncertainty <= 1 m)."""
    def load():
        year, window = 2021, FS["piomas_window"]
        m = [d for d in sorted(glob.glob(os.path.join(BASE, "results", FS["piomas2010"], f"*_{window}_{year}_*")))
             if os.path.exists(os.path.join(d, "eval_data", "fields.npz"))]
        assert len(m) == 1, m
        edir = os.path.join(m[0], "eval_data")
        z = np.load(os.path.join(edir, "fields.npz"))
        D = SimpleNamespace(year=year, Y_pred=z["Y_pred_phys"][:, 0], members=z["preds_all_phys"][:, :, 0], ocean_hw=z["mask_test"][0, 0] <= 0.5)
        del z
        geo = pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb"))[0]
        D.times = pd.to_datetime(pd.read_csv(os.path.join(edir, "sample_times.csv"))["time"])
        D.months = D.times.dt.month.values
        D.points = pd.read_csv(os.path.join(edir, "candidate_point_timeseries.csv"))
        is2c = "/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells"
        D.pcd = pd.read_parquet(f"{is2c}/pass_cells.parquet").query("in_domain").reset_index(drop=True)
        D.qc = ((D.pcd.extent_km >= 2.0) & (D.pcd.n_seg >= 20) & (D.pcd.unc_med <= 1.0)).values
        D.E = np.load(f"{is2c}/eval/{FS['cells_eng']}_cells.npz")
        D.D = np.load(f"{is2c}/eval/{FS['cells_det']}_cells.npz")
        assert list(D.E["windows"]) == list(D.D["windows"]) and len(D.E["members"]) == len(D.pcd)
        D.Mm, D.Dm, D.Bl = D.E["members"].astype(float), D.D["members"].astype(float), D.E["bilinear"].astype(float)
        w_idx = [i for i, w in enumerate(D.E["windows"]) if str(w).startswith(window)][0]
        D.ens_win = D.E["ensmean"][:, w_idx].astype(float)  # the map's window, for the difference panels
        D.y = D.pcd.h_mean.values
        D.dates = pd.DatetimeIndex(D.pcd.date)
        D.plot_bbox = {"lon_min": -182, "lon_max": -151, "lat_min": 60, "lat_max": 75}
        D.proj, D.boundary, _ = fe.make_polar_proj(D.plot_bbox)
        D.lon2d, D.lat2d = np.meshgrid(np.asarray(geo["target_lon"]), np.asarray(geo["target_lat"]))
        D.in_year = D.qc & (D.dates.year == year) & np.isfinite(D.ens_win)
        D.diff = D.y - D.ens_win
        D.track_lon = ((D.pcd.lon.values + 180) % 360) - 180
        return D
    return cached("fig09", load)


def fig09_point_series(D, name, p_lo=16, p_hi=84):
    """Daily ensemble mean and the p_lo-p_hi percentile band of the members at a community's nearest cell."""
    row = D.points[(D.points.point == name) & (D.points.method == "stochastic_unet_mean")]
    iy, ix = int(row["grid_iy"].iloc[0]), int(row["grid_ix"].iloc[0])
    lo, hi = np.percentile(D.members[:, :, iy, ix], [p_lo, p_hi], axis=1)
    return D.Y_pred[:, iy, ix], lo, hi, D.members[:, :, iy, ix].std(axis=1)


def crps_fair(X, yy):
    K = X.shape[1]
    Xs = np.sort(X, 1)
    w = 2 * np.arange(1, K + 1) - K - 1
    return (np.abs(X - yy[:, None]).mean(1) - (Xs * w).sum(1) / (K * (K - 1))).mean()


# %%
def sensitivity_table(ds, groups, columns, direction):
    """Figs. A1/A2: rows for the heatmap from results/revisions_2026-11/sensitivity_final2/<ds>_{summary,runs}.csv
    (evaluation/revisions/sensitivity_final2_metrics.py) -> (labels, values, kinds, badness) or None if not there yet.
    kinds: "group" (header row), "row" (a configuration, mean over seeds), "band" (the final configuration's seed min-max).
    badness: 0 = best row, 1 = worst, per column (bias towards 0, spread-skill towards 1)."""
    sdir = os.path.join(BASE, "results", "revisions_2026-11", "sensitivity_final2")
    path = os.path.join(sdir, f"{ds}_summary.csv")
    if not os.path.exists(path):
        print("no sensitivity results yet:", path)
        return None
    summ = pd.read_csv(path, index_col=0)
    runs = pd.read_csv(os.path.join(sdir, f"{ds}_runs.csv"))
    labels, vals, kinds = [], [], []
    for group, items in groups:
        present = [(k, lab) for k, lab in items if k in summ.index]
        if not present:
            continue
        labels.append(group); vals.append([np.nan] * len(columns)); kinds.append("group")
        for k, lab in present:
            n = int(summ.loc[k, "seeds"])
            labels.append(f"{lab}" + (" (1 seed)" if n == 1 else f" ({n} seeds)"))
            vals.append([summ.loc[k, c] for c in columns]); kinds.append("row")
            if k == "final" and n > 1:
                r = runs[runs.config == "final"]
                labels.append(f"   seed band, {n} seeds (min-max)")
                vals.append([(r[c].min(), r[c].max()) for c in columns]); kinds.append("band")
    V = np.array([[np.nan if isinstance(v, tuple) else v for v in row] for row in vals], dtype=float)
    rows = np.array([k == "row" for k in kinds])
    bad = np.full(V.shape, np.nan)
    for j, c in enumerate(columns):
        x = V[rows, j]
        d = direction.get(c, "lower")
        b = -x if d == "higher" else np.abs(x) if d == "zero" else np.abs(x - 1) if d == "one" else x
        lo, hi = np.nanmin(b), np.nanmax(b)
        bad[rows, j] = (b - lo) / (hi - lo) if hi > lo else 0.0
    return labels, vals, kinds, bad


SENS_COLUMNS = ["MAE", "RMSE", "Bias", "Grad MAE", "Pattern Corr", "SSIM", "IIEE (1e-4 m)", "IIEE (0.15 m)", "Coastal MAE",
                "Coastal RMSE", "CRPS", "Energy score", "Spread-skill"]
SENS_DIRECTION = {"Bias": "zero", "Pattern Corr": "higher", "SSIM": "higher", "Spread-skill": "one"}  # others: lower is better
SENS_GROUPS = [  # (group, [(config key in the summary CSV, row label)]), top to bottom
    ("Final configuration", [("final", "repeated per-stage noise (final)")]),
    ("Data processing", [("interp", "interp coarsening (final: avg)"), ("land05", "land threshold 0.5 (final: 0.1)"),
                         ("final (land05 cells)", "final, scored on the land-0.5 ocean cells"),
                         ("construth", "conservative truth (own truth)"),
                         ("final (Mar-Jul)", "final, scored Mar-Jul"), ("season", "Mar-Jul only, scored Mar-Jul")]),
    ("Architecture", [("bigdomain", "larger domain (190-143W)"), ("depth5", "encoder depth +1"),
                      ("attn", "end-of-decoder attention")]),
    ("Noise mechanism", [("uncorr", "uncorrelated noise (1x1 kernel)"), ("globalz", "single global noise field"),
                         ("latefusion", "late mask fusion"), ("nch2", "2 noise channels (final: 1)"),
                         ("k3", "noise kernel 3 (final: 5)"), ("k9", "noise kernel 9")]),
    ("Calibration", [("tile30", "energy tile 30 (final: 10)"), ("tile15", "energy tile 15"), ("tile5", "energy tile 5"),
                     ("noampmap", "no state-dependent amplitude"), ("nlr1", "noise lr x1 (final: x10)"), ("noema", "no EMA")]),
]


# %%
def table_values():
    """Tables 2-4 perfect-model values (final configuration, ocean cells, coarse-zero rule) -> ({"MESACLIP": DataFrame,
    "FOSI": DataFrame} with one row per method, the methods shown in Table 2 (final1 also has MC-dropout)).
    Sources: evaluation/corrected_calibration/{MESA,FOSI}_*{,_ocean_metrics}.csv (det/bilinear CRPS = MAE)."""
    def load():
        C = f"{BASE}/evaluation/corrected_calibration"
        def perfect(ds):
            pre = {"MESACLIP": "MESA", "FOSI": "FOSI"}[ds]
            eng, det = FS[pre]["eng"], FS[pre]["det"]
            om_e = pd.read_csv(f"{C}/{eng}_ocean_metrics.csv"); om_d = pd.read_csv(f"{C}/{det}_ocean_metrics.csv")
            cal = pd.read_csv(f"{C}/{eng}.csv"); cal = cal[cal.run != "POOLED"]
            g = lambda df, m: df[(df.run == "MEAN over windows") & (df.method == m)].iloc[0]
            e, d, b = g(om_e, "EngressNet ensemble mean"), g(om_d, "Deterministic (zero-noise pass)"), g(om_e, "Bilinear")
            cols = ["RMSE", "MAE", "Pattern Corr", "SSIM", "Grad MAE", "IIEE", "Bias"]
            rows = {"EngressNet": {**e[cols].to_dict(), "CRPS": cal["CRPS (ocean)"].mean()},
                    "Deterministic U-Net": {**d[cols].to_dict(), "CRPS": d["MAE"]},
                    "Bilinear": {**b[cols].to_dict(), "CRPS": b["MAE"]}}
            return pd.DataFrame(rows).T
        P = {ds: perfect(ds) for ds in ("MESACLIP", "FOSI")}
        methods = ["EngressNet", "Deterministic U-Net", "Bilinear"]
        if FINAL_SET == "final1":  # 2026-11-14: MC-dropout baseline (E2; 4-window mean of seed 0) from the D-metric outputs
            DM = f"{BASE}/results/revisions_2026-11/D_metrics"
            keys = {"RMSE": "Ensemble-mean RMSE", "Pattern Corr": "Pattern Corr (ensemble mean / point value)", "CRPS": "CRPS (point mass: MAE)"}
            for ds, pre in (("MESACLIP", "MESA"), ("FOSI", "FOSI")):
                cols = [pd.read_csv(f"{DM}/{pre}_{w}_extra_E2.csv", index_col=0)["MC-dropout U-Net"] for w in ("2000-2005", "2005-2010", "2010-2015")]
                cols.append(pd.read_csv(f"{DM}/{pre}_2015-2020_extra_E.csv", index_col=0)["MC-dropout U-Net s0"])
                mc = pd.concat(cols, axis=1).apply(pd.to_numeric, errors="coerce").mean(axis=1)
                for k, dk in keys.items():
                    P[ds].loc["MC-dropout U-Net", k] = mc[dk]
            methods.append("MC-dropout U-Net")
        return P, methods
    return cached("tables", load)


# %%
def map_inputs():
    """Supplementary maps: the EngressNet run of RESULTS_DIR and the det-MSE U-Net (DET_FIELDS, Data cell)."""
    def load():
        edir = os.path.join(RESULTS_DIR, "eval_data")
        meta = json.load(open(os.path.join(edir, "meta.json")))
        M = SimpleNamespace(bbox=meta["bbox"], points={**meta["candidate_points"], **POINT_HOPE},
                            tile_geometry=pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb")))
        f = np.load(os.path.join(edir, "fields.npz"))
        M.bilinear, M.det, M.ens = f["Y_base_phys"], DET_FIELDS["Y_pred_det_phys"], f["Y_pred_phys"]  # det-MSE U-Net, not the zero-noise pass
        M.truth, M.mask, M.tile_ids = f["Y_test_phys"], f["mask_test"], f["test_tile_ids"]
        M.months = pd.read_csv(os.path.join(edir, "sample_times.csv"), parse_dates=["time"])["time"].dt.month.values
        M.proj, M.boundary, _ = rdeb.make_polar_proj(M.bbox)
        g = M.tile_geometry[0]
        M.lon, M.lat = g["target_lon"], g["target_lat"]
        M.land = M.mask[0, 0] > 0.5
        return M
    return cached("maps", load)


def polar_frame(M, ax, lon_=None, lat_=None):
    """Polar frame, coastlines, gridlines and community stars of the supplementary maps."""
    rdeb.style_polar_ax(ax, M.proj, M.boundary, M.bbox, M.points, lon_, lat_)


def rolled(df, cols, days):
    """Centered rolling mean over `days` (rdeb.ROLLING_DAYS_*), as the original supplementary figures."""
    df = df.copy()
    if days:
        win = rdeb.rolling_window_samples(df["time"], days)
        df[cols] = df[cols].rolling(win, center=True, min_periods=1).mean()
    return df


# %%
def seasonal_cycle_values():
    """Monthly means of the domain-mean SIT (2021, ocean cells; same pooling as the time series) plus two EngressNet spreads:
    engressnet_cell_std (domain-mean per-cell ensemble std) and engressnet_member_std (std of the members' domain means,
    < 1 mm). The first call computes them from fields.npz (~8 GB, a few minutes) and caches
    data_<FINAL_SET>/MESA_aggregate/seasonal_cycle_monthly.csv; delete that file to recompute."""
    path = os.path.join(DATA_DIR, "seasonal_cycle_monthly.csv")
    if os.path.exists(path) and "engressnet_cell_std" in pd.read_csv(path, nrows=1).columns:
        return pd.read_csv(path, index_col=0)
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
        Pm = fz["preds_all_phys"]  # (samples, members, 1, H, W)
        mem = pd.DataFrame(np.stack([dmean(Pm[:, k, 0]) for k in range(Pm.shape[1])], axis=1), index=times).groupby(level=0).mean()
        dom_std.append(mem.groupby(mem.index.month).mean().std(axis=1, ddof=1))  # spread of the members' monthly domain means
        cs = np.concatenate([dmean(Pm[i:i + 100, :, 0].std(axis=1, ddof=1)) for i in range(0, Pm.shape[0], 100)])  # per-cell spread
        cs = pd.Series(cs, index=times).groupby(level=0).mean()
        cell_std.append(cs.groupby(cs.index.month).mean())
        del Pm, fz
    clim = ts.assign(month=ts["time"].dt.month).groupby("month")[["truth", "stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"]].mean()
    clim["engressnet_member_std"] = pd.concat(dom_std, axis=1).mean(axis=1)
    clim["engressnet_cell_std"] = pd.concat(cell_std, axis=1).mean(axis=1)
    clim.to_csv(path)
    return clim


# %%
def subgrid_values(nmin=6):
    """Sub-grid variability bar: pooled within-1 deg standard deviation of 0.1 deg thickness, every source sampled at the
    same QC'd ICESat-2 pass-cells on the same day (Oct 2018 - Dec 2021, groups of >= nmin pass-cells per (day, 1 deg cell));
    ICESat-2 measurement-noise variance removed. Also writes observations/data/alongtrack_eval/cice_ceiling_bar<SFX>.csv
    (used by the Table 5 chart). Returns {bar name: std (m)}, first entry ICESat-2."""
    import calibration_options_compare as c  # noqa: E402  (shared ICESat-2 pass-cell tools: QC masks, noise model)

    def pooled_std(F, groups, dof):
        X = F[:, None] if F.ndim == 1 else F
        v = np.zeros(X.shape[1])
        for g in groups:
            v += (len(g) - 1) * X[g].var(axis=0, ddof=1)
        return float(np.sqrt(np.mean(v / dof)))

    ev = c.EV2
    ct, cb = np.load(f"{ev}/cice_truth_cells.npz"), np.load(f"{ev}/cice_truth_bins.npz")
    eng, engb = np.load(f"{ev}/{FS['cells_eng']}_cells.npz"), np.load(f"{ev}/{FS['cells_eng']}_bins.npz")
    det = np.load(f"{ev}/{FS['cells_det']}_cells.npz")
    F = {"CICE (FOSI)\ntruth": ct["fosi"].astype(float), "CICE (MESACLIP)\ntruth": ct["mesa"].astype(float),
         "EngressNet (final)\non PIOMAS": eng["members"].astype(float), "Deterministic U-Net\n(det-MSE) on PIOMAS": det["ensmean"].astype(float),
         "PIOMAS\n(bilinear)": eng["bilinear"].astype(float)}
    fin = lambda X: np.isfinite(X).all(1) if X.ndim == 2 else np.isfinite(X)
    y = c.PCD.h_mean.values.astype(float)
    ok = c.qcC & (c.PCD.date.values <= np.datetime64("2021-12-31")) & np.isfinite(y) & np.all([fin(X) for X in F.values()], 0)
    FB = [cb["fosi"], cb["mesa"], engb["members"]]  # ICESat-2 measurement-noise model sigma(h), from the along-track 10 km bins
    okb = c.qcB & (c.BIN.date.values <= np.datetime64("2021-12-31")) & np.all([fin(X.astype(float)) for X in FB], 0)
    sig_fn = c.noise_fn(c.BIN.h_mean.values.astype(float), c.contiguous_windows(okb))
    d = c.PCD.loc[ok, ["date", "clat", "clon"]].copy()
    d["row"] = np.where(ok)[0]
    groups = [g["row"].values for _, g in d.groupby(["date", "clat", "clon"]) if len(g) >= nmin]
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
        f"{BASE}/observations/data/alongtrack_eval/cice_ceiling_bar{SFX}.csv")
    return vals

# %% [markdown]
# # Paper figures

# %% [markdown]
# ## Fig. 2: domain overview map
#
# Polar stereographic frame of the original figure, ETOPO2 relief on land (NOAA, GDEX d759003), the 0.1° training domain,
# sea and region names, an Alaska locator inset and a 200 km scale bar. Output: `fig02_domain_overview.png`

# %%
edir = os.path.join(glob.glob(f"{BASE}/results/MESA_final_ema/*2015-2020*")[0], "eval_data")
meta = json.load(open(os.path.join(edir, "meta.json")))
bbox = {**meta["bbox"], "lat_min": 54.0}  # southern edge extended 60 -> 54 N to show more of the Bering Sea
candidate_points = {**meta["candidate_points"], **POINT_HOPE}
geo0 = pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb"))[0]

# relief: land colours low -> high, hill shading, light-blue sea
z, lon360, lat = etopo_relief()
land_cmap = LinearSegmentedColormap.from_list("land", ["#c5d6a4", "#d9e0bb", "#e8e6d6", "#f5f4ef"])
dy = 111e3 * 2 / 60
rgb_land = LightSource(azdeg=315, altdeg=45).shade(np.clip(z, 0, 1800), cmap=land_cmap, vmin=0, vmax=1800, blend_mode="soft",
                                                   vert_exag=6, dx=dy * np.cos(np.deg2rad(68)), dy=dy)[..., :3]
sea = np.array([0.80, 0.90, 0.96])
rgb = np.where((z > 0)[..., None], rgb_land, sea)

proj, boundary_path, _ = rdeb.make_polar_proj(bbox)
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

# training domain outline
lon_t, lat_t = geo0["target_lon"], geo0["target_lat"]
x0, x1, y0, y1 = float(np.min(lon_t)), float(np.max(lon_t)), float(np.min(lat_t)), float(np.max(lat_t))
t = np.linspace(0, 1, 100)
box_lon = np.r_[x0 + (x1 - x0) * t, np.full(100, x1), x1 - (x1 - x0) * t, np.full(100, x0)]
box_lat = np.r_[np.full(100, y0), y0 + (y1 - y0) * t, np.full(100, y1), y1 - (y1 - y0) * t]
ax.plot(box_lon, box_lat, color="tab:blue", linewidth=1.5, transform=ccrs.PlateCarree())
ax.text(-166.5, 73.6, "Model training domain", color="tab:blue", fontsize=9, ha="center", va="center", transform=ccrs.PlateCarree())

# place names: (text, lon, lat, style)
for txt, lo, la, kw in [("Chukchi Sea", -169.5, 71.0, dict(style="italic", color="#2b5d8c", fontsize=10)),
                        ("Bering Sea", -175.0, 58.5, dict(style="italic", color="#2b5d8c", fontsize=10)),
                        ("Arctic Ocean", -160.0, 76.5, dict(style="italic", color="#2b5d8c", fontsize=10)),
                        ("Alaska", -153.0, 66.0, dict(color="#4d4d4d", fontsize=11)),
                        ("Russia", -186.0, 66.2, dict(color="#4d4d4d", fontsize=11))]:
    ax.text(lo, la, txt, ha="center", va="center", transform=ccrs.PlateCarree(), **kw)

# 200 km scale bar near the bottom of the domain
xa, ya = proj.transform_point(-187.0, 56.5, ccrs.PlateCarree())
ax.plot([xa, xa + 200e3], [ya, ya], color="k", lw=1.5, transform=proj)
ax.text(xa + 100e3, ya + 25e3, "200 km", ha="center", va="bottom", fontsize=8, transform=proj)

# Alaska locator inset in the empty upper-left corner of the rounded frame
iax = fig.add_axes([0.06, 0.70, 0.24, 0.20], projection=ccrs.LambertConformal(central_longitude=-166.5, standard_parallels=(60, 75)))
iax.set_extent([-206, -127, 52, 84], crs=ccrs.PlateCarree())
iax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "110m"), facecolor="white", edgecolor="none")
iax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"), facecolor="lightgray", edgecolor="0.4", lw=0.4)
iax.plot(box_lon, box_lat, color="k", lw=1.1, transform=ccrs.PlateCarree())

fig.savefig(out("fig02_domain_overview.png"), dpi=300, bbox_inches="tight")
# fig.savefig(out("fig02_domain_overview.pdf"), bbox_inches="tight")   # vector copy
plt.close(fig)
show("fig02_domain_overview.png")

# %% [markdown]
# ## Fig. 3: domain and resolution
#
# One test day of the low-resolution input next to the high-resolution truth. Output: `fig03_domain_resolution.png`

# %%
edir = os.path.join(RESULTS_DIR, "eval_data")
bbox = json.load(open(os.path.join(edir, "meta.json")))["bbox"]
geo0 = pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb"))[0]
fields = np.load(os.path.join(edir, "fields.npz"))
SAMPLE_IDX = 150  # which test day is shown
panels = [("Low Resolution (X)", fields["X_test_sit_phys"][SAMPLE_IDX, 0], geo0["context_lon"], geo0["context_lat"]),
          ("High Resolution (Y)", fields["Y_test_phys"][SAMPLE_IDX, 0], geo0["target_lon"], geo0["target_lat"])]

proj, boundary_path, _ = rdeb.make_polar_proj(bbox)
fig, axs = plt.subplots(1, 2, figsize=(10, 6), subplot_kw={"projection": proj})
for ax, (title, field, lon_, lat_) in zip(axs, panels):
    im = ax.pcolormesh(lon_, lat_, field, transform=ccrs.PlateCarree(), cmap=cmocean.cm.ice_r, vmin=0, vmax=3, shading="auto")
    rdeb.style_polar_ax(ax, proj, boundary_path, bbox, {}, lon_, lat_, points={})
    ax.set_title(title, fontsize=14)
fig.colorbar(im, ax=list(axs), shrink=0.7, aspect=15, pad=0.02, label="SIT (m)")
fig.savefig(out("fig03_domain_resolution.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig03_domain_resolution.png")

# %% [markdown]
# ## Fig. 5 (paper): example fields with difference rows
#
# Three test days (rng seed 0 on days with > 25 % of cells above 0.5 m), member 4, deterministic = det-MSE U-Net. Each day's
# field row is followed by its method − truth row. The difference colour limit is the 95th percentile of |method − truth|.
# Output: `fig05_example_fields.png` (saving is commented out, as you left it on 2026-11-16).

# %%
D = fig05_data()
COLS = [("Low-Res Input", "X"), ("Bilinear", "bil"), ("Deterministic", "det"), ("One Member", "mem"),
        ("Ensemble Mean", "ens"), ("High-Res Truth", "truth")]
DIFF_KEYS = ["bil", "det", "mem", "ens"]

fig = plt.figure(figsize=(18, 19.8), constrained_layout=True)
gs = fig.add_gridspec(7, 6, height_ratios=[1] * 6 + [0.07])  # last row: the two horizontal colourbars
for r in range(3):
    clon, clat, tlon, tlat = fig05_geo(D, r)
    for c, (title, k) in enumerate(COLS):
        ax = fig.add_subplot(gs[2 * r, c], projection=D.proj)
        im_f = fig05_map(D, ax, D.F[k][r, 0], clon if k == "X" else tlon, clat if k == "X" else tlat, cmocean.cm.ice_r, 0, 2, True)
        if r == 0:
            ax.set_title(title, fontsize=18)
        if c == 0:
            d, m = D.labels[r]
            ax.text(-0.08, 0.5, f"{d:%-d %b %Y}\nmember {m}", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=13)
        if k in DIFF_KEYS:
            axd = fig.add_subplot(gs[2 * r + 1, c], projection=D.proj)
            im_d = fig05_map(D, axd, np.where(D.F["ocean"][r], D.F[k][r, 0] - D.F["truth"][r, 0], np.nan), tlon, tlat,
                             "RdBu_r", -D.dlim, D.dlim, False)
            axd.set_title("− Truth", fontsize=12)
cb = fig.colorbar(im_f, cax=fig.add_subplot(gs[6, 0:3]), orientation="horizontal", extend="max")
cb.set_label("Sea ice thickness (m)", fontsize=16)
cb = fig.colorbar(im_d, cax=fig.add_subplot(gs[6, 3:6]), orientation="horizontal", extend="both")
cb.set_label("Method − truth (m)", fontsize=16)
# fig.savefig(out("fig05_example_fields.png"), dpi=250, bbox_inches="tight")
plt.close(fig)
show("fig05_example_fields.png")

# %% [markdown]
# ### Fig. 5 alternative: fields only (ice_r)
#
# Output: `fig05_example_fields_alt_ice_r_nodiff.png` (saving commented out)

# %%
D = fig05_data()
COLS = [("Low-Res Input", "X"), ("Bilinear", "bil"), ("Deterministic", "det"), ("One Member", "mem"),
        ("Ensemble Mean", "ens"), ("High-Res Truth", "truth")]
fig, axs = plt.subplots(3, 6, figsize=(18, 9.6), constrained_layout=True, subplot_kw={"projection": D.proj})
for r in range(3):
    clon, clat, tlon, tlat = fig05_geo(D, r)
    for c, (title, k) in enumerate(COLS):
        im = fig05_map(D, axs[r, c], D.F[k][r, 0], clon if k == "X" else tlon, clat if k == "X" else tlat, cmocean.cm.ice_r, 0, 2, True)
        if r == 0:
            axs[r, c].set_title(title, fontsize=18)
    d, m = D.labels[r]
    axs[r, 0].text(-0.08, 0.5, f"{d:%-d %b %Y}\nmember {m}", transform=axs[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=13)
cb = fig.colorbar(im, ax=axs, aspect=30, shrink=0.9, pad=0.02, extend="max")
cb.set_label("Sea ice thickness (m)", fontsize=18)
cb.ax.tick_params(labelsize=14)
# fig.savefig(out("fig05_example_fields_alt_ice_r_nodiff.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Fig. 5 alternative: colormap comparison
#
# Sample 1 in Blues / ice_r / viridis. Output: `fig05_example_fields_cmap_compare.png` (saving commented out)

# %%
D = fig05_data()
COLS = [("Low-Res Input", "X"), ("Bilinear", "bil"), ("Deterministic", "det"), ("One Member", "mem"),
        ("Ensemble Mean", "ens"), ("High-Res Truth", "truth")]
fig, axs = plt.subplots(3, 6, figsize=(18, 9.6), constrained_layout=True, subplot_kw={"projection": D.proj})
clon, clat, tlon, tlat = fig05_geo(D, 0)
for r, (cname, cmap, land_on_top) in enumerate((("Blues", "Blues", False), ("cmocean ice_r", cmocean.cm.ice_r, True),
                                                ("viridis", "viridis", True))):
    for c, (title, k) in enumerate(COLS):
        im = fig05_map(D, axs[r, c], D.F[k][0, 0], clon if k == "X" else tlon, clat if k == "X" else tlat, cmap, 0, 2, land_on_top)
        if r == 0:
            axs[r, c].set_title(title, fontsize=18)
    axs[r, 0].text(-0.08, 0.5, cname, transform=axs[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=14)
    cb = fig.colorbar(im, ax=axs[r, :], aspect=12, shrink=0.9, pad=0.01, extend="max")
    cb.set_label("SIT (m)", fontsize=12)
fig.suptitle(f"Colormap comparison, {D.labels[0][0]:%-d %b %Y} (member {D.labels[0][1]}), 0-2 m", fontsize=16)
# fig.savefig(out("fig05_example_fields_cmap_compare.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Fig. 5 alternative: differences as a separate figure
#
# Output: `fig05_example_diffs_alt_separate.png` (saving commented out)

# %%
D = fig05_data()
DIFF_TITLES = {"bil": "Bilinear − Truth", "det": "Deterministic − Truth", "mem": "One Member − Truth", "ens": "Ensemble Mean − Truth"}
fig, axs = plt.subplots(3, 4, figsize=(12.5, 9.6), constrained_layout=True, subplot_kw={"projection": D.proj})
for r in range(3):
    _, _, tlon, tlat = fig05_geo(D, r)
    for c, (k, title) in enumerate(DIFF_TITLES.items()):
        im = fig05_map(D, axs[r, c], np.where(D.F["ocean"][r], D.F[k][r, 0] - D.F["truth"][r, 0], np.nan), tlon, tlat,
                       "RdBu_r", -D.dlim, D.dlim, False)
        if r == 0:
            axs[r, c].set_title(title, fontsize=16)
    d, m = D.labels[r]
    axs[r, 0].text(-0.08, 0.5, f"{d:%-d %b %Y}\nmember {m}", transform=axs[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=13)
cb = fig.colorbar(im, ax=axs, aspect=30, shrink=0.9, pad=0.02, extend="both")
cb.set_label("Difference from high-res truth (m)", fontsize=16)
cb.ax.tick_params(labelsize=13)
# fig.savefig(out("fig05_example_diffs_alt_separate.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Fig. 6: rank histogram
#
# Output: `fig06_rank_histogram.png`

# %%
rank_hist_df = pd.read_csv(os.path.join(AGG_DIR, "14_rank_histogram_data_4window.csv"), index_col=0)
rank_regime_df = pd.read_csv(os.path.join(AGG_DIR, "14b_rank_histogram_by_regime_data_4window.csv"), index_col=0)
K = len(rank_hist_df) - 1
expected_freq = rank_hist_df["expected_frequency"].iloc[0]
regime_colors = [SODIUM[0], SODIUM[3], SODIUM[5], SODIUM[6]]   # open water, thin, moderate, thick
regime_short = {"Open water/frazil (0-0.15 m)": "Open Water/Frazil", "Thin ice (0.15-0.5 m)": "Thin Ice",
                "Moderate ice (0.5-1.0 m)": "Moderate Ice", "Thick ice (>1.0 m)": "Thick Ice"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.bar(rank_hist_df["rank"], rank_hist_df["frequency"], width=0.85, color=SODIUM[3], alpha=0.85, label="Observed Frequency")
ax1.axhline(expected_freq, color="black", linestyle="--", linewidth=2, label="Ideal Uniform Frequency")
ax1.set_xlabel(f"Rank of Truth Among {K} Ensemble Members", fontsize=14)
ax1.set_ylabel("Frequency", fontsize=14)
ax1.set_title("(A) Rank Histogram, All Ocean Cells", fontsize=16)
ax1.tick_params(axis="both", labelsize=14)
ax1.legend(fontsize=11, frameon=False)

total = rank_regime_df["count"].sum()
for (label, grp), color in zip(rank_regime_df.groupby("SIT regime", sort=False), regime_colors):
    share = grp["count"].sum() / total
    ax2.plot(grp["rank"], grp["relative_frequency"], marker="o", markersize=6, linewidth=2, color=color,
             label=f"{regime_short.get(label, label)} ({share:.0%} of cells)")
ax2.axhline(1.0, color="black", linestyle="--", linewidth=2, label="Ideal Uniform Frequency")
ax2.set_xlabel(f"Rank of Truth Among {K} Ensemble Members", fontsize=14)
ax2.set_ylabel("Frequency / Ideal Frequency", fontsize=14)
ax2.tick_params(axis="both", labelsize=14)
ax2.set_title("(B) Rank Histogram by SIT Regime", fontsize=16)
ax2.legend(fontsize=11, frameon=False, loc="upper center")
fig.tight_layout()
fig.savefig(out("fig06_rank_histogram.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig06_rank_histogram.png")

# %% [markdown]
# ## Fig. 7: spread vs error by SIT regime
#
# Output: `fig07_spread_skill_regime.png`

# %%
ss = pd.read_csv(os.path.join(AGG_DIR, "15_spread_skill_by_regime_data_4window.csv"), index_col=0)
ss = ss[ss["SIT regime"] != "All ocean cells"]
bins = ss["SIT regime"].tolist()
x = np.arange(len(bins))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
width = 0.35
ax1.bar(x - width / 2, ss["RMS spread, (K+1)/K-corrected (m)"], width, color=SODIUM[3], label=r"$\sqrt{\frac{K+1}{K}\,\overline{s^2}}$")
ax1.bar(x + width / 2, ss["RMSE (m)"], width, color=SODIUM[0], label="Ensemble-Mean RMSE")
ax1.set_xticks(x)
ax1.set_xticklabels(bins, rotation=45, ha="right")
ax1.tick_params(axis="both", labelsize=12)
ax1.set_ylabel("SIT (m)", fontsize=14)
ax1.set_title("(A) Spread vs. Error by SIT Regime", fontsize=16)
ax1.legend(fontsize=14, frameon=False, loc="upper left")

ax2.bar(x, ss["Spread-skill ratio (1.0 = calibrated)"], color=SODIUM[5], alpha=0.85)
ax2.axhline(1.0, color="black", linestyle="--", linewidth=2, label="Perfect Calibration")
ax2.set_xticks(x)
ax2.set_xticklabels(bins, rotation=45, ha="right")
ax2.tick_params(axis="both", labelsize=12)
ax2.set_ylabel("Spread-Skill Ratio\n(<1 under-dispersed, >1 over-dispersed)", fontsize=14)
ax2.set_title("(B) Spread-Skill Ratio by SIT Regime", fontsize=16)
ax2.legend(fontsize=14, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(out("fig07_spread_skill_regime.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig07_spread_skill_regime.png")

# %% [markdown]
# ## Fig. 8: observed seasonal cycle (PIOMAS vs FOSI / MESACLIP truth; model-independent)
#
# Output: `fig08_obs_seasonal_cycle.png`

# %%
seasonal_df = pd.read_csv(os.path.join(BASE, "figures", "piomas_fosi_mesa_seasonal_cycle_2021.csv"), index_col=0)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(seasonal_df.index, seasonal_df["PIOMAS (obs)"], color=MANGANESE[4], linewidth=2, linestyle="--", label="PIOMAS")
ax.plot(seasonal_df.index, seasonal_df["FOSI truth"], color=MAGNESIUM[6], linewidth=2, label="FOSI")
ax.plot(seasonal_df.index, seasonal_df["MESA truth"], color=MAGNESIUM[4], linewidth=2, label="MESACLIP")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_ABBR)
ax.set_ylabel("Domain-mean SIT (m)")
ax.set_xlim(1, 12)
ax.set_ylim(0, 1.91)
ax.grid(alpha=0.2)
ax.legend(fontsize=11, loc="upper right")
fig.tight_layout()
fig.savefig(out("fig08_obs_seasonal_cycle.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig08_obs_seasonal_cycle.png")

# %% [markdown]
# ## Fig. 9 (paper): PIOMAS application
#
# FOSI-trained final model on PIOMAS input (2021). Top: ICESat-2 pass-cell mean minus EngressNet ensemble mean at the same
# cell and day, over a light grey ensemble-mean background (difference colour limit: 95th percentile of |difference| in the
# shown months). Bottom: daily point series. The Fig. 9 data load on the first Fig. 9 cell you run (about a minute).
# Output: `fig09_piomas_application.png`

# %%
D = fig09_data()
MONTHS = {1: "January", 4: "April", 7: "July", 10: "October"}
DLIM = float(np.ceil(np.nanpercentile(np.abs(D.diff[D.in_year & np.isin(D.dates.month, list(MONTHS))]), 95) / 0.25) * 0.25)
ph_lon = ((rdeb.POINT_HOPE_LON_360 + 180) % 360) - 180

fig = plt.figure(figsize=(16, 12.6))
axes = []
for i, mo in enumerate(MONTHS):
    ax = fig.add_subplot(2, 4, i + 1, projection=D.proj)
    axes.append(ax)
    fe.style_polar_ax(ax, D.proj, D.boundary, D.plot_bbox)
    ax.plot(ph_lon, rdeb.POINT_HOPE_LAT, marker="*", color="red", markersize=10, transform=ccrs.PlateCarree())
    ax.text(ph_lon + 1, rdeb.POINT_HOPE_LAT + 0.35, "Point Hope", color="red", fontsize=7, transform=ccrs.PlateCarree())
    mm = np.where(D.ocean_hw, D.Y_pred[D.months == mo].mean(axis=0), np.nan)
    bg = ax.pcolormesh(D.lon2d, D.lat2d, mm, transform=ccrs.PlateCarree(), cmap="Greys", vmin=0, vmax=5, shading="auto", alpha=0.6)
    s = D.in_year & (D.dates.month == mo)
    if s.any():
        sc = ax.scatter(D.track_lon[s], D.pcd.lat.values[s], c=D.diff[s], cmap="RdBu_r", vmin=-DLIM, vmax=DLIM, s=3, linewidths=0,
                        transform=ccrs.PlateCarree(), zorder=5, rasterized=True)
        ax.set_title(f"{MONTHS[mo]} {D.year}\nmean {np.mean(D.diff[s]):+.2f} m, N = {s.sum():,}", fontsize=13)
    else:
        ax.set_title(f"{MONTHS[mo]} {D.year}\n(no ICESat-2 data)", fontsize=13)
cb = fig.colorbar(sc, ax=axes, orientation="horizontal", aspect=30, shrink=0.9, pad=0.02, extend="both")
cb.set_label("ICESat-2 pass-cell mean − EngressNet ensemble mean (m)", fontsize=15)
cb.ax.tick_params(labelsize=13)
cb2 = fig.colorbar(bg, ax=axes, orientation="horizontal", aspect=60, shrink=0.45, pad=0.01)
cb2.set_label("Background: EngressNet ensemble-mean SIT (m)", fontsize=11)
cb2.ax.tick_params(labelsize=10)
for j, name in enumerate(["Kivalina", "Point Hope"]):   # bottom row: point series
    ax = fig.add_subplot(2, 2, 3 + j)
    mean, lo, hi, _ = fig09_point_series(D, name, 16, 84)
    ax.fill_between(D.times.values, lo, hi, alpha=0.3, color=SODIUM[1], label="16–84th Percentile of Members")
    ax.plot(D.times.values, mean, color=SODIUM[0], linewidth=2, label="Ensemble Mean")
    ax.set_title(f"{name} Daily Downscaled PIOMAS SIT ({D.year})", fontsize=16)
    ax.set_ylabel("SIT (m)", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=14)
    ax.tick_params(axis="x", rotation=30)
fig.savefig(out("fig09_piomas_application.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"diff limit +/-{DLIM} m (95th pct |diff|, shown months)")
show("fig09_piomas_application.png")

# %% [markdown]
# ### Fig. 9 supplement: ICESat-2 tracks on the thickness scale
#
# Output: `fig09_piomas_application_supp_original_tracks.png`

# %%
D = fig09_data()
MONTHS = {1: "January", 4: "April", 7: "July", 10: "October"}
ph_lon = ((rdeb.POINT_HOPE_LON_360 + 180) % 360) - 180

fig = plt.figure(figsize=(16, 12))
axes = []
for i, mo in enumerate(MONTHS):
    ax = fig.add_subplot(2, 4, i + 1, projection=D.proj)
    axes.append(ax)
    fe.style_polar_ax(ax, D.proj, D.boundary, D.plot_bbox)
    ax.plot(ph_lon, rdeb.POINT_HOPE_LAT, marker="*", color="red", markersize=10, transform=ccrs.PlateCarree())
    ax.text(ph_lon + 1, rdeb.POINT_HOPE_LAT + 0.35, "Point Hope", color="red", fontsize=7, transform=ccrs.PlateCarree())
    mm = np.where(D.ocean_hw, D.Y_pred[D.months == mo].mean(axis=0), np.nan)
    pc = ax.pcolormesh(D.lon2d, D.lat2d, mm, transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=2.5, shading="auto")
    s = D.in_year & (D.dates.month == mo)
    if s.any():
        ax.scatter(D.track_lon[s], D.pcd.lat.values[s], c=D.y[s], cmap="Blues", vmin=0, vmax=2.5, s=2.5, linewidths=0.15,
                   edgecolors="#3a3a3a", transform=ccrs.PlateCarree(), zorder=5, rasterized=True)
    ax.set_title(f"{MONTHS[mo]} {D.year}", fontsize=14)
cb = fig.colorbar(pc, ax=axes, orientation="horizontal", aspect=30, shrink=0.9, pad=0.02)
cb.set_label("Ensemble Mean SIT (m)", fontsize=16)
cb.ax.tick_params(labelsize=14)
for j, name in enumerate(["Kivalina", "Point Hope"]):   # bottom row: point series
    ax = fig.add_subplot(2, 2, 3 + j)
    mean, lo, hi, sd = fig09_point_series(D, name, 16, 84)
    ax.fill_between(D.times.values, lo, hi, alpha=0.3, color=SODIUM[1], label="16–84th Percentile of Members")
    ax.plot(D.times.values, mean, color=SODIUM[0], linewidth=2, label="Ensemble Mean")
    ax.set_title(f"{name} Daily Downscaled PIOMAS SIT ({D.year})", fontsize=16)
    ax.set_ylabel("SIT (m)", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=14)
    ax.tick_params(axis="x", rotation=30)
    print(f"  {name}: days with mean-1std < 0: {(mean - sd < 0).sum()}; days with 16th pct < 0: {(lo < 0).sum()}")
fig.savefig(out("fig09_piomas_application_supp_original_tracks.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig09_piomas_application_supp_original_tracks.png")

# %% [markdown]
# ### Fig. 9 supplement: ICESat-2 vs downscaled SIT (hexbin)
#
# All QC'd pass-cells, Oct 2018 – Apr 2022. Output: `fig09b_supp_is2_scatter.png`

# %%
D = fig09_data()
ok = D.qc & np.isfinite(D.Mm).all(1) & np.isfinite(D.Dm).all(1) & np.isfinite(D.Bl)
est = {"Bilinear": D.Bl, "Deterministic U-Net": D.Dm.mean(1), "EngressNet": D.Mm.mean(1)}
NU, K = len(D.E["windows"]), int(D.E["K"])  # training units (4 windows for final1, 1 model for final2)
TITLES = {"Bilinear": "Bilinear PIOMAS",
          "Deterministic U-Net": "Deterministic U-Net" + (f" ({NU}-window mean)" if NU > 1 else ""),
          "EngressNet": f"EngressNet ensemble mean ({NU} × {K})" if NU > 1 else f"EngressNet ensemble mean ({K} members)"}
yy = D.y[ok]

fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True, constrained_layout=True)
stats = {}
for ax, (k, x) in zip(axs, est.items()):
    xx = x[ok]
    r = xx - yy
    stats[k] = dict(bias=r.mean(), rmse=np.sqrt((r ** 2).mean()), mae=np.abs(r).mean(), r=np.corrcoef(xx, yy)[0, 1])
    hb = ax.hexbin(xx, yy, gridsize=50, extent=(0, 5, 0, 5), mincnt=1, norm=LogNorm(vmin=1, vmax=2000), cmap="Greys")
    ax.plot([0, 5], [0, 5], color=METHOD_COLORS[k], lw=1.4, label="1:1")
    ax.set_title(TITLES[k], fontsize=12, color=INK)
    ax.text(0.03, 0.97, f"bias {stats[k]['bias']:+.3f} m\nRMSE {stats[k]['rmse']:.3f} m\nr {stats[k]['r']:.3f}",
            transform=ax.transAxes, va="top", fontsize=11, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlabel("Downscaled SIT (m)", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
axs[0].set_ylabel("ICESat-2 pass-cell mean (m)", fontsize=12)
fig.colorbar(hb, ax=axs, shrink=0.85, pad=0.01).set_label("Pass-cells per bin", fontsize=12)
fig.suptitle(f"All QC'd ICESat-2 0.1° pass-cells, Oct 2018 – Apr 2022 (N = {ok.sum():,}), PIOMAS input", fontsize=13, y=1.06)
fig.savefig(out("fig09b_supp_is2_scatter.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

crps_eng = crps_fair(D.Mm[ok], yy)
KD = int(D.D["K"])
crps_det4 = crps_fair(D.Dm[ok][:, 0::KD], yy) if NU > 1 else float("nan")  # distinct windows as an ensemble (final1 only)
print(f"N = {ok.sum()}")
for k, s_ in stats.items():
    print(f"  {k:22s} bias {s_['bias']:+.4f}  RMSE {s_['rmse']:.4f}  MAE {s_['mae']:.4f}  r {s_['r']:.4f}")
print(f"  EngressNet fair CRPS ({D.Mm.shape[1]} members) {crps_eng:.4f}; det-MSE as a {NU}-window ensemble, fair CRPS {crps_det4:.4f}; "
      f"det MAE {stats['Deterministic U-Net']['mae']:.4f}")
show("fig09b_supp_is2_scatter.png")

# %% [markdown]
# ## Fig. A1: sensitivity tests, MESACLIP (final design)
#
# One change at a time from the final configuration (final round, 2026-11); values = means over seeds; colour = badness
# within each column (0 = best row, 1 = worst; bias towards 0, spread-skill towards 1). The grey row is the final
# configuration's seed band (min-max). Rows and columns: `SENS_GROUPS`, `SENS_COLUMNS` (Data loaders).
# Output: `figA1_mesa_sensitivity.png`

# %%
res = sensitivity_table("MESA", SENS_GROUPS, SENS_COLUMNS, SENS_DIRECTION)
if res:
    labels, vals, kinds, bad = res
    fig, ax = plt.subplots(figsize=(1.05 * len(SENS_COLUMNS) + 3, 0.42 * len(labels) + 1.2))
    ax.imshow(np.ma.masked_invalid(bad), cmap="Reds", vmin=0, vmax=1, aspect="auto")
    for i, (row, kind) in enumerate(zip(vals, kinds)):
        if kind == "group":
            ax.axhspan(i - 0.5, i + 0.5, color="#f1f0ec", zorder=0)
            continue
        if kind == "band":
            ax.axhspan(i - 0.5, i + 0.5, color="#e4e3de", zorder=0)
        for j, v in enumerate(row):
            if isinstance(v, tuple):
                ax.text(j, i, f"{v[0]:.3f}-{v[1]:.3f}", ha="center", va="center", fontsize=6.5, color="#3d3d3a")
            elif np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if bad[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(SENS_COLUMNS))); ax.set_xticklabels(SENS_COLUMNS, rotation=40, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i, kind in enumerate(kinds):
        if kind == "group":
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_title("Sensitivity tests, MESACLIP (final design; one change at a time)", fontsize=13)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(SENS_COLUMNS), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(out("figA1_mesa_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    show("figA1_mesa_sensitivity.png")

# %% [markdown]
# ## Fig. A2: sensitivity tests, FOSI (final design)
#
# Output: `figA2_fosi_sensitivity.png`

# %%
res = sensitivity_table("FOSI", SENS_GROUPS, SENS_COLUMNS, SENS_DIRECTION)
if res:
    labels, vals, kinds, bad = res
    fig, ax = plt.subplots(figsize=(1.05 * len(SENS_COLUMNS) + 3, 0.42 * len(labels) + 1.2))
    ax.imshow(np.ma.masked_invalid(bad), cmap="Reds", vmin=0, vmax=1, aspect="auto")
    for i, (row, kind) in enumerate(zip(vals, kinds)):
        if kind == "group":
            ax.axhspan(i - 0.5, i + 0.5, color="#f1f0ec", zorder=0)
            continue
        if kind == "band":
            ax.axhspan(i - 0.5, i + 0.5, color="#e4e3de", zorder=0)
        for j, v in enumerate(row):
            if isinstance(v, tuple):
                ax.text(j, i, f"{v[0]:.3f}-{v[1]:.3f}", ha="center", va="center", fontsize=6.5, color="#3d3d3a")
            elif np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if bad[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(SENS_COLUMNS))); ax.set_xticklabels(SENS_COLUMNS, rotation=40, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i, kind in enumerate(kinds):
        if kind == "group":
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_title("Sensitivity tests, FOSI (final design; one change at a time)", fontsize=13)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(SENS_COLUMNS), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(out("figA2_fosi_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    show("figA2_fosi_sensitivity.png")

# %% [markdown]
# # Tables as figures
#
# The paper's Tables 2-5 as compact charts (tables kept). Each chart is drawn by a `panel_*` function below, so that the
# summary page can redraw all of them; change a chart's styling in its panel function, and a figure's size or file name in
# its figure cell. Colours: `METHOD_COLORS` (Shared colours).

# %% [markdown]
# ## Table panels

# %%
TABLE_RC = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
            "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False}
COMMUNITIES = ["Point Hope", "Kivalina", "Kotzebue", "Shishmaref", "Nome"]


def bar_group(ax, labels, series, ylabel, fmt="{:.3f}"):
    """series: {method: [values per label]} -> grouped bars in the method colours, value labels on top."""
    n = len(series); x = np.arange(len(labels)); w = 0.8 / n
    for i, (m, v) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, color=METHOD_COLORS[m], label=m, edgecolor="white", linewidth=1)
        for xi, vi in zip(x + (i - (n - 1) / 2) * w, v):
            ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)


# %%
def panel_t2(axs):
    """Table 2: RMSE, pattern r, CRPS per dataset and method (three axes)."""
    P, methods = table_values()
    for ax, (metric, ylabel, title) in zip(axs, (("RMSE", "RMSE (m)", "Table 2: RMSE"),
                                                 ("Pattern Corr", "Pattern correlation r", "Table 2: pattern r"),
                                                 ("CRPS", "CRPS (m)\n(deterministic, bilinear: MAE)", "Table 2: CRPS"))):
        bar_group(ax, list(P), {m: [P[ds].loc[m, metric] for ds in P] for m in methods}, ylabel,
                  fmt="{:.3f}" if metric != "Pattern Corr" else "{:.2f}")
        if metric == "Pattern Corr":
            ax.set_ylim(0.8, 1.0)
        ax.set_title(title, loc="left")
    axs[0].legend(fontsize=8, loc="upper left")


# %%
def panel_t3(ax):
    """Table 3 (MESACLIP): EngressNet relative to the deterministic U-Net, % change per metric (filled = EngressNet better)."""
    ds = table_values()[0]["MESACLIP"]
    spec = [("IIEE", "IIEE", "lower"), ("CRPS", "CRPS (det: MAE)", "lower"), ("SSIM", "SSIM", "higher"),
            ("Pattern Corr", "Pattern r", "higher"), ("Grad MAE", "Gradient MAE", "lower")]   # (metric, label, better), top to bottom
    y = np.arange(len(spec))[::-1]
    c = METHOD_COLORS["EngressNet"]
    for yi, (k, lab, better) in zip(y, spec):
        pct = 100 * (ds.loc["EngressNet", k] / ds.loc["Deterministic U-Net", k] - 1)
        good = (pct < 0) if better == "lower" else (pct > 0)
        ax.plot([0, pct], [yi, yi], color=c, lw=2)
        ax.plot(pct, yi, "o", ms=9, color=c, mfc=c if good else "white", mew=2)
        ax.text(pct + (1.2 if pct >= 0 else -1.2), yi + 0.22, f"{pct:+.1f}%", ha="left" if pct >= 0 else "right", fontsize=9)
        ax.text(1.02, yi, "better: ←" if better == "lower" else "better: →", transform=ax.get_yaxis_transform(), fontsize=8,
                color=MUTED, va="center")
    ax.axvline(0, color=INK, lw=1)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.8)
    ax.set_yticks(y); ax.set_yticklabels([s[1] for s in spec]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("EngressNet relative to deterministic U-Net (%)")
    ax.set_title("Table 3 (MESACLIP): filled = EngressNet better", loc="left")


# %%
def panel_t4_bias(ax):
    """Table 4, left: nearest-cell bias at each community, per dataset and method (FOSI hatched)."""
    P = table_values()[0]
    t = pd.read_csv(f"{BASE}/results/revisions_2026-11/table4_community{SFX}.csv")
    x = np.arange(len(COMMUNITIES)); w = 0.2
    for i, (ds, m, col) in enumerate([("MESACLIP", "EngressNet", "EngressNet bias yr"), ("MESACLIP", "Deterministic U-Net", "Det bias yr"),
                                      ("FOSI", "EngressNet", "EngressNet bias yr"), ("FOSI", "Deterministic U-Net", "Det bias yr")]):
        v = [t[(t.Dataset == ds) & (t.Community == c)][col].item() for c in COMMUNITIES]
        ax.bar(x + (i - 1.5) * w, v, w * 0.92, color=METHOD_COLORS[m], edgecolor="white", linewidth=1,
               hatch="" if ds == "MESACLIP" else "///", label=f"{m}, {ds}")
    for ds, ls in (("MESACLIP", "--"), ("FOSI", ":")):
        ax.axhline(P[ds].loc["EngressNet", "Bias"], color=INK, ls=ls, lw=1.2,
                   label=f"domain-mean EngressNet bias, {ds} ({P[ds].loc['EngressNet', 'Bias']:+.4f} m)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(COMMUNITIES, rotation=20, ha="right"); ax.set_ylabel("Nearest-cell bias, 2021 (m)")
    ax.set_title("Table 4: community bias (hatched = FOSI)", loc="left"); ax.set_ylim(-0.1, 0.04)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(axis="x", visible=False)


# %%
def panel_t4_values(ax):
    """Table 4, right: yearly mean SIT at each community's nearest cell (open = truth)."""
    t = pd.read_csv(f"{BASE}/results/revisions_2026-11/table4_community{SFX}.csv")
    x = np.arange(len(COMMUNITIES))
    for i, (ds, mk) in enumerate((("MESACLIP", "o"), ("FOSI", "s"))):
        sub = t[t.Dataset == ds].set_index("Community").loc[COMMUNITIES]
        off = (i - 0.5) * 0.3
        ax.scatter(x + off, sub["Truth mean yr"], marker=mk, s=60, facecolor="white", edgecolor=INK, lw=1.5, zorder=3, label=f"truth, {ds}")
        ax.scatter(x + off, sub["EngressNet mean yr"], marker=mk, s=28, color=METHOD_COLORS["EngressNet"], zorder=4, label=f"EngressNet, {ds}")
        ax.scatter(x + off, sub["Det mean yr"], marker=mk, s=28, color=METHOD_COLORS["Deterministic U-Net"], zorder=4,
                   label=f"deterministic, {ds}")
    ax.set_xticks(x); ax.set_xticklabels(COMMUNITIES, rotation=20, ha="right"); ax.set_ylabel("Mean SIT at nearest cell, 2021 (m)")
    ax.set_title("Table 4: actual values (open = truth)", loc="left"); ax.legend(fontsize=7, ncol=3, loc="upper center")
    ax.set_ylim(0, 0.95); ax.grid(axis="x", visible=False)


# %%
def panel_t5_crps(ax):
    """Table 5, left: CRPS against ICESat-2 pass-cell means (deterministic and bilinear: MAE). Returns the values."""
    if FINAL_SET == "final1":
        s = pd.read_csv(f"{BASE}/observations/data/alongtrack_eval/scoreboard.csv")
        s = s[(s.reference == "mean") & (s.subset == "all")].set_index("estimate")
        crps = {"EngressNet": s.loc["EngressNet (final)", "CRPS"], "Deterministic U-Net": s.loc["det-MSE", "MAE of mean"],
                "Bilinear": s.loc["bilinear", "CRPS"]}
    else:  # revised final model: evaluation/revisions/is2_crps_basis.py (FINAL_SET=final2), pass-cell means
        s = pd.read_csv(f"{BASE}/results/revisions_2026-11/is2_crps_basis{SFX}.csv", index_col=0).loc["pass-cell mean"]
        crps = {"EngressNet": s["EngressNet CRPS K=20 (per window, mean of 4)"], "Deterministic U-Net": s["Deterministic MAE (per window, mean of 4)"],
                "Bilinear": s["Bilinear MAE"]}
    for i, m in enumerate(["EngressNet", "Deterministic U-Net", "Bilinear"]):
        ax.bar(i, crps[m], 0.6, color=METHOD_COLORS[m], edgecolor="white")
        ax.text(i, crps[m], f"{crps[m]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["EngressNet", "Deterministic\n(MAE)", "Bilinear\n(MAE)"])
    ax.set_ylabel("CRPS vs ICESat-2 (m)"); ax.set_title("Table 5: ICESat-2 CRPS", loc="left"); ax.grid(axis="x", visible=False)
    return crps


# %%
def panel_t5_var(ax):
    """Table 5, right: sub-grid variability (within-1 deg std) relative to ICESat-2, with the CICE truths. Returns the values."""
    b = pd.read_csv(f"{BASE}/observations/data/alongtrack_eval/cice_ceiling_bar{SFX}.csv", index_col=0).iloc[:, 0]
    is2 = b["ICESat-2 (observed)"]
    var = {"EngressNet": b["EngressNet (final) on PIOMAS"] / is2, "Deterministic U-Net": b["Deterministic U-Net (det-MSE) on PIOMAS"] / is2,
           "Bilinear": b["PIOMAS (bilinear)"] / is2}
    for i, m in enumerate(["EngressNet", "Deterministic U-Net", "Bilinear"]):
        ax.bar(i, var[m], 0.6, color=METHOD_COLORS[m], edgecolor="white")
        ax.text(i, var[m], f"{var[m]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color=INK, lw=1.2, label="ICESat-2 = 1.0")
    for v, lab, ls in ((b["CICE (FOSI) truth"] / is2, "FOSI truth", "--"), (b["CICE (MESACLIP) truth"] / is2, "MESACLIP truth", ":")):
        ax.axhline(v, color=MUTED, lw=1.2, ls=ls, label=f"{lab} ({v:.2f})")
    ax.set_xticks(range(3)); ax.set_xticklabels(["EngressNet", "Deterministic", "Bilinear"])
    ax.set_ylabel("Within-1° std / ICESat-2"); ax.set_ylim(0, 1.1)
    ax.set_title("Table 5: sub-grid variability", loc="left"); ax.legend(fontsize=7, loc="upper right"); ax.grid(axis="x", visible=False)
    return var


# %% [markdown]
# ## Table 2: headline metrics
#
# Output: `tab02_headline.png`

# %%
with matplotlib.rc_context(TABLE_RC):
    fig = plt.figure(figsize=(13, 3.8), constrained_layout=True)
    panel_t2(fig.subplots(1, 3))
    fig.savefig(out("tab02_headline.png"), dpi=250)
    plt.close(fig)
show("tab02_headline.png")

# %% [markdown]
# ## Table 2b: probabilistic baseline (seed ensemble)
#
# EngressNet (one model, 20 members) vs the seed ensemble (10 det-MSE models trained on the same 1825 days, seeds 0-9), from
# `spatial_prob_metrics.py --seedwin`. (final1: EngressNet pooled over 4 windows vs the 12-model seed/window ensemble.)
# Output: `tab02b_seed_ensemble_baseline.png` (final1: `tab02b_pooled_window_baselines.png`)

# %%
DM = f"{BASE}/results/revisions_2026-11/D_metrics"
if FINAL_SET == "final1":
    lab_e, lab_s, title, outfile = ("EngressNet, 4 windows pooled (80 members)", "Seed/window ensemble (det.)",
                                    "Pooled over the 4 training windows", "tab02b_pooled_window_baselines.png")
    pw = pd.read_csv(f"{BASE}/results/revisions_2026-11/pooled_window_ensembles.csv")
    vals = {lab_e: {}, lab_s: {}}
    for ds, pre in (("MESACLIP", "MESA"), ("FOSI", "FOSI")):
        r = pw[(pw.dataset == ds) & pw.estimate.str.startswith("EngressNet")].iloc[0]
        sw = pd.read_csv(f"{DM}/{pre}_2015-2020_extra_E.csv", index_col=0)["Seed/window ensemble (det.)"].apply(pd.to_numeric, errors="coerce")
        vals[lab_e][ds] = {"CRPS": r["CRPS"], "RMSE": r["ens-mean RMSE"], "SSR": r["spread-skill"]}
        vals[lab_s][ds] = {"CRPS": sw["CRPS (point mass: MAE)"], "RMSE": sw["Ensemble-mean RMSE"], "SSR": sw["Spread-skill (corrected)"]}
else:
    lab_e, lab_s, title, outfile = ("EngressNet (1 model, 20 members)", "Seed ensemble (10 det. models)",
                                    "Probabilistic baseline: seed ensemble", "tab02b_seed_ensemble_baseline.png")
    vals = {lab_e: {}, lab_s: {}}
    for ds, pre in (("MESACLIP", "MESA"), ("FOSI", "FOSI")):
        d = pd.read_csv(f"{DM}{SFX}/{pre}_{FS[pre]['windows'][0]}.csv", index_col=0).apply(pd.to_numeric, errors="coerce")
        for lab, col in ((lab_e, "EngressNet"), (lab_s, "Seed/window ensemble (det.)")):
            vals[lab][ds] = {"CRPS": d.loc["CRPS (point mass: MAE)", col], "RMSE": d.loc["Ensemble-mean RMSE", col],
                             "SSR": d.loc["Spread-skill (corrected)", col]}
cols = {lab_e: METHOD_COLORS["EngressNet"], lab_s: METHOD_COLORS["Seed/window ensemble (det.)"]}

with matplotlib.rc_context(TABLE_RC):
    fig = plt.figure(figsize=(13, 3.8), constrained_layout=True)
    axs = fig.subplots(1, 3)
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
    axs[0].set_title(title, loc="left")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.savefig(out(outfile), dpi=250)
    plt.close(fig)
show(outfile)

# %% [markdown]
# ## Table 3: EngressNet relative to the deterministic U-Net (MESACLIP)
#
# Output: `tab03_relative.png`

# %%
with matplotlib.rc_context(TABLE_RC):
    fig = plt.figure(figsize=(6.5, 3.8), constrained_layout=True)
    panel_t3(fig.subplots())
    fig.savefig(out("tab03_relative.png"), dpi=250)
    plt.close(fig)
show("tab03_relative.png")

# %% [markdown]
# ## Table 4: communities (bias and actual values)
#
# Output: `tab04_community.png`

# %%
with matplotlib.rc_context(TABLE_RC):
    fig = plt.figure(figsize=(13, 4), constrained_layout=True)
    ax_bias, ax_vals = fig.subplots(1, 2)
    panel_t4_bias(ax_bias)
    panel_t4_values(ax_vals)
    fig.savefig(out("tab04_community.png"), dpi=250)
    plt.close(fig)
show("tab04_community.png")

# %% [markdown]
# ## Table 4a: community bias only
#
# Output: `tab04a_bias.png`

# %%
with matplotlib.rc_context(TABLE_RC):
    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    panel_t4_bias(fig.subplots())
    fig.savefig(out("tab04a_bias.png"), dpi=250)
    plt.close(fig)
show("tab04a_bias.png")

# %% [markdown]
# ## Table 5: ICESat-2
#
# Output: `tab05_icesat2.png`

# %%
with matplotlib.rc_context(TABLE_RC):
    fig = plt.figure(figsize=(10, 3.8), constrained_layout=True)
    ax_crps, ax_var = fig.subplots(1, 2)
    crps = panel_t5_crps(ax_crps)
    var = panel_t5_var(ax_var)
    fig.savefig(out("tab05_icesat2.png"), dpi=250)
    plt.close(fig)
print("Table 5 CRPS", {k: round(v, 4) for k, v in crps.items()}, "| sub-grid ratio", {k: round(v, 3) for k, v in var.items()})
show("tab05_icesat2.png")

# %% [markdown]
# ## All tables on one page
#
# Output: `tab02-05_summary.png`

# %%
with matplotlib.rc_context(TABLE_RC):
    fig = plt.figure(figsize=(17, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 6)
    panel_t2([fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])])
    panel_t3(fig.add_subplot(gs[1, 0:2]))
    panel_t4_bias(fig.add_subplot(gs[1, 2:4]))
    panel_t4_values(fig.add_subplot(gs[1, 4:6]))
    panel_t5_crps(fig.add_subplot(gs[2, 0:3]))
    panel_t5_var(fig.add_subplot(gs[2, 3:6]))
    fig.savefig(out("tab02-05_summary.png"), dpi=250)
    plt.close(fig)
show("tab02-05_summary.png")

# %% [markdown]
# ## Table values (optional)

# %%
P, _ = table_values()
for ds in P:
    print(ds); print(P[ds][["RMSE", "Pattern Corr", "CRPS", "IIEE", "SSIM", "Grad MAE", "Bias"]].round(4).to_string())

# %% [markdown]
# ## Sub-grid variability bar (ICESat-2 vs CICE truths vs downscaled PIOMAS)
#
# How much 0.1° thickness varies inside one 1° input cell on one day, at QC'd ICESat-2 pass-cells (Oct 2018 - Dec 2021, the
# years the CICE truth covers; >= 6 pass-cells per day and 1° cell). Pooled within-cell standard deviation, every source
# sampled at the same pass-cells on the same day; ICESat-2 measurement noise removed; percentages = ratio to ICESat-2.
# Output: `cice_ceiling_bar_final2.png` (final1: `figures/final_figures/cice_ceiling_bar.png`)

# %%
vals = subgrid_values(nmin=6)
ref = vals["ICESat-2\n(observed)"]
ink, muted = "#14202b", "#57667a"
colors = {"ICESat-2\n(observed)": ink, "CICE (FOSI)\ntruth": "#7fa7c9", "CICE (MESACLIP)\ntruth": "#a9c3da",
          "EngressNet (final)\non PIOMAS": "#1d5d8f", "Deterministic U-Net\n(det-MSE) on PIOMAS": "#d08a4a", "PIOMAS\n(bilinear)": "#c9cfd6"}

fig, ax = plt.subplots(figsize=(9.6, 4.8))
for i, (k, v) in enumerate(vals.items()):
    ax.bar(i, v, width=0.62, color=colors[k])
    ax.text(i, v + ref * 0.025, f"{v:.2f} m" if i == 0 else f"{v:.2f} m\n{v / ref:.0%} of ICESat-2", ha="center", va="bottom",
            fontsize=9, color=ink, zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
fosi = vals["CICE (FOSI)\ntruth"]
ax.axhline(fosi, color=muted, ls="--", lw=1, zorder=1)
ax.text(len(vals) - 0.5, fosi + ref * 0.012, "FOSI CICE: EngressNet's training target", ha="right", va="bottom", fontsize=8.5, color=muted)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals.keys(), fontsize=9)
ax.set_ylabel("sub-grid variability (m)\nstd of 0.1° thickness within a 1° input cell", fontsize=9.5)
ax.set_ylim(0, ref * 1.28)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Sea ice thickness variability inside EngressNet's 1° input cells, at ICESat-2 tracks (2018–2021)", loc="left", fontsize=10)
fig.text(0.01, -0.03, "Pooled within-cell standard deviation, every source sampled at the same ICESat-2 pass-cells on the same "
         "day. ICESat-2: measurement noise removed.\n"
         + ("EngressNet: average over its 80 ensemble members; det-MSE: over its 4 training windows; "
            if FINAL_SET == "final1" else "EngressNet: average over its 20 ensemble members; det-MSE: one model; ")
         + "MESACLIP: over its 6 members.", fontsize=8, color=muted)
fig.tight_layout()
outpath = f"{BASE}/figures/final_figures/cice_ceiling_bar.png" if FINAL_SET == "final1" else f"{BASE}/figures/v2/cice_ceiling_bar{SFX}.png"
fig.savefig(outpath, dpi=250, bbox_inches="tight")
plt.close(fig)
if FINAL_SET == "final1":
    shutil.copy(outpath, f"{BASE}/observations/figures/15b_cice_ceiling_bar.png")
show(outpath)

# %% [markdown]
# # Supplementary figures (the unnumbered `figures/final_figures/` set)
#
# MESACLIP, test year 2021. "Deterministic" = det-MSE U-Net (see the Data cell). Line colours: `MANGANESE` (Shared colours):
# 0 truth, 1 EngressNet, 2 deterministic, 3 bilinear, 4 PIOMAS.

# %% [markdown]
# ## Domain-mean SIT time series
#
# Output: `domain_mean_timeseries.png`

# %%
ts = pd.read_csv(os.path.join(AGG_DIR, "01_domain_mean_sit_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])
ts = rolled(ts, ["truth", "stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"], rdeb.ROLLING_DAYS_TIMESERIES)
fig, ax = plt.subplots(figsize=(10, 4))
for label, col, color, ls, lw in [("Truth", "truth", MANGANESE[0], "-", 2),
                                  ("EngressNet Ensemble Mean", "stochastic_unet_mean", MANGANESE[1], "-", 1.5),
                                  ("Deterministic", "deterministic_unet", MANGANESE[2], "-", 1.5),
                                  ("Bilinear", "bilinear", MANGANESE[3], "-", 1.5),
                                  ("PIOMAS", "piomas", MANGANESE[4], "--", 1.5)]:
    if ts[col].notna().sum():
        ax.plot(ts["time"], ts[col], label=label, color=color, ls=ls, lw=lw)
ax.set_ylabel("Domain-mean SIT (m)")
ax.legend(frameon=True, fontsize=10, ncols=2)
fig.autofmt_xdate()
ax.set_xlim(ts["time"].min(), ts["time"].max())
fig.tight_layout()
ax.grid(alpha=0.2)
fig.savefig(out("domain_mean_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("domain_mean_timeseries.png")

# %% [markdown]
# ## Domain-mean pattern correlation time series
#
# Output: `domain_mean_pattern_corr.png`

# %%
pc = pd.read_csv(os.path.join(AGG_DIR, "01b_domain_mean_pattern_corr_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])
pc = rolled(pc, ["bilinear", "piomas", "deterministic_unet", "stochastic_unet_mean"], rdeb.ROLLING_DAYS_TIMESERIES)
fig, ax = plt.subplots(figsize=(10, 4))
for label, col, color, ls in [("EngressNet Ensemble Mean", "stochastic_unet_mean", MANGANESE[1], "-"),
                              ("Deterministic", "deterministic_unet", MANGANESE[2], "-"),
                              ("Bilinear", "bilinear", MANGANESE[3], "-"),
                              ("PIOMAS", "piomas", MANGANESE[4], "--")]:
    if pc[col].notna().sum():
        ax.plot(pc["time"], pc[col], label=label, color=color, ls=ls, linewidth=1.5)
ax.set_ylabel("Domain-mean Pattern Correlation to Truth")
ax.set_xlim(pc["time"].min(), pc["time"].max())
ax.set_ylim(top=1)
ax.legend(frameon=True, fontsize=10, ncols=2)
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(out("domain_mean_pattern_corr.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("domain_mean_pattern_corr.png")

# %% [markdown]
# ## Community nearest-cell time series
#
# Prediction minus truth at each community's nearest valid ocean cell. Output: `candidate_point_timeseries.png`

# %%
point_df = pd.read_csv(os.path.join(AGG_DIR, "02_candidate_point_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])
points_to_plot = ["Point Hope", "Kivalina", "Kotzebue", "Shishmaref", "Nome"]
fig, axs = plt.subplots(len(points_to_plot), 1, figsize=(10, 2.5 * len(points_to_plot)), sharex=True)
for ax, name in zip(np.atleast_1d(axs), points_to_plot):
    sub = point_df[point_df["point"] == name]
    dist_km = sub["dist_km"].iloc[0] if len(sub) else np.nan
    piv = sub.pivot_table(index="time", columns="method", values="value").sort_index()
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    for label, col, color, ls in [("EngressNet Ensemble Mean", "stochastic_unet_mean", MANGANESE[1], "-"),
                                  ("Deterministic", "deterministic_unet", MANGANESE[2], "-"),
                                  ("Bilinear", "bilinear", MANGANESE[3], "-"),
                                  ("PIOMAS", "piomas", MANGANESE[4], "--")]:
        if col not in piv.columns or "truth" not in piv.columns:
            continue
        y = (piv[col] - piv["truth"]).dropna()
        if y.empty:
            continue
        if rdeb.ROLLING_DAYS_TIMESERIES:
            y = y.rolling(rdeb.rolling_window_samples(y.index.to_series(), rdeb.ROLLING_DAYS_TIMESERIES), center=True, min_periods=1).mean()
        ax.plot(y.index, y, label=label, color=color, linestyle=ls, linewidth=1.2)
    ax.set_title(f"{name} (nearest valid ocean grid cell ~{dist_km:.0f} km)", fontsize=12)
    ax.set_xlim(sub["time"].min(), sub["time"].max())
    ax.set_ylim(-0.3, 1.3)
    ax.grid(alpha=0.2)
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], labels=["-0.2", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2"])
    ax.set_ylabel("Pred - Truth SIT (m)")
    if name == "Point Hope":
        ax.legend(fontsize=10, loc="upper right", ncols=2, frameon=True)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(out("candidate_point_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("candidate_point_timeseries.png")

# %% [markdown]
# ## Absolute-error maps
#
# Three test days (rng seed 0 on days with > 25 % of cells above 0.5 m). Output: `error_figure.png`

# %%
M = map_inputs()
ice_frac = (M.truth[:, 0] > 0.5).mean(axis=(1, 2))
idxs = np.random.default_rng(0).choice(np.where(ice_frac > 0.25)[0], min(3, M.truth.shape[0]), replace=False)
fig, axs = plt.subplots(len(idxs), 3, figsize=(9, 2.5 * len(idxs)), constrained_layout=True, subplot_kw={"projection": M.proj})
for row, idx in enumerate(idxs):
    g = M.tile_geometry[int(M.tile_ids[idx])]
    for col, (title, field) in enumerate((("Bilinear", M.bilinear), ("Deterministic", M.det), ("Ensemble Mean", M.ens))):
        ax = axs[row, col]
        im = ax.pcolormesh(g["target_lon"], g["target_lat"], np.abs(field[idx, 0] - M.truth[idx, 0]), transform=ccrs.PlateCarree(),
                           cmap="viridis", vmin=0, vmax=2, shading="auto")
        polar_frame(M, ax, g["target_lon"], g["target_lat"])
        if row == 0:
            ax.set_title(title, fontsize=18)
    axs[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=13)
cbar = fig.colorbar(im, ax=axs, aspect=20, shrink=0.9, pad=0.02)
cbar.set_label("Absolute Error to Truth (m)", fontsize=18)
cbar.ax.tick_params(labelsize=14)
fig.savefig(out("error_figure.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("error_figure.png")

# %% [markdown]
# ## Temporal pattern-correlation maps
#
# Per-cell temporal correlation with truth over the days of one month. Output: `temporal_pattern_corr.png`

# %%
M = map_inputs()
MONTH = 3
month_idx = np.where(M.months == MONTH)[0]
fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": M.proj})
for ax, (title, field) in zip(axs, (("Bilinear", M.bilinear), ("Deterministic", M.det), ("EngressNet Ensemble Mean", M.ens))):
    corr = np.where(M.land, np.nan, rdeb.temporal_pixel_corr(field, M.truth, month_idx))
    im = ax.pcolormesh(M.lon, M.lat, corr, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
    polar_frame(M, ax, M.lon, M.lat)
    ax.set_title(title, fontsize=18)
cbar = fig.colorbar(im, ax=axs, aspect=20, shrink=0.9, pad=0.02)
cbar.set_label("Temporal Correlation to Truth (March)", fontsize=14)
cbar.ax.tick_params(labelsize=14)
fig.savefig(out("temporal_pattern_corr.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("temporal_pattern_corr.png")

# %% [markdown]
# ## Bias and std-ratio maps
#
# Time-mean bias (top) and the ratio of temporal std to truth's (bottom). Output: `bias_std_ratio_maps.png`

# %%
M = map_inputs()
t_mean, t_std = M.truth[:, 0].mean(axis=0), M.truth[:, 0].std(axis=0)
safe_std = np.where(t_std > 1e-8, t_std, np.nan)
fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True, dpi=150, subplot_kw={"projection": M.proj})
for col, (title, field) in enumerate((("Bilinear", M.bilinear), ("Deterministic", M.det), ("EngressNet Ensemble Mean", M.ens))):
    f = field[:, 0]
    im0 = axs[0, col].pcolormesh(M.lon, M.lat, np.where(M.land, np.nan, f.mean(axis=0) - t_mean), transform=ccrs.PlateCarree(),
                                 cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto")
    polar_frame(M, axs[0, col], M.lon, M.lat)
    axs[0, col].set_title(title, fontsize=18)
    im1 = axs[1, col].pcolormesh(M.lon, M.lat, np.where(M.land, np.nan, f.std(axis=0) / safe_std), transform=ccrs.PlateCarree(),
                                 cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=1, vmin=0, vmax=2), shading="auto")
    polar_frame(M, axs[1, col], M.lon, M.lat)
cbar1 = fig.colorbar(im0, ax=axs[0, :], aspect=20, shrink=0.8, pad=0.02)
cbar1.set_label("Time-mean Bias (m)", fontsize=14)
cbar1.ax.tick_params(labelsize=14)
cbar2 = fig.colorbar(im1, ax=axs[1, :], aspect=20, shrink=0.8, pad=0.02)
cbar2.set_label("Pred/Truth", fontsize=14)
cbar2.ax.tick_params(labelsize=14)
fig.savefig(out("bias_std_ratio_maps.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("bias_std_ratio_maps.png")

# %% [markdown]
# ## Domain-mean bias time series
#
# Output: `domain_mean_bias_timseries.png`

# %%
ts = pd.read_csv(os.path.join(AGG_DIR, "01_domain_mean_sit_timeseries_data_4window.csv"), index_col=0, parse_dates=["time"])
for col in ("stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"):
    ts[col] = ts[col] - ts["truth"]
ts = rolled(ts, ["stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"], rdeb.ROLLING_DAYS_BATCH)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axhline(0, color="black", linewidth=1, linestyle=":")
for label, col, ls, color in [("EngressNet Ensemble Mean", "stochastic_unet_mean", "-", MANGANESE[1]),
                              ("Deterministic", "deterministic_unet", "-", MANGANESE[2]),
                              ("Bilinear", "bilinear", "-", MANGANESE[3]),
                              ("PIOMAS", "piomas", "--", MANGANESE[4])]:
    if ts[col].notna().sum():
        ax.plot(ts["time"], ts[col], label=label, color=color, linewidth=1.5, linestyle=ls)
ax.set_ylabel("Domain-mean Pred - Truth SIT (m)")
ax.legend(fontsize=12, ncols=2, loc="upper right")
ax.set_xlim(ts["time"].min(), ts["time"].max())
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(out("domain_mean_bias_timseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("domain_mean_bias_timseries.png")

# %% [markdown]
# ## Anomaly-correlation maps
#
# Per-cell correlation with truth after removing each one's monthly climatology. Output: `anomaly_correlation.png`

# %%
M = map_inputs()
truth_anom = rdeb.remove_monthly_climatology(M.truth, M.months)
all_idx = np.arange(len(M.months))
fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": M.proj})
for ax, (title, field) in zip(axs, (("Bilinear", M.bilinear), ("Deterministic", M.det), ("EngressNet Ensemble Mean", M.ens))):
    acc = np.where(M.land, np.nan, rdeb.temporal_pixel_corr(rdeb.remove_monthly_climatology(field, M.months), truth_anom, all_idx))
    im = ax.pcolormesh(M.lon, M.lat, acc, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    polar_frame(M, ax, M.lon, M.lat)
    ax.set_title(title, fontsize=18)
cbar = fig.colorbar(im, ax=axs, aspect=20, shrink=0.9, pad=0.02)
cbar.set_label("Anomaly Correlation", fontsize=14)
cbar.ax.tick_params(labelsize=14)
fig.savefig(out("anomaly_correlation.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("anomaly_correlation.png")

# %% [markdown]
# ## Taylor diagram
#
# Output: `taylor_diagram.png`

# %%
taylor_df = pd.read_csv(os.path.join(AGG_DIR, "11_taylor_stats_4window.csv"), index_col=0)
stats = {row["Method"]: (row["Correlation (R)"], row["Std Ratio"]) for _, row in taylor_df.iterrows()}
colors = {"Bilinear": MANGANESE[3], "Deterministic UNet": MANGANESE[2], "Stochastic UNet Mean": MANGANESE[1]}
names = {"Bilinear": "Bilinear", "Deterministic UNet": "Deterministic", "Stochastic UNet Mean": "EngressNet Ensemble Mean"}
r_lim = max(1.3, max(sr for _, sr in stats.values()) * 1.2)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, polar=True)
ax.set_thetamin(0); ax.set_thetamax(90); ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
ax.set_ylim(0, r_lim)
corr_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
ax.set_xticks(np.arccos(corr_ticks))
ax.set_xticklabels([str(c) for c in corr_ticks])
ax.set_xlabel("Correlation", labelpad=15)
ax.set_ylabel("Normalized Standard Deviation", labelpad=10)
for rms in np.arange(0.5, r_lim, 0.5):   # dotted arcs: centred RMSE
    t = np.linspace(0, 2 * np.pi, 200)
    xc, yc = 1 + rms * np.cos(t), rms * np.sin(t)
    th, rr = np.arctan2(yc, xc), np.sqrt(xc ** 2 + yc ** 2)
    keep = (th >= 0) & (th <= np.pi / 2)
    ax.plot(th[keep], rr[keep], color="gray", linestyle=":", linewidth=0.8)
th = np.linspace(0, np.pi / 2, 100)
ax.plot(th, np.ones_like(th), color="black", linestyle="--", linewidth=0.8)
ax.plot([0], [1], marker="*", color=MANGANESE[0], markersize=16, linestyle="none", label="Truth")
for method, (R, std_ratio) in stats.items():
    ax.plot([np.arccos(np.clip(R, -1, 1))], [std_ratio], marker="o", markersize=10, color=colors.get(method, method),
            label=names.get(method, method), linestyle="none")
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.25), fontsize=11, ncols=2, frameon=False)
fig.tight_layout()
fig.savefig(out("taylor_diagram.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("taylor_diagram.png")

# %% [markdown]
# ## Domain-mean power spectra
#
# Output: `domain_mean_psd.png`

# %%
psd = pd.read_csv(os.path.join(AGG_DIR, "13_psd_data_4window.csv"), index_col=0)
k = psd["wavenumber_cycles_per_km"].values
fig, ax = plt.subplots(figsize=(6, 4))
for label, col, color, ls, lw in [("Truth", "Truth", MANGANESE[0], "-", 2),
                                  ("EngressNet Ensemble Mean", "Stochastic UNet Mean", MANGANESE[1], "-", 1.5),
                                  ("Deterministic", "Deterministic UNet", MANGANESE[2], "-", 1.5),
                                  ("Bilinear", "Bilinear", MANGANESE[3], "-", 1.5),
                                  ("PIOMAS", "PIOMAS (obs)", MANGANESE[4], "--", 1.5)]:
    valid = (k > 0) & (psd[col] > 0)
    ax.plot(1 / k[valid], 1 / psd[col][valid], label=label, color=color, ls=ls, lw=lw)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Wavelength (km)")
ax.set_ylabel("PSD")
ax.legend(fontsize=10, ncols=2, loc="upper right")
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
ax.set_title("2021 Annual-Mean")
fig.savefig(out("domain_mean_psd.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("domain_mean_psd.png")

# %% [markdown]
# ## Per-member power-spectrum spread
#
# Output: `per_member_psd_spread.png`

# %%
psdb = pd.read_csv(os.path.join(AGG_DIR, "13b_psd_per_member_spread_data_4window.csv"), index_col=0)
k = psdb["wavenumber_cycles_per_km"].values
valid = (k > 0) & (psdb["Truth"] > 0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(1 / k[valid], 1 / psdb["Truth"][valid], label="Truth", color=MANGANESE[0], linewidth=2)
ax1.fill_between(1 / k[valid], 1 / psdb["Member p10"][valid], 1 / psdb["Member p90"][valid], color=MANGANESE[1], alpha=0.25,
                 label="EngressNet Members (10th–90th pct.)")
ax1.plot(1 / k[valid], 1 / psdb["Member mean"][valid], color=MANGANESE[1], linewidth=1.5, label="EngressNet Ensemble Mean")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Wavelength (km)")
ax1.set_ylabel("PSD")
ax1.set_title("Per-member PSD Spread")
ax1.legend(fontsize=10)
ax1.grid(True, which="both", alpha=0.2)
ax2.plot(1 / k[valid], psdb["Relative member spread pct"][valid], color=MANGANESE[1], linewidth=1.5)
ax2.set_xscale("log")
ax2.set_xlabel("Wavelength (km)")
ax2.set_ylabel("Relative member spread (%)")
ax2.set_title("Inter-member Spectral-Texture Spread")
ax2.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig(out("per_member_psd_spread.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("per_member_psd_spread.png")

# %% [markdown]
# ## CRPS by SIT regime
#
# Output: `crps_regime.png`

# %%
crps_df = pd.read_csv(os.path.join(AGG_DIR, "16_crps_by_regime_data_4window.csv"), index_col=0)
rows = crps_df.iloc[1:]
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(np.arange(len(rows)), rows["CRPS (m)"], color=SODIUM[3], alpha=0.85)
ax.axhline(crps_df.iloc[0]["CRPS (m)"], color="black", linestyle="--", linewidth=2,
           label=f"Domain-mean CRPS ({crps_df.iloc[0]['CRPS (m)']:.2f} m)")
ax.set_xticks(np.arange(len(rows)))
ax.set_xticklabels(rows["SIT regime"], fontsize=12, rotation=45, ha="right")
ax.set_ylabel("CRPS (m)", fontsize=14)
ax.set_title("CRPS by SIT Regime (test=2021)", fontsize=14)
ax.legend(fontsize=12, frameon=False)
ax.tick_params(axis="both", labelsize=12)
fig.tight_layout()
fig.savefig(out("crps_regime.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("crps_regime.png")

# %% [markdown]
# ## Seasonal cycle: Truth, EngressNet mean ± std, Deterministic, Bilinear
#
# Monthly means of the domain-mean SIT (2021, ocean cells). Band: ±1 domain-mean per-cell ensemble std (about 0.1 m in
# winter; the quantity in Fig. 7). Use `clim["engressnet_member_std"]` instead for the std of the members' domain means
# (under 1 mm, so invisible). The first run computes and caches the values (a few minutes). Output: `seasonal_cycle.png`

# %%
clim = seasonal_cycle_values()
sd = clim["engressnet_cell_std"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(clim.index, (clim["stochastic_unet_mean"] - sd).clip(lower=0), clim["stochastic_unet_mean"] + sd,
                color=MANGANESE[1], alpha=0.35, lw=0, label="EngressNet ±1 std")
for col, label, ls, lw, color in [("truth", "Truth", "-", 2, MANGANESE[0]),
                                  ("stochastic_unet_mean", "EngressNet Ensemble Mean", "-", 1.5, MANGANESE[1]),
                                  ("deterministic_unet", "Deterministic", "-", 1.5, MANGANESE[2]),
                                  ("bilinear", "Bilinear", "-", 1.5, MANGANESE[3]),
                                  # ("piomas", "PIOMAS", "--", 1.5, MANGANESE[4]),  # left out so the methods' differences are visible
                                  ]:
    if clim[col].notna().sum():
        ax.plot(clim.index, clim[col], label=label, color=color, linestyle=ls, linewidth=lw)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_ABBR)
ax.set_ylabel("Domain-mean SIT (m)")
ax.set_xlim(1, 12)
ax.set_ylim(0, 1.2)
ax.legend(fontsize=11, ncols=1, loc="upper right")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out("seasonal_cycle.png"), dpi=300, bbox_inches="tight")
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
