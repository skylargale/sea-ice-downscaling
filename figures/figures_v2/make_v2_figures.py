# %% converter cell (not part of the notebook)
"""
make_v2_figures.py: every figures/figures_v2 figure (paper figures, table charts, supplementary set) in one file.

The code is split into cells marked "# %%". Three ways to use it:
  python make_v2_figures.py                     run every cell top to bottom (what submit_make_v2_figures.sh does)
  python make_v2_figures.py --write-notebook    write make_figures.ipynb from this file (for JupyterHub)
  python make_v2_figures.py --from-notebook     copy edits made in make_figures.ipynb back into this file
In VS Code the "# %%" cells also run one at a time ("Run Cell"). Map figures need a compute node (about 64 GB).
This file is the source of truth: --write-notebook overwrites the notebook, --from-notebook overwrites this file
(keeping this first cell).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PY, _NB = os.path.join(_HERE, "make_v2_figures.py"), os.path.join(_HERE, "make_figures.ipynb")  # notebook renamed 2026-11-17 (VS Code kept restoring a stale copy of make_figures.ipynb)
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
# # Make final figures
#
# Saves the PNG into `figures/figures_v2/` (overwriting the current file) and shows it.
#
# `FINAL_SET` in Setup picks the model (`"final2"` = revised final model, the paper; `"final1"` = submitted model, written to `figures/figures_v2/superseded_final1/`).

# %% [markdown]
# ## Setup

# %%
FINAL_SET = "final2"

try:
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
V2 = f"{BASE}/figures/figures_v2"
for _p in (f"{BASE}/evaluation/revisions", f"{BASE}/training", f"{BASE}/evaluation", f"{BASE}/observations/processing"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["FINAL_SET"] = FINAL_SET
import final_set                     # noqa: E402  (evaluation/revisions/final_set.py: batch names of each final model)
importlib.reload(final_set)          # pick up a changed FINAL_SET
import functions_engressnet as fe    # noqa: E402
import run_daily_eval_batch as rdeb  # noqa: E402

FS = final_set.SETS[FINAL_SET]
SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"
OUT_DIR = V2 if FINAL_SET == "final2" else os.path.join(V2, "superseded_final1")  # final1 never overwrites the paper set
RUN = FS["MESA"]["eng"]       # EngressNet batch (MESACLIP) used by the map and supplementary figures
WINDOW = FS["map_window"]     # training-window substring of the run folder


def out(name):
    """Path of an output file in the output folder."""
    path = os.path.join(OUT_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)   # also sub-folders such as supp/
    return path


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

# %%
INK, MUTED = "#0b0b0b", "#8a8984"

METHOD_COLORS = {
    "EngressNet": "#1793e3",                       # Sodium blue
    "Deterministic U-Net": "#7a4fb0",              # medium purple
    "Bilinear": "seagreen",                          # darker Sodium mint
    "Seed/window ensemble (det.)": "darkgoldenrod",  # Table 2b only
}

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MANGANESE = ncp.pal_sample("Manganese_r", 5)   # (no longer used by the figures: methods use METHOD_COLORS)
SODIUM = ncp.pal_sample("Sodium", 8)           # Figs. 6, 7, 9 and CRPS by regime
MAGNESIUM = ncp.pal_sample("Magnesium", 8)

# %% [markdown]
# ## Data Loaders

# %%
'''
The summary data of the MESACLIP line and curve figures (domain-mean and community time series, Taylor, power spectra,
rank histograms, spread-skill and CRPS by regime) are made by `build_figure_data.py` in this folder and saved in
`data_<FINAL_SET>/`, one CSV per figure input: EngressNet = the final model's run, deterministic = the det-MSE U-Net trained
on the same days, both pooled over the 6 MESACLIP members per date; `sources.json` lists where each file came from. Run
`python build_figure_data.py` once per model (`FINAL_SET=final1 python build_figure_data.py` for the submitted model) and
again if a model's evaluation changes. This cell only checks that the files are there. The det-MSE U-Net's fields for the
map figures come from `det_fields()` (last of the Data loaders).
'''

FIG_DATA = os.path.join(V2, f"data_{FINAL_SET}")   # CSVs written by build_figure_data.py
if not os.path.exists(os.path.join(FIG_DATA, "sources.json")):
    raise FileNotFoundError(f"No figure data in {FIG_DATA}: run  python build_figure_data.py  in figures/figures_v2/ "
                            f"(FINAL_SET={FINAL_SET})")
_src = json.load(open(os.path.join(FIG_DATA, "sources.json")))
print(f"Figure data: {FIG_DATA} (written {_src['written']})")
print(f"EngressNet: {_src['engressnet_batch']} | deterministic U-Net: {_src['det_batch']} {_src['det_runs']}")

# %%
POINT_HOPE = {"Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
_CACHE = {}


def cached(key, fn):
    """fn() on the first call for (FINAL_SET, key), the stored result afterwards."""
    if (FINAL_SET, key) not in _CACHE:
        _CACHE[(FINAL_SET, key)] = fn()
    return _CACHE[(FINAL_SET, key)]


def det_fields():
    """The deterministic U-Net's saved fields for the map figures: eval_data/fields.npz of the det-MSE U-Net trained on the same
    days as EngressNet (FS["MESA"]["det"]; final2: one model), with the same test dates (checked against RESULTS_DIR). Keys as in
    any fields.npz; its prediction is "Y_pred_det_phys" (truth "Y_test_phys", bilinear "Y_base_phys", mask "mask_test")."""
    def load():
        z = np.load(os.path.join(resolve_run_window_dir("results", FS["MESA"]["det"], WINDOW), "eval_data", "fields.npz"))
        eng = np.load(os.path.join(RESULTS_DIR, "eval_data", "fields.npz"))
        assert np.allclose(z["Y_test_phys"][:20], eng["Y_test_phys"][:20]), "test dates differ between runs"
        return z
    return cached("det_fields", load)


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


def fig12_data():
    """Fig. 12: model fields for the maps and point series (PIOMAS 2021 input) and the ICESat-2 pass-cells with the matched
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


def fig12_point_series(D, name, p_lo=16, p_hi=84):
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


SENS_COLUMNS = ["MAE", "RMSE", "Bias", "Grad MAE", "Pattern Corr", "SSIM", "IIEE (0.15 m)", "Coastal MAE",
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


def map_inputs():
    """Supplementary maps: the EngressNet run of RESULTS_DIR and the det-MSE U-Net (det_fields())."""
    def load():
        edir = os.path.join(RESULTS_DIR, "eval_data")
        meta = json.load(open(os.path.join(edir, "meta.json")))
        M = SimpleNamespace(bbox=meta["bbox"], points={**meta["candidate_points"], **POINT_HOPE},
                            tile_geometry=pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb")))
        f = np.load(os.path.join(edir, "fields.npz"))
        M.bilinear, M.det, M.ens = f["Y_base_phys"], det_fields()["Y_pred_det_phys"], f["Y_pred_phys"]  # det-MSE U-Net, not the zero-noise pass
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


def seasonal_cycle_values():
    """Monthly means of the domain-mean SIT (2021, ocean cells; same pooling as the time series) plus two EngressNet spreads:
    engressnet_cell_std (domain-mean per-cell ensemble std) and engressnet_member_std (std of the members' domain means,
    < 1 mm). The first call computes them from fields.npz (~8 GB, a few minutes) and caches
    data_<FINAL_SET>/seasonal_cycle_monthly.csv; delete that file to recompute."""
    path = os.path.join(FIG_DATA, "seasonal_cycle_monthly.csv")
    if os.path.exists(path) and "engressnet_cell_std" in pd.read_csv(path, nrows=1).columns:
        return pd.read_csv(path, index_col=0)
    ts = pd.read_csv(os.path.join(FIG_DATA, "domain_mean_sit_timeseries.csv"), index_col=0, parse_dates=["time"])
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
# ## Figures

# %% [markdown]
# ### Figure 2: Domain overview map

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
plt.close(fig)
show("fig02_domain_overview.png")

# %% [markdown]
# ### Figure 3: Resolution comparison map

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
fig.colorbar(im, ax=list(axs), shrink=0.6, extend='max', aspect=15, pad=0.02, label="Sea ice thickness (m)")
fig.savefig(out("fig03_domain_resolution.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig03_domain_resolution.png")

# %% [markdown]
# ### Figure 5: Accuracy metrics (perfect model)

# %%
# --- style ---
rc = {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
      "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False}
figsize = (16, 7.5)
panels = [  # (metric column, y label, value format, y limits or None)
    ("RMSE", "RMSE (m)", "{:.3f}", None),
    ("MAE", "MAE (m)", "{:.3f}", None),
    ("Pattern Corr", "Pattern Correlation (r)", "{:.2f}", (0.8, 1.0)),
    ("Grad MAE", "Gradient MAE (m)", "{:.3f}", None),
    ("SSIM", "SSIM", "{:.2f}", (0.8, 1.0)),
    ("IIEE (0.15 m)", "IIEE, 0.15 m threshold", "{:.3f}", None),
    ("ACC", "Anomaly Correlation", "{:.2f}", (0.6, 1.0)),
]
methods = ["Bilinear", "Deterministic U-Net", "EngressNet"]
LEGEND_LABELS = {"EngressNet": "EngressNet Ensemble Mean"}
bar_span, bar_fill = 0.8, 0.92
value_fontsize, value_rotation, legend_fontsize = 10, 0, 12
headroom = 1.22   # y-axis room above the tallest bar for the labels

# --- data ---
A = pd.read_csv(f"{BASE}/results/revisions_2026-11/accuracy_metrics{SFX}.csv").set_index(["dataset", "method"])
datasets = ["MESACLIP", "FOSI"]

# --- plot ---
with matplotlib.rc_context(rc):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axs = fig.subplots(2, 4).ravel()
    for ax, (metric, ylabel, fmt, ylim) in zip(axs, panels):
        x = np.arange(len(datasets)); w = bar_span / len(methods)
        for i, m in enumerate(methods):
            v = [A.loc[(ds, m), metric] for ds in datasets]
            xs = x + (i - (len(methods) - 1) / 2) * w
            ax.bar(xs, v, w * bar_fill, color=METHOD_COLORS[m], label=LEGEND_LABELS.get(m, m), edgecolor="white", linewidth=1)
            for xi, vi in zip(xs, v):
                ax.text(xi, vi, " " + fmt.format(vi), ha="center", va="bottom", fontsize=value_fontsize, color=INK, rotation=value_rotation)
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
        lo, hi = ylim if ylim else (0, ax.get_ylim()[1])
        ax.set_ylim(lo, lo + (hi - lo) * headroom)
    for ax in axs[len(panels):]:   # remove axes the panels don't use
        ax.remove()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=legend_fontsize, frameon=False, loc="outside lower center", ncols=len(methods))
    fig.savefig(out("fig05_accuracy_metrics.png"), dpi=300)
    plt.close(fig)
show("fig05_accuracy_metrics.png")

# %% [markdown]
# ### Figure 6: Example fields with difference rows

# %%
# --- dates ---
SEED = 284278128      # None = new random dates every run (the seed used is printed); an integer locks them (SEED = 0: the original three days)
N_DAYS = 1            # number of example days (rows)
MIN_ICE_FRAC = 0.25   # only days with more than this fraction of cells thicker than 0.5 m
MEMBER = 1            # ensemble member shown in "One Member"

# --- colour scales ---
FIELD_CMAP, FIELD_VMIN, FIELD_VMAX, FIELD_EXTEND = cmocean.cm.ice_r, 0, 2, "max"   # thickness panels
DIFF_CMAP, DIFF_EXTEND = "RdBu_r", "both"                                            # method - truth panels
DIFF_LIM = 0.5

# --- layout and labels ---
COLS = [("Low-Res Input", "X"), ("Bilinear", "bil"), ("Deterministic", "det"), ("One Member", "mem"),
        ("Ensemble Mean", "ens"), ("High-Res Truth", "truth")]
DIFF_KEYS = ["bil", "det", "mem", "ens"]


def _fig05_all():
    def run_dir(batch):
        m = [d for d in glob.glob(os.path.join(BASE, "results", batch, f"*{WINDOW}*")) if os.path.isdir(d)]
        assert len(m) == 1, m
        return m[0]
    edir = os.path.join(run_dir(FS["MESA"]["eng"]), "eval_data")
    meta = json.load(open(os.path.join(edir, "meta.json")))
    A = SimpleNamespace(bbox=meta["bbox"], points={**meta["candidate_points"], **POINT_HOPE},
                        tile_geometry=pickle.load(open(os.path.join(edir, "tile_geometry.pkl"), "rb")))
    z = np.load(os.path.join(edir, "fields.npz"))
    A.F = {"X": z["X_test_sit_phys"], "bil": z["Y_base_phys"], "mem": z["preds_all_phys"][:, MEMBER], "ens": z["Y_pred_phys"],
           "truth": z["Y_test_phys"], "ocean": z["mask_test"][:, 0] <= 0.5, "tile": z["test_tile_ids"]}
    del z
    zd = np.load(os.path.join(run_dir(FS["MESA"]["det"]), "eval_data", "fields.npz"))
    assert np.array_equal(zd["Y_test_phys"], A.F["truth"]), "det-MSE test set differs"
    A.F["det"] = zd["Y_pred_det_phys"]
    del zd
    A.ice_frac = (A.F["truth"][:, 0] > 0.5).mean(axis=(1, 2))
    st = pd.read_csv(os.path.join(edir, "sample_times.csv"))
    A.dates = pd.to_datetime(st["time"]).values
    A.members = np.arange(len(st)) // st["time"].nunique()   # samples are stored member-block by member-block (6 x 365)
    A.proj, A.boundary, _ = rdeb.make_polar_proj(A.bbox)
    return A


A = cached(f"fig05_all_member{MEMBER}", _fig05_all)

seed = SEED if SEED is not None else int(np.random.SeedSequence().entropy % 2**32)
pool = np.where(A.ice_frac > MIN_ICE_FRAC)[0]
idxs = np.random.default_rng(seed).choice(pool, N_DAYS, replace=False)
idxs = idxs[np.lexsort((A.members[idxs], A.dates[idxs]))]
D = SimpleNamespace(bbox=A.bbox, points=A.points, tile_geometry=A.tile_geometry, proj=A.proj, boundary=A.boundary,
                    F={k: v[idxs] for k, v in A.F.items()})
print(f"SEED = {seed} (set SEED = {seed} to keep these dates)")
for idx in idxs:
    print(f"{pd.Timestamp(A.dates[idx]):%Y-%m-%d}, MESACLIP member {A.members[idx]}, ice frac > 0.5 m: {A.ice_frac[idx]:.2f}")
if DIFF_LIM is None:
    diffs = np.concatenate([np.abs(D.F[k][:, 0] - D.F["truth"][:, 0])[D.F["ocean"]] for k in DIFF_KEYS])
    dlim = float(np.ceil(np.percentile(diffs, 95) / 0.05) * 0.05)
else:
    dlim = float(DIFF_LIM)

fig = plt.figure(figsize=(18, 7), constrained_layout=True)
gs = fig.add_gridspec(2 * N_DAYS + 1, 6, height_ratios=[1] * (2 * N_DAYS) + [0.12])  # last row: the two colourbars
for r, idx in enumerate(idxs):
    clon, clat, tlon, tlat = fig05_geo(D, r)
    for c, (title, k) in enumerate(COLS):
        ax = fig.add_subplot(gs[2 * r, c], projection=D.proj)
        im_f = fig05_map(D, ax, D.F[k][r, 0], clon if k == "X" else tlon, clat if k == "X" else tlat,
                         FIELD_CMAP, FIELD_VMIN, FIELD_VMAX, True)
        if r == 0:
            ax.set_title(title, fontsize=18)
        if c == 0:
            ax.text(-0.08, 0.5, f"{pd.Timestamp(A.dates[idx]):%b %-d %Y}", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=18)
        if k in DIFF_KEYS:
            axd = fig.add_subplot(gs[2 * r + 1, c], projection=D.proj)
            im_d = fig05_map(D, axd, np.where(D.F["ocean"][r], D.F[k][r, 0] - D.F["truth"][r, 0], np.nan), tlon, tlat,
                             DIFF_CMAP, -dlim, dlim, False)
            axd.set_title("− Truth", fontsize=18)
cb = fig.colorbar(im_f, cax=fig.add_subplot(gs[2 * N_DAYS, 0:3]), orientation="horizontal", extend=FIELD_EXTEND)
cb.set_label("Sea ice thickness (m)", fontsize=16)
cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
cb.set_ticklabels(["0", "0.5", "1.0", "1.5", "2.0"])
cb.ax.tick_params(labelsize=16)
cb = fig.colorbar(im_d, cax=fig.add_subplot(gs[2 * N_DAYS, 3:6]), orientation="horizontal", extend=DIFF_EXTEND)
cb.set_label("Method − Truth (m)", fontsize=16)
cb.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
cb.set_ticklabels(["-0.5", "-0.25", "0", "0.25", "0.5"])
cb.ax.tick_params(labelsize=16)

fig.savefig(out("fig06_example_fields.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig06_example_fields.png")

# %% [markdown]
# ### Figure 7: Probabilistic metrics (perfect model)

# %%
# --- style ---
rc = {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
      "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False}
figsize = (13, 7.5)
panels = [  # (D_metrics row, y label, value format, scale, y limits or None)
    ("CRPS (point mass: MAE)", "CRPS (m)", "{:.3f}", 1, None),
    ("ES whole field / sqrt(n_ocean) (beta=1)", "Energy Score / √n (m)", "{:.2f}", 1, None),
    ("Variogram score (p=0.5, 1000 cells)", "Variogram Score", "{:.2f}", 1, None),
    ("FSS thr=0.5 m n=1", "Fraction Skill Score", "{:.2f}", 1, (0.9, 1.0)),
    ("Spread-skill (corrected)", "Spread-Skill (calibrated = 1)", "{:.2f}", 1, None),
    ("Rank: fraction outside ensemble", "Truths Outside Ensemble (%)", "{:.1f}", 100, None),
]
methods = ["Bilinear", "Deterministic U-Net", "EngressNet", "Seed/window ensemble (det.)"]
LEGEND_LABELS = {"EngressNet": "EngressNet (K = 20)", "Seed/window ensemble (det.)": "Seed ensemble (K = 10 det. U-Nets)"}
SINGLE = ["Bilinear", "Deterministic U-Net"]   # single predictions: CRPS = MAE (†), no spread-skill / % outside
ENSEMBLE_ONLY = ["Spread-skill (corrected)", "Rank: fraction outside ensemble"]
bar_span, bar_fill = 0.8, 0.92
value_fontsize, value_rotation, legend_fontsize = 10, 0, 12
headroom = 1.22
ideal_line = dict(color=INK, lw=2)          # ideal fraction outside, per bar
ssr_line = dict(color=INK, lw=1, ls="--")   # spread-skill = 1
footnote = ("† single prediction: CRPS = MAE. FSS of a single prediction uses its 0/1 exceedance field."
            " Spread-skill and % outside: ensembles only.")
outfile = "fig07_probabilistic_metrics.png"

# --- data ---
D = {ds: pd.read_csv(f"{BASE}/results/revisions_2026-11/D_metrics{SFX}/{pre}_{FS[pre]['windows'][0]}.csv", index_col=0)
         .apply(pd.to_numeric, errors="coerce") for ds, pre in (("MESACLIP", "MESA"), ("FOSI", "FOSI"))}
datasets = list(D)

# --- plot ---
with matplotlib.rc_context(rc):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axs = fig.subplots(2, 3).ravel()
    for ax, (row, ylabel, fmt, scale, ylim) in zip(axs, panels):
        x = np.arange(len(datasets)); w = bar_span / len(methods)
        x = np.arange(len(datasets)); w = bar_span / len(methods)   # same bar width in every panel
        shown = [m for m in methods if not (row in ENSEMBLE_ONLY and m in SINGLE)]   # spread-skill / % outside: ensembles only
        for i, m in enumerate(shown):
            xs = x + (i - (len(shown) - 1) / 2) * w   # centred on each dataset tick
            v = [scale * D[ds].loc[row, m] for ds in datasets]
            ax.bar(xs, v, w * bar_fill, color=METHOD_COLORS[m], label=LEGEND_LABELS.get(m, m), edgecolor="white", linewidth=1)
            for xi, vi, ds in zip(xs, v, datasets):
                dagger = "†" if row.startswith("CRPS") and m in SINGLE else ""
                rot = 45 if row.startswith("CRPS") else value_rotation   # tilt only the upper-left (CRPS) panel's values
                ax.text(xi, vi, " " + fmt.format(vi) + dagger, ha="left" if rot else "center", va="bottom",
                        fontsize=value_fontsize, color=INK, rotation=rot, rotation_mode="anchor")
                if row.startswith("Rank"):
                    ideal = scale * D[ds].loc["Rank: ideal outside fraction", m]
                    ax.plot([xi - w * bar_fill / 2, xi + w * bar_fill / 2], [ideal, ideal], **ideal_line, zorder=4)
        if row.startswith("Spread"):
            ax.axhline(1.0, **ssr_line)
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
        lo, hi = ylim if ylim else (0, ax.get_ylim()[1])
        ax.set_ylim(lo, lo + (hi - lo) * headroom)
    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], **ideal_line)); labels.append("Ideal 2/(K+1)")
    fig.legend(handles, labels, fontsize=legend_fontsize, frameon=False, loc="outside upper center", ncols=len(handles))
    fig.text(0.5, -0.01, footnote, ha="center", va="top", fontsize=10, color=MUTED)
    fig.savefig(out(outfile), dpi=300, bbox_inches="tight")
    plt.close(fig)
show(outfile)

# %% [markdown]
# ### Figure 8: Rank histogram, spread-skill by ice type

# %%
# --- data ---
rank_hist_df = pd.read_csv(os.path.join(FIG_DATA, "rank_histogram.csv"), index_col=0)
rank_regime_df = pd.read_csv(os.path.join(FIG_DATA, "rank_histogram_by_regime.csv"), index_col=0)
ss = pd.read_csv(os.path.join(FIG_DATA, "spread_skill_by_regime.csv"), index_col=0)
ss = ss[ss["SIT regime"] != "All ocean cells"]
K = len(rank_hist_df) - 1
expected_freq = rank_hist_df["expected_frequency"].iloc[0]
regime_colors = [SODIUM[0], SODIUM[3], SODIUM[5], SODIUM[6]]   # open water, thin, moderate, thick
regime_short = {"Open water/frazil (0-0.15 m)": "Open Water/Frazil", "Thin ice (0.15-0.5 m)": "Thin Ice",
                "Moderate ice (0.5-1.0 m)": "Moderate Ice", "Thick ice (>1.0 m)": "Thick Ice"}
bins = [b.replace(" (", "\n").replace(")", "") for b in ss["SIT regime"].tolist()]
x = np.arange(len(bins))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# (A) rank histogram, all ocean cells
ax1.bar(rank_hist_df["rank"], rank_hist_df["frequency"], width=0.85, color=SODIUM[3], alpha=0.85, label="Observed Frequency")
ax1.axhline(expected_freq, color="black", linestyle="--", linewidth=2, label="Ideal Uniform Frequency")
ax1.set_xlabel(f"Number of ensemble members below the truth (of {K})", fontsize=14)
ax1.set_ylabel("Frequency", fontsize=14)
ax1.set_title("(A) Rank Histogram (All Ocean Cells)", fontsize=16, loc="left")
ax1.tick_params(axis="both", labelsize=14)
ax1.legend(fontsize=14, frameon=True, loc="lower left")
ax1.set_xlim(-0.5, 20.5)
ax1.set_ylim(0, 0.059)
ax1.set_xticks([0, 5, 10, 15, 20])

# (B) rank curves by SIT type
total = rank_regime_df["count"].sum()
ax2.axhline(1.0, color="black", linestyle="--", linewidth=2)
for (label, grp), color in zip(rank_regime_df.groupby("SIT regime", sort=False), regime_colors):
    share = grp["count"].sum() / total
    ax2.plot(grp["rank"], grp["relative_frequency"], linewidth=4, color=color, label=f"{regime_short.get(label, label)} ({share:.0%} of cells)")
ax2.set_xlabel(f"Number of ensemble members below the truth (of {K})", fontsize=14)
ax2.set_ylabel("Relative to ideal (calibrated = 1)", fontsize=14)
ax2.set_title("(B) Rank Curves by SIT Type", fontsize=16, loc="left")
ax2.tick_params(axis="both", labelsize=14)
ax2.legend(fontsize=14, frameon=False, loc="upper center")
ax2.set_xlim(0, 20)
ax2.set_ylim(0.75, 1.25)
ax2.set_xticks([0, 5, 10, 15, 20])

# (C) spread vs error by SIT type
width = 0.35
ax3.bar(x - width / 2, ss["RMS spread, (K+1)/K-corrected (m)"], width, color=SODIUM[3], label=r"Ensemble Spread ($\sqrt{\frac{K+1}{K}\,\overline{s^2}}$)")
ax3.bar(x + width / 2, ss["RMSE (m)"], width, color=SODIUM[0], label="Ensemble-Mean RMSE")
ax3.set_xticks(x)
ax3.set_xticklabels(bins)
ax3.tick_params(axis="both", labelsize=14)
ax3.set_ylabel("Sea ice thickness (m)", fontsize=14)
ax3.set_title("(C) Spread vs. Error by SIT Type", fontsize=16, loc="left")
ax3.legend(fontsize=14, frameon=False, loc="upper left")
ax3.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
ax3.set_yticklabels(["0", "0.05", "0.1", "0.15", "0.2"])

# (D) spread-skill ratio by SIT type
ax4.bar(x, ss["Spread-skill ratio (1.0 = calibrated)"], color=SODIUM[5], alpha=0.85)
ax4.axhline(1.0, color="black", linestyle="--", linewidth=2, label="Perfect Calibration")
ax4.set_xticks(x)
ax4.set_xticklabels(bins)
ax4.tick_params(axis="both", labelsize=14)
ax4.set_ylabel("Spread-Skill Ratio", fontsize=14)
ax4.set_title("(D) Spread-Skill Ratio by SIT Type", fontsize=16, loc="left")
ax4.legend(fontsize=14, frameon=False, loc="upper right")
ax4.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
ax4.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0"])

fig.tight_layout()
fig.savefig(out("fig08_rank_spread_skill.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig08_rank_spread_skill.png")


# %% [markdown]
# ### Figure 9: Reliability diagrams, P(SIT ≥ t), t = 0.15 m and 0.5 m
#
# Observed event frequency against forecast probability, all ocean cells of the 2021 perfect-model test (MESACLIP, FOSI) and the
# ICESat-2 testbed pass-cells (FOSI-trained models on PIOMAS). One point per member count, no binning: EngressNet k/20 of its members,
# seed ensemble k/10 of its det-MSE U-Nets; deterministic U-Net and bilinear: 0/1 forecasts (two points each). Inset: share of cells
# at each forecast probability (log scale; sharpness). Legend: Brier skill score relative to bilinear. Data: `results/revisions_2026-11/{reliability,brier_scores}_<FINAL_SET>.csv` (`evaluation/revisions/brier_scores.py`).
# Output: `fig09_reliability.png`

# %%
# --- style ---
rc = {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
      "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False}
figsize = (14, 9)
columns = [  # (setting, dataset, region, column title)
    ("perfect model 2021", "MESACLIP", "all ocean", "MESACLIP (perfect model)"),
    ("perfect model 2021", "FOSI", "all ocean", "FOSI (perfect model)"),
    ("ICESat-2 testbed", "FOSI on PIOMAS", "Table 5 pass-cells", "ICESat-2 testbed"),
]
thresholds = [0.15, 0.5]
methods = ["Bilinear", "Deterministic U-Net", "EngressNet", "Seed/window ensemble (det.)"]
LEGEND_LABELS = {"EngressNet": "EngressNet", "Seed/window ensemble (det.)": "Seed ensemble"}
ENSEMBLES = ["EngressNet", "Seed/window ensemble (det.)"]   # curves; the 0/1 forecasts are drawn as points
line_width, marker_size, legend_fontsize = 2, 4.5, 9
diag_line = dict(color=INK, lw=1, ls="--")
inset_bounds = [0.55, 0.07, 0.42, 0.27]   # sharpness histogram, axes fraction (x0, y0, w, h)
inset_fontsize = 8

# --- data ---
R = pd.read_csv(f"{BASE}/results/revisions_2026-11/reliability{SFX}.csv")
S = pd.read_csv(f"{BASE}/results/revisions_2026-11/brier_scores{SFX}.csv")

# --- plot ---
with matplotlib.rc_context(rc):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axs = fig.subplots(len(thresholds), len(columns), sharex=True, sharey=True)
    for r, t in enumerate(thresholds):
        for c, (setting, dset, region, title) in enumerate(columns):
            ax = axs[r, c]
            ax.plot([0, 1], [0, 1], **diag_line)
            for m in methods:
                d = R[(R.setting == setting) & (R.dataset == dset) & (R["threshold (m)"] == t) & (R.region == region) & (R.method == m)]
                b = S[(S.setting == setting) & (S.dataset == dset) & (S["threshold (m)"] == t) & (S.region == region) & (S.method == m)]
                lab = LEGEND_LABELS.get(m, m) + ("" if m == "Bilinear" else f" (BSS {b['BSS vs bilinear'].iloc[0]:.2f})")
                if m in ENSEMBLES:
                    ax.plot(d["forecast probability"], d["observed frequency"], "-o", color=METHOD_COLORS[m], lw=line_width, ms=marker_size, label=lab)
                else:
                    ax.plot(d["forecast probability"], d["observed frequency"], "s", color=METHOD_COLORS[m], ms=marker_size + 3, label=lab)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.legend(fontsize=legend_fontsize, loc="upper left")
            # sharpness inset: share of cells at each forecast probability, ensembles only
            ins = ax.inset_axes(inset_bounds)
            for m in ENSEMBLES:
                d = R[(R.setting == setting) & (R.dataset == dset) & (R["threshold (m)"] == t) & (R.region == region) & (R.method == m)]
                ins.plot(d["forecast probability"], d["count"] / d["count"].sum(), "-", drawstyle="steps-mid", color=METHOD_COLORS[m], lw=1.2)
            ins.set_yscale("log"); ins.set_xlim(0, 1); ins.set_ylim(1e-4, 1)
            ins.set_yticks([1e-4, 1e-2, 1]); ins.set_yticklabels(["0.01%", "1%", "100%"])
            ins.tick_params(labelsize=inset_fontsize, length=2); ins.grid(False)
            ins.set_xticks([0, 0.5, 1]); ins.set_xticklabels(["0", "0.5", "1"])
            ins.set_title("share of cells", fontsize=inset_fontsize, loc="left", pad=2)
            ins.patch.set_alpha(0.9)
            if r == 0:
                ax.set_title(title, loc="left")
            if c == 0:
                ax.set_ylabel(f"Observed frequency, SIT ≥ {t:g} m")
            if r == len(thresholds) - 1:
                ax.set_xlabel("Forecast probability")
    fig.savefig(out("fig09_reliability.png"), dpi=300)
    plt.close(fig)
show("fig09_reliability.png")

# %% [markdown]
# ### Figure 10: Community bias vs. actual values
#
# Left: nearest-cell bias at each community (FOSI hatched). Right: yearly mean SIT at the nearest cell (open = truth).

# %%
# --- style ---
rc = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
      "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False}
figsize = (8, 4)
communities = ["Point Hope", "Kivalina", "Kotzebue", "Shishmaref", "Nome"]
tick_rotation = 0
value_markers = {"MESACLIP": "o", "FOSI": "s"}
truth_size, model_size, dataset_offset = 60, 28, 0.3
values_ylabel, values_title, values_ylim = "Mean SIT at nearest cell, 2021 (m)", "", (0, 0.95)
# legend text per series; {dset} becomes MESACLIP / FOSI
legend_text = {
    "truth": "Truth, {dset}",
    "EngressNet": "EngressNet, {dset}",
    "det": "Deterministic, {dset}",
}
values_legend = dict(ncol=2, loc="upper center", frameon=True)
# font sizes
label_fontsize = 12    # axis labels (y label)
tick_fontsize = 12     # tick labels (community names, y values)
title_fontsize = 12
legend_fontsize = 12

# --- data ---
t = pd.read_csv(f"{BASE}/results/revisions_2026-11/table4_community{SFX}.csv")

# --- plot ---
with matplotlib.rc_context(rc):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax_vals = fig.subplots()
    x = np.arange(len(communities))

    for i, (dset, mk) in enumerate(value_markers.items()):
        sub = t[t.Dataset == dset].set_index("Community").loc[communities]
        off = (i - 0.5) * dataset_offset
        ax_vals.scatter(x + off, sub["Truth mean yr"], marker=mk, s=truth_size, facecolor="white", edgecolor=INK, lw=1.5, zorder=3,
                        label=legend_text["truth"].format(dset=dset))
        ax_vals.scatter(x + off, sub["EngressNet mean yr"], marker=mk, s=model_size, color=METHOD_COLORS["EngressNet"], zorder=4,
                        label=legend_text["EngressNet"].format(dset=dset))
        ax_vals.scatter(x + off, sub["Det mean yr"], marker=mk, s=model_size, color=METHOD_COLORS["Deterministic U-Net"], zorder=4,
                        label=legend_text["det"].format(dset=dset))
    ax_vals.set_xticks(x); ax_vals.set_xticklabels(communities, rotation=tick_rotation, ha="center")
    ax_vals.set_ylabel(values_ylabel, fontsize=label_fontsize)
    ax_vals.tick_params(axis="both", labelsize=tick_fontsize)
    ax_vals.legend(fontsize=legend_fontsize, **values_legend)
    ax_vals.set_ylim(*values_ylim); ax_vals.grid(axis="x", visible=False)

    fig.savefig(out("fig10_community_bias.png"), dpi=300)
    plt.close(fig)
show("fig10_community_bias.png")

# %% [markdown]
# ### Figure 11: Observed seasonal cycle

# %%
seasonal_df = pd.read_csv(os.path.join(BASE, "figures", "piomas_fosi_mesa_seasonal_cycle_2021.csv"), index_col=0)
mesa = pd.read_csv(os.path.join(FIG_DATA, "mesaclip_seasonal_cycle_members.csv"), index_col=0)
N_SD = 2

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(seasonal_df.index, seasonal_df["FOSI truth"], color='royalblue', linewidth=2, label="FOSI")
ax.plot(mesa.index, mesa["mean"], color='rebeccapurple', linewidth=2, label="MESACLIP Ensemble Mean")
ax.fill_between(mesa.index, (mesa["mean"] - N_SD * mesa["std"]).clip(lower=0), mesa["mean"] + N_SD * mesa["std"],
                color='rebeccapurple', alpha=0.2, lw=0, label=f"±{N_SD}σ")
ax.plot(seasonal_df.index, seasonal_df["PIOMAS (obs)"], color='k', linewidth=2, linestyle="--", label="PIOMAS")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_ABBR, fontsize=12)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([f"{y:.1f}" for y in ax.get_yticks()], fontsize=12)
ax.set_ylabel("Domain-mean sea ice thickness, 2021 (m)", fontsize=12)
ax.set_xlim(1, 12)
ax.set_ylim(0, 2)
ax.grid(alpha=0.2)
ax.legend(fontsize=12, loc="upper right")

fig.tight_layout()
fig.savefig(out("fig11_obs_seasonal_cycle.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig11_obs_seasonal_cycle.png")

# %%
sc = pd.read_csv(os.path.join(FIG_DATA, "seasonal_cycle_2021.csv"), index_col=0)
em = pd.read_csv(os.path.join(FIG_DATA, "engressnet_members_seasonal_cycle_2021.csv"), index_col=0)  # build_figure_data.py

LINES = [
    ("truth", "Truth", "k", 3),
    ("engressnet_mean", "EngressNet Ensemble Mean", METHOD_COLORS["EngressNet"], 1.5),
    ("deterministic_unet", "Deterministic", METHOD_COLORS["Deterministic U-Net"], 1.5),
    ("bilinear", "Bilinear", METHOD_COLORS["Bilinear"], 1.5),
]

STYLES = {"MESACLIP": "-", "FOSI": "--"}
ENG = METHOD_COLORS["EngressNet"]
SPREAD = "cell_std"   # band: "cell_std" = domain mean of the per-cell ensemble std (local spread, visible)
                      # "member_std" = std of the members' domain means (1-2 mm, invisible at this scale)

fig, ax = plt.subplots(figsize=(8, 4))
for ds, ls in STYLES.items():
    d = sc[sc["dataset"] == ds].set_index("month")
    e = em[em["dataset"] == ds].set_index("month")
    ax.fill_between(e.index, (e["mean"] - e[SPREAD]).clip(lower=0), e["mean"] + e[SPREAD], color=ENG, alpha=0.2, lw=0)
    for col, label, color, lw in LINES:
        ax.plot(d.index, d[col], color=color, linewidth=lw, linestyle=ls, zorder=0 if col == "truth" else 2)

# Legend: colours = methods, line styles = datasets; EngressNet entries grouped, mean directly above its band
style = {col: (label, color, lw) for col, label, color, lw in LINES}
handles = [plt.Line2D([], [], color=style[c][1], linewidth=style[c][2], label=style[c][0])
           for c in ("truth", "deterministic_unet", "bilinear", "engressnet_mean")]
handles += [plt.Rectangle((0, 0), 1, 1, color=ENG, alpha=0.2, lw=0, label="EngressNet ±σ ")]
handles += [plt.Line2D([], [], color="0.4", linewidth=1.8, linestyle=ls, label=ds) for ds, ls in STYLES.items()]
ax.legend(handles=handles, fontsize=10, loc="upper right")

ax.set_ylabel("Domain-mean sea ice thickness, 2021 (m)", fontsize=12)
ax.set_ylim(0, 2)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_ABBR, fontsize=12)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([f"{y:.1f}" for y in ax.get_yticks()], fontsize=12)
ax.set_xlim(1, 12)
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(out("supplementary/simulated_seasonal_cycle.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/simulated_seasonal_cycle.png")

# %% [markdown]
# ### Figure 12: PIOMAS application

# %%
import matplotlib.dates as mdates

D = fig12_data()
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
        ax.set_title(f"{MONTHS[mo]} {D.year}\nN = {s.sum():,}", fontsize=14)
    else:
        ax.set_title(f"{MONTHS[mo]} {D.year}\n(no ICESat-2 data)", fontsize=14)

cb = fig.colorbar(sc, ax=axes, orientation="horizontal", aspect=30, shrink=0.9, pad=0.06, extend="both")
cb.set_label("ICESat-2 Pass-cell Mean − EngressNet Ensemble Mean (m)", fontsize=14)
cb.ax.tick_params(labelsize=14)

cb2 = fig.colorbar(bg, ax=axes, orientation="horizontal", aspect=50, shrink=0.7, pad=0.02, extend='max')
cb2.set_label("EngressNet Ensemble Mean SIT (m)", fontsize=12)
cb2.ax.tick_params(labelsize=12)

for j, name in enumerate(["Kivalina", "Point Hope"]):   # bottom row: point series
    ax = fig.add_subplot(2, 2, 3 + j)
    mean, lo, hi, _ = fig12_point_series(D, name, 16, 84)
    ax.plot(D.times.values, mean, color=SODIUM[0], linewidth=2, label="Ensemble Mean")
    ax.fill_between(D.times.values, lo, hi, alpha=0.3, color=SODIUM[1], label="±σ")
    ax.set_title(f"{name} Daily Downscaled PIOMAS SIT ({D.year})", fontsize=16)
    ax.set_ylabel("Sea ice thickness (m)", fontsize=16)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=16)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=16)
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlim(D.times.values[0], D.times.values[-1])
    ax.set_ylim(0, 2.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_yticks(np.arange(0, 2.51, 0.5))
    ax.set_yticklabels(["0", "0.5", "1", "1.5", "2", "2.5"])

fig.savefig(out("fig12_piomas_application.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("fig12_piomas_application.png")

# %% [markdown]
# #### Figure 12 supplement: ICESat-2 vs downscaled SIT

# %%
D = fig12_data()
ok = D.qc & np.isfinite(D.Mm).all(1) & np.isfinite(D.Dm).all(1) & np.isfinite(D.Bl)
est = {"Bilinear": D.Bl, "Deterministic U-Net": D.Dm.mean(1), "EngressNet": D.Mm.mean(1)}   # drawn in this order
LABELS = {"Bilinear": "Bilinear", "Deterministic U-Net": "Deterministic", "EngressNet": "EngressNet"}
BINS = np.arange(0, 4.01, 0.2)   # downscaled-SIT bins (m)
MIN_N = 200                      # skip bins with fewer pass-cells
SHADE = ["EngressNet"]           # methods that get a 25-75% band (all three overlap into grey)
LIM = 3.5                        # axis limit (m)
yy = D.y[ok]

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, LIM], [0, LIM], color="k", lw=2, ls="--", zorder=1)
for k, x in est.items():
    xx = x[ok]
    r = xx - yy
    stat = f"(bias = {r.mean():+.3f} m)"
    idx = np.digitize(xx, BINS) - 1
    centers, mean, q25, q75 = [], [], [], []
    for b in range(len(BINS) - 1):
        sel = idx == b
        if sel.sum() >= MIN_N:
            centers.append((BINS[b] + BINS[b + 1]) / 2)
            mean.append(yy[sel].mean()); q25.append(np.percentile(yy[sel], 25)); q75.append(np.percentile(yy[sel], 75))
    if k in SHADE:
        ax.fill_between(centers, q25, q75, color=METHOD_COLORS[k], alpha=0.15, lw=0)
    ax.plot(centers, mean, color=METHOD_COLORS[k], lw=2, marker="o", ms=4, label=f"{LABELS[k]} {stat}")
ax.set_xlabel("Downscaled SIT (m), binned", fontsize=12)
ax.set_ylabel("ICESat-2 pass-cell mean SIT (m)", fontsize=12)
ax.set_xlim(0, LIM); ax.set_ylim(0, LIM)
ax.set_aspect("equal")
ax.grid(alpha=0.2)
ax.set_xlim(0, 3.2); ax.set_ylim(0, 3.2)
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.set_title(f"ICESat-2 vs downscaled PIOMAS\n(N = {ok.sum():,} pass-cells)", fontsize=12)

fig.tight_layout()
fig.savefig(out("supplementary/fig12_is2_scatter.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/fig12_is2_scatter.png")

# %% [markdown]
# ### Figure A1: MESACLIP sensitivity tests

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
# ### Figure A2: FOSI sensitivity tests

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
# ### Figure A3: Accuracy metrics with the seed-ensemble mean

# %%
# --- style ---
rc = {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
      "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False}
figsize = (16, 7.5)
panels = [  # (metric column, y label, value format, y limits or None)
    ("RMSE", "RMSE (m)", "{:.3f}", None),
    ("MAE", "MAE (m)", "{:.3f}", None),
    ("Pattern Corr", "Pattern correlation (r)", "{:.3f}", (0.8, 1.0)),
    ("Grad MAE", "Gradient MAE (m)", "{:.4f}", None),
    ("SSIM", "SSIM", "{:.3f}", (0.8, 1.0)),
    ("IIEE (0.15 m)", "IIEE, 0.15 m threshold", "{:.3f}", None),
    ("ACC", "Anomaly correlation", "{:.3f}", (0.6, 1.0)),
]
methods = ["Bilinear", "Deterministic U-Net", "EngressNet", "Seed/window ensemble (det.)"]
LEGEND_LABELS = {"EngressNet": "EngressNet Ensemble Mean", "Seed/window ensemble (det.)": "Seed-Ensemble Mean (10 det. U-Nets)"}
bar_span, bar_fill = 0.8, 0.92
value_fontsize, value_rotation, legend_fontsize = 10, 60, 12
headroom = 1.22
outfile = "figA3_accuracy_seed_ensemble.png"

# --- data ---
A = pd.read_csv(f"{BASE}/results/revisions_2026-11/accuracy_metrics{SFX}.csv").set_index(["dataset", "method"])
datasets = ["MESACLIP", "FOSI"]

# --- plot ---
with matplotlib.rc_context(rc):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axs = fig.subplots(2, 4).ravel()
    for ax, (metric, ylabel, fmt, ylim) in zip(axs, panels):
        x = np.arange(len(datasets)); w = bar_span / len(methods)
        for i, m in enumerate(methods):
            v = [A.loc[(ds, m), metric] for ds in datasets]
            xs = x + (i - (len(methods) - 1) / 2) * w
            ax.bar(xs, v, w * bar_fill, color=METHOD_COLORS[m], label=LEGEND_LABELS.get(m, m), edgecolor="white", linewidth=1)
            for xi, vi in zip(xs, v):
                ax.text(xi, vi, " " + fmt.format(vi), ha="center", va="bottom", fontsize=value_fontsize, color=INK, rotation=value_rotation)
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
        lo, hi = ylim if ylim else (0, ax.get_ylim()[1])
        ax.set_ylim(lo, lo + (hi - lo) * headroom)
    for ax in axs[len(panels):]:   # remove axes the panels don't use
        ax.remove()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=legend_fontsize, frameon=False, loc="outside lower center", ncols=len(methods))
    fig.savefig(out(outfile), dpi=300)
    plt.close(fig)
show(outfile)

# %% [markdown]
# ## Supplementary figures

# %% [markdown]
# ### Figure S1: Domain-mean SIT time series

# %%
ts = pd.read_csv(os.path.join(FIG_DATA, "domain_mean_sit_timeseries.csv"), index_col=0, parse_dates=["time"])
ts = rolled(ts, ["truth", "stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"], rdeb.ROLLING_DAYS_TIMESERIES)
fig, ax = plt.subplots(figsize=(10, 4))
for label, col, color, ls, lw in [("Truth", "truth", "k", "-", 2),
                                  ("EngressNet Ensemble Mean", "stochastic_unet_mean", METHOD_COLORS["EngressNet"], "-", 1.5),
                                  ("Deterministic", "deterministic_unet", METHOD_COLORS["Deterministic U-Net"], "-", 1.5),
                                  ("Bilinear", "bilinear", METHOD_COLORS["Bilinear"], "-", 1.5),
                                  ("PIOMAS", "piomas", MUTED, "--", 1.5)]:
    if ts[col].notna().sum():
        ax.plot(ts["time"], ts[col], label=label, color=color, ls=ls, lw=lw)
ax.set_ylabel("Domain-mean SIT (m)")
ax.legend(frameon=True, fontsize=10, ncols=2)
fig.autofmt_xdate()
ax.set_xlim(ts["time"].min(), ts["time"].max())
fig.tight_layout()
ax.grid(alpha=0.2)
fig.savefig(out("supplementary/domain_mean_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/domain_mean_timeseries.png")

# %% [markdown]
# ### Figure S2: Domain-mean pattern correlation time series

# %%
pc = pd.read_csv(os.path.join(FIG_DATA, "domain_mean_pattern_corr_timeseries.csv"), index_col=0, parse_dates=["time"])
pc = rolled(pc, ["bilinear", "piomas", "deterministic_unet", "stochastic_unet_mean"], rdeb.ROLLING_DAYS_TIMESERIES)
fig, ax = plt.subplots(figsize=(10, 4))
for label, col, color, ls in [("EngressNet Ensemble Mean", "stochastic_unet_mean", METHOD_COLORS["EngressNet"], "-"),
                              ("Deterministic", "deterministic_unet", METHOD_COLORS["Deterministic U-Net"], "-"),
                              ("Bilinear", "bilinear", METHOD_COLORS["Bilinear"], "-"),
                              ("PIOMAS", "piomas", MUTED, "--")]:
    if pc[col].notna().sum():
        ax.plot(pc["time"], pc[col], label=label, color=color, ls=ls, linewidth=1.5)
ax.set_ylabel("Domain-mean Pattern Correlation to Truth")
ax.set_xlim(pc["time"].min(), pc["time"].max())
ax.set_ylim(top=1)
ax.legend(frameon=True, fontsize=10, ncols=2)
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(out("supplementary/domain_mean_pattern_corr.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/domain_mean_pattern_corr.png")

# %% [markdown]
# ### Figure S3: Community nearest-cell time series

# %%
point_df = pd.read_csv(os.path.join(FIG_DATA, "community_point_timeseries.csv"), index_col=0, parse_dates=["time"])
points_to_plot = ["Point Hope", "Kivalina", "Kotzebue", "Shishmaref", "Nome"]
fig, axs = plt.subplots(len(points_to_plot), 1, figsize=(10, 2.5 * len(points_to_plot)), sharex=True)
for ax, name in zip(np.atleast_1d(axs), points_to_plot):
    sub = point_df[point_df["point"] == name]
    dist_km = sub["dist_km"].iloc[0] if len(sub) else np.nan
    piv = sub.pivot_table(index="time", columns="method", values="value").sort_index()
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    for label, col, color, ls in [("EngressNet Ensemble Mean", "stochastic_unet_mean", METHOD_COLORS["EngressNet"], "-"),
                                  ("Deterministic", "deterministic_unet", METHOD_COLORS["Deterministic U-Net"], "-"),
                                  ("Bilinear", "bilinear", METHOD_COLORS["Bilinear"], "-"),
                                  ("PIOMAS", "piomas", MUTED, "--")]:
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
fig.savefig(out("supplementary/candidate_point_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/candidate_point_timeseries.png")

# %% [markdown]
# ### Figure S4: Temporal pattern-correlation maps
#
# Per-cell temporal correlation with truth over the days of one month.

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
fig.savefig(out("supplementary/temporal_pattern_corr.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/temporal_pattern_corr.png")

# %% [markdown]
# ### Figure S5: Bias and std-ratio maps
#
# Time-mean bias (top) and the ratio of temporal std to truth's (bottom).

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
fig.savefig(out("supplementary/bias_std_ratio_maps.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/bias_std_ratio_maps.png")

# %% [markdown]
# ### Figure S6: Domain-mean bias time series

# %%
ts = pd.read_csv(os.path.join(FIG_DATA, "domain_mean_sit_timeseries.csv"), index_col=0, parse_dates=["time"])
for col in ("stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"):
    ts[col] = ts[col] - ts["truth"]
ts = rolled(ts, ["stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"], rdeb.ROLLING_DAYS_BATCH)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axhline(0, color="black", linewidth=1, linestyle=":")
for label, col, ls, color in [("EngressNet Ensemble Mean", "stochastic_unet_mean", "-", METHOD_COLORS["EngressNet"]),
                              ("Deterministic", "deterministic_unet", "-", METHOD_COLORS["Deterministic U-Net"]),
                              ("Bilinear", "bilinear", "-", METHOD_COLORS["Bilinear"]),
                              ("PIOMAS", "piomas", "--", MUTED)]:
    if ts[col].notna().sum():
        ax.plot(ts["time"], ts[col], label=label, color=color, linewidth=1.5, linestyle=ls)
ax.set_ylabel("Domain-mean Pred - Truth SIT (m)")
ax.legend(fontsize=12, ncols=2, loc="upper right")
ax.set_xlim(ts["time"].min(), ts["time"].max())
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(out("supplementary/domain_mean_bias_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/domain_mean_bias_timeseries.png")

# %% [markdown]
# ### Figure S7: Anomaly-correlation maps
#
# Per-cell correlation with truth after removing each one's monthly climatology.

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
fig.savefig(out("supplementary/anomaly_correlation.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/anomaly_correlation.png")

# %% [markdown]
# ### Figure S8: Taylor diagram

# %%
taylor_df = pd.read_csv(os.path.join(FIG_DATA, "taylor_stats.csv"), index_col=0)
stats = {row["Method"]: (row["Correlation (R)"], row["Std Ratio"]) for _, row in taylor_df.iterrows()}
colors = {"Bilinear": METHOD_COLORS["Bilinear"], "Deterministic UNet": METHOD_COLORS["Deterministic U-Net"], "Stochastic UNet Mean": METHOD_COLORS["EngressNet"]}
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
ax.plot([0], [1], marker="*", color="k", markersize=16, linestyle="none", label="Truth")
for method, (R, std_ratio) in stats.items():
    ax.plot([np.arccos(np.clip(R, -1, 1))], [std_ratio], marker="o", markersize=10, color=colors.get(method, method),
            label=names.get(method, method), linestyle="none")
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.25), fontsize=11, ncols=2, frameon=False)
fig.tight_layout()
fig.savefig(out("supplementary/taylor_diagram.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/taylor_diagram.png")

# %% [markdown]
# ### Figure S9: Domain-mean power spectra

# %%
psd = pd.read_csv(os.path.join(FIG_DATA, "psd.csv"), index_col=0)
k = psd["wavenumber_cycles_per_km"].values
fig, ax = plt.subplots(figsize=(6, 4))
for label, col, color, ls, lw in [("Truth", "Truth", "k", "-", 2),
                                  ("EngressNet Ensemble Mean", "Stochastic UNet Mean", METHOD_COLORS["EngressNet"], "-", 1.5),
                                  ("Deterministic", "Deterministic UNet", METHOD_COLORS["Deterministic U-Net"], "-", 1.5),
                                  ("Bilinear", "Bilinear", METHOD_COLORS["Bilinear"], "-", 1.5),
                                  ("PIOMAS", "PIOMAS (obs)", MUTED, "--", 1.5)]:
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
fig.savefig(out("supplementary/domain_mean_psd.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/domain_mean_psd.png")

# %% [markdown]
# ### Figure S10: Per-member power-spectrum spread

# %%
psdb = pd.read_csv(os.path.join(FIG_DATA, "psd_member_spread.csv"), index_col=0)
k = psdb["wavenumber_cycles_per_km"].values
valid = (k > 0) & (psdb["Truth"] > 0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(1 / k[valid], 1 / psdb["Truth"][valid], label="Truth", color="k", linewidth=2)
ax1.fill_between(1 / k[valid], 1 / psdb["Member p10"][valid], 1 / psdb["Member p90"][valid], color=METHOD_COLORS["EngressNet"], alpha=0.25,
                 label="EngressNet Members (10th–90th pct.)")
ax1.plot(1 / k[valid], 1 / psdb["Member mean"][valid], color=METHOD_COLORS["EngressNet"], linewidth=1.5, label="EngressNet Ensemble Mean")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Wavelength (km)")
ax1.set_ylabel("PSD")
ax1.set_title("Per-member PSD Spread")
ax1.legend(fontsize=10)
ax1.grid(True, which="both", alpha=0.2)
ax2.plot(1 / k[valid], psdb["Relative member spread pct"][valid], color=METHOD_COLORS["EngressNet"], linewidth=1.5)
ax2.set_xscale("log")
ax2.set_xlabel("Wavelength (km)")
ax2.set_ylabel("Relative member spread (%)")
ax2.set_title("Inter-member Spectral-Texture Spread")
ax2.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig(out("supplementary/per_member_psd_spread.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/per_member_psd_spread.png")

# %% [markdown]
# ### Figure S11: CRPS by SIT regime

# %%
crps_df = pd.read_csv(os.path.join(FIG_DATA, "crps_by_regime.csv"), index_col=0)
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
fig.savefig(out("supplementary/crps_regime.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
show("supplementary/crps_regime.png")

# %% [markdown]
# ### Figure S12: Sub-grid variability bar (ICESat-2 vs CICE truths vs downscaled PIOMAS)

# %%
vals = subgrid_values(nmin=6)
ref = vals["ICESat-2\n(observed)"]
ink, muted = "#14202b", "#57667a"
colors = {"ICESat-2\n(observed)": ink, "CICE (FOSI)\ntruth": "#8f8f8f", "CICE (MESACLIP)\ntruth": "#bdbdbd",   # truths: neutral greys
          "EngressNet (final)\non PIOMAS": METHOD_COLORS["EngressNet"], "Deterministic U-Net\n(det-MSE) on PIOMAS": METHOD_COLORS["Deterministic U-Net"],
          "PIOMAS\n(bilinear)": METHOD_COLORS["Bilinear"]}

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
outpath = f"{BASE}/figures/figures_v1/cice_ceiling_bar.png" if FINAL_SET == "final1" else f"{BASE}/figures/figures_v2/supplementary/cice_ceiling_bar.png"
fig.savefig(outpath, dpi=250, bbox_inches="tight")
plt.close(fig)
if FINAL_SET == "final1":
    shutil.copy(outpath, f"{BASE}/observations/figures/15b_cice_ceiling_bar.png")
show(outpath)

# %% [markdown]
# ### Figure S13: ICESat-2 CRPS and sub-grid variability
#
# Left: CRPS against ICESat-2 pass-cell means (deterministic and bilinear: MAE). Right: sub-grid variability (within-1° std) relative to ICESat-2, with the CICE truths.

# %%
# --- style ---
rc = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
      "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False}
figsize = (10, 3.8)
methods = ["EngressNet", "Deterministic U-Net", "Bilinear"]
bar_width, value_fontsize = 0.6, 8
# left panel (CRPS)
crps_ticklabels = ["EngressNet", "Deterministic\n(MAE)", "Bilinear\n(MAE)"]
crps_ylabel, crps_title = "CRPS vs ICESat-2 (m)", "Table 5: ICESat-2 CRPS"
# right panel (sub-grid variability)
var_ticklabels = ["EngressNet", "Deterministic", "Bilinear"]
var_ylabel, var_title, var_ylim = "Within-1° std / ICESat-2", "Table 5: sub-grid variability", (0, 1.1)
truth_line_styles = {"FOSI truth": "--", "MESACLIP truth": ":"}
var_legend = dict(fontsize=7, loc="upper right")

# --- data ---
if FINAL_SET == "final1":
    s = pd.read_csv(f"{BASE}/observations/data/alongtrack_eval/scoreboard.csv")
    s = s[(s.reference == "mean") & (s.subset == "all")].set_index("estimate")
    crps = {"EngressNet": s.loc["EngressNet (final)", "CRPS"], "Deterministic U-Net": s.loc["det-MSE", "MAE of mean"],
            "Bilinear": s.loc["bilinear", "CRPS"]}
else:  # revised final model: evaluation/revisions/is2_crps_basis.py (FINAL_SET=final2), pass-cell means
    s = pd.read_csv(f"{BASE}/results/revisions_2026-11/is2_crps_basis{SFX}.csv", index_col=0).loc["pass-cell mean"]
    crps = {"EngressNet": s["EngressNet CRPS K=20 (per window, mean of 4)"], "Deterministic U-Net": s["Deterministic MAE (per window, mean of 4)"],
            "Bilinear": s["Bilinear MAE"]}
b = pd.read_csv(f"{BASE}/observations/data/alongtrack_eval/cice_ceiling_bar{SFX}.csv", index_col=0).iloc[:, 0]
is2 = b["ICESat-2 (observed)"]
var = {"EngressNet": b["EngressNet (final) on PIOMAS"] / is2, "Deterministic U-Net": b["Deterministic U-Net (det-MSE) on PIOMAS"] / is2,
       "Bilinear": b["PIOMAS (bilinear)"] / is2}
truths = {"FOSI truth": b["CICE (FOSI) truth"] / is2, "MESACLIP truth": b["CICE (MESACLIP) truth"] / is2}

# --- plot ---
with matplotlib.rc_context(rc):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax_crps, ax_var = fig.subplots(1, 2)

    for i, m in enumerate(methods):
        ax_crps.bar(i, crps[m], bar_width, color=METHOD_COLORS[m], edgecolor="white")
        ax_crps.text(i, crps[m], f"{crps[m]:.3f}", ha="center", va="bottom", fontsize=value_fontsize)
    ax_crps.set_xticks(range(3)); ax_crps.set_xticklabels(crps_ticklabels)
    ax_crps.set_ylabel(crps_ylabel); ax_crps.set_title(crps_title, loc="left"); ax_crps.grid(axis="x", visible=False)

    for i, m in enumerate(methods):
        ax_var.bar(i, var[m], bar_width, color=METHOD_COLORS[m], edgecolor="white")
        ax_var.text(i, var[m], f"{var[m]:.2f}", ha="center", va="bottom", fontsize=value_fontsize)
    ax_var.axhline(1.0, color=INK, lw=1.2, label="ICESat-2 = 1.0")
    for lab, v in truths.items():
        ax_var.axhline(v, color=MUTED, lw=1.2, ls=truth_line_styles[lab], label=f"{lab} ({v:.2f})")
    ax_var.set_xticks(range(3)); ax_var.set_xticklabels(var_ticklabels)
    ax_var.set_ylabel(var_ylabel); ax_var.set_ylim(*var_ylim)
    ax_var.set_title(var_title, loc="left"); ax_var.legend(**var_legend); ax_var.grid(axis="x", visible=False)

    fig.savefig(out("supplementary/icesat2_crps_var.png"), dpi=250)
    plt.close(fig)
print("Table 5 CRPS", {k: round(v, 4) for k, v in crps.items()}, "| sub-grid ratio", {k: round(v, 3) for k, v in var.items()})
show("supplementary/icesat2_crps_var.png")

# %%

