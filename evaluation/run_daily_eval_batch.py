#!/usr/bin/env python3
"""Standalone driver for evaluation_plots_daily.ipynb's "Batch mode" section (section 16),
so it can run headlessly under PBS instead of interactively in a live kernel.

Regenerates every figure/table (sections 00-16) for every run under a results/<BATCH_NAME>/
directory, saving into evaluation/saved_figs/<BATCH_NAME>/<run_name>/ -- exactly matching what the
notebook's own batch-mode cell (the BATCH_RUN_DIRS loop) produces. Kept as a plain .py file
rather than `jupyter nbconvert --execute` since downscaling_env has no nbconvert/nbclient,
the same constraint noted throughout this project's other batch-processing notebooks.

`run_all_sections` below is a direct copy of evaluation_plots_daily.ipynb's batch-mode cell
(cell id c8c99533, as of the 2026-08-10 PIOMAS-overlap/rolling-mean fix) -- if that cell
changes, re-sync this file from it.
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.spatial import cKDTree

# Same POINT_HOPE_LAT/LON, and the same section-by-section logic, as sections 2/7 above --
# just parameterized by eval_dir/save_dir instead of the interactively-set globals, and using
# a seeded RNG for the ensemble-figure sample choice (section 3) so batch reruns are reproducible.
POINT_HOPE_LAT = 68.3415
POINT_HOPE_LON_360 = -166.7578 % 360


def rounded_boundary_path(proj, lon_min, lon_max, lat_min, lat_max, n=50):
    lons = np.concatenate([
        np.linspace(lon_min, lon_max, n), np.full(n, lon_max),
        np.linspace(lon_max, lon_min, n), np.full(n, lon_min),
    ])
    lats = np.concatenate([
        np.full(n, lat_min), np.linspace(lat_min, lat_max, n),
        np.full(n, lat_max), np.linspace(lat_max, lat_min, n),
    ])
    pts = proj.transform_points(ccrs.PlateCarree(), lons, lats)
    return mpath.Path(pts[:, :2])


def make_polar_proj(bbox, n=50):
    central_lon = (bbox["lon_min"] + bbox["lon_max"]) / 2
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    boundary_path = rounded_boundary_path(proj, bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"], n)
    return proj, boundary_path, central_lon


def style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_=None, lat_=None, pad_frac=0.001,
                    points=None, point_color="red"):
    lon_min = bbox["lon_min"] % 360
    lon_max = bbox["lon_max"] % 360
    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    if lon_ is not None and lat_ is not None:
        lo0, lo1 = float(np.min(lon_)), float(np.max(lon_))
        la0, la1 = float(np.min(lat_)), float(np.max(lat_))
        pad_lo = (lo1 - lo0) * pad_frac or 0.5
        pad_la = (la1 - la0) * pad_frac or 0.5
        ext_lo0, ext_lo1 = lo0 - pad_lo, lo1 + pad_lo
        ext_la0, ext_la1 = la0 - pad_la, la1 + pad_la
        ax.set_extent([ext_lo0, ext_lo1, ext_la0, ext_la1], crs=ccrs.PlateCarree())
        panel_boundary = rounded_boundary_path(proj, ext_lo0, ext_lo1, ext_la0, ext_la1)
    else:
        ax.set_extent([lon_min, lon_max, bbox["lat_min"], bbox["lat_max"]], crs=ccrs.PlateCarree())
        panel_boundary = boundary_path
    ax.set_boundary(panel_boundary, transform=proj)
    if points is None:
        points = candidate_points
    for name, pt in points.items():
        ax.plot(pt["lon"], pt["lat"], marker="*", color=point_color, markersize=10, transform=ccrs.PlateCarree())
        ax.text(pt["lon"] + 1, pt["lat"], name, color=point_color, fontsize=7, transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, linestyle="--", alpha=0.4)


def unweighted_pattern_corr(pred, truth):
    p = pred.reshape(pred.shape[0], -1)
    t = truth.reshape(truth.shape[0], -1)
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    num = (p * t).sum(axis=1)
    den = np.sqrt((p ** 2).sum(axis=1) * (t ** 2).sum(axis=1)) + 1e-8
    return float((num / den).mean())


def coslat_weighted_pattern_corr(pred, truth, tile_ids, tile_geometry):
    N, _, H, W = pred.shape
    corrs = np.empty(N)
    for i in range(N):
        target_lat = tile_geometry[int(tile_ids[i])]["target_lat"]
        w = np.cos(np.deg2rad(target_lat))[:, None] * np.ones((1, W))
        w = w / w.sum()
        p, t = pred[i, 0], truth[i, 0]
        p_anom = p - (w * p).sum()
        t_anom = t - (w * t).sum()
        num = (w * p_anom * t_anom).sum()
        den = np.sqrt((w * p_anom ** 2).sum() * (w * t_anom ** 2).sum()) + 1e-8
        corrs[i] = num / den
    return float(corrs.mean())


def temporal_pixel_corr(pred_phys, truth_phys, idx):
    p = pred_phys[idx, 0]
    t = truth_phys[idx, 0]
    p_anom = p - p.mean(axis=0, keepdims=True)
    t_anom = t - t.mean(axis=0, keepdims=True)
    num = (p_anom * t_anom).sum(axis=0)
    den = np.sqrt((p_anom ** 2).sum(axis=0) * (t_anom ** 2).sum(axis=0))
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-8)


def remove_monthly_climatology(field_phys, months):
    anom = np.empty_like(field_phys)
    for mo in range(1, 13):
        idx = np.where(months == mo)[0]
        if len(idx) == 0:
            continue
        clim = field_phys[idx].mean(axis=0, keepdims=True)
        anom[idx] = field_phys[idx] - clim
    return anom


def nearest_valid_index(lat_grid, lon_grid, exclude_hw, point_lat, point_lon_360):
    lat_grid = np.asarray(lat_grid)
    lon_grid = np.asarray(lon_grid) % 360
    dlat_deg = lat_grid[:, None] - point_lat
    dlon_deg = (lon_grid[None, :] - point_lon_360 + 180) % 360 - 180
    dlon_deg = dlon_deg * np.cos(np.deg2rad(point_lat))
    dist2 = dlat_deg ** 2 + dlon_deg ** 2
    valid = ~np.asarray(exclude_hw).astype(bool)
    dist2 = np.where(valid, dist2, np.inf)
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(iy), int(ix), float(np.sqrt(dist2[iy, ix])) * 111.0


PIOMAS_PATH = Path("/glade/campaign/cgd/ccr/yeager/OBS/seaice/PIOMAS/PIOMAS.hi.1978-2020.nc")
# Real daily 2021 PIOMAS on the same native grid (confirmed 2026-09-15: identical (nlat, nlon)
# = (120, 360), same grid_source as the monthly campaign file), extending coverage past the
# monthly file's 2020 cutoff without touching that read-only /glade/campaign file. Additive
# only -- if this path is ever missing, load_piomas() silently falls back to monthly-only
# 1978-2020 coverage (unchanged prior behavior) rather than raising.
PIOMAS_DAILY_2021_PATH = Path("/glade/derecho/scratch/skygale/observations/PIOMAS_daily/PIOMAS_hiday_2021.nc")
PIOMAS_MAX_GAP_DAYS = 20  # see the "Observational comparison" section markdown above


def load_piomas(path=PIOMAS_PATH, daily_extra_paths=(PIOMAS_DAILY_2021_PATH,)):
    ds = xr.open_dataset(path)
    year = np.floor(ds["time"].values).astype(int)
    month = np.clip(np.round((ds["time"].values - year) * 12 + 0.5).astype(int), 1, 12)
    times = pd.to_datetime({"year": year, "month": month, "day": 1})
    hi = ds["hi"].values
    lat = ds["lat"].values
    lon = ds["lon"].values % 360

    # Splice in any available real daily data past the monthly file's coverage (e.g. 2021),
    # on the same native grid, so batch-mode PIOMAS comparisons aren't silently limited to
    # whatever the monthly campaign archive happens to end at. Matches by nearest calendar day
    # instead of nearest month for whatever period this covers, tightening the max_gap_days
    # match for those samples specifically -- everything before this file's coverage is
    # unaffected.
    times = pd.DatetimeIndex(times)
    for extra_path in daily_extra_paths:
        if not extra_path.exists():
            continue
        ds_extra = xr.open_dataset(extra_path)
        if ds_extra["hi"].shape[1:] != hi.shape[1:]:
            raise ValueError(
                f"{extra_path} grid shape {ds_extra['hi'].shape[1:]} doesn't match "
                f"{path}'s {hi.shape[1:]} -- can't concatenate onto the same (nlat, nlon) grid."
            )
        hi = np.concatenate([hi, ds_extra["hi"].values], axis=0)
        # This file's time coord is a 365-day "noleap" cftime calendar (no Feb 29), not
        # plain datetime64 -- convert via isoformat rather than pd.DatetimeIndex(...) directly,
        # which can't parse cftime objects. Safe here since noleap -> standard-calendar dates
        # are unambiguous (standard calendar is a strict superset).
        extra_times = pd.DatetimeIndex([pd.Timestamp(t.isoformat()) for t in ds_extra["time"].values])
        times = times.append(extra_times)
    return hi, lat, lon, times


def nearest_grid_indices(src_lat, src_lon, query_lat, query_lon):
    def to_xyz(lat_deg, lon_deg):
        lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    tree = cKDTree(to_xyz(np.asarray(src_lat).ravel(), np.asarray(src_lon).ravel()))
    _, idx = tree.query(to_xyz(np.asarray(query_lat), np.asarray(query_lon)))
    return idx


def build_piomas_regridded(target_lat, target_lon, sample_times, max_gap_days=PIOMAS_MAX_GAP_DAYS):
    """Same NaN-out-of-coverage logic as the interactive "Observational comparison" cell
    above, so batch-mode PIOMAS comparisons only ever use the genuinely overlapping period."""
    piomas_hi, piomas_lat, piomas_lon, piomas_times = load_piomas()
    tlat_2d, tlon_2d = np.meshgrid(target_lat, target_lon, indexing="ij")
    space_idx = nearest_grid_indices(piomas_lat, piomas_lon, tlat_2d.ravel(), tlon_2d.ravel())
    piomas_hi_flat = piomas_hi.reshape(piomas_hi.shape[0], -1)
    sample_times = pd.to_datetime(sample_times).values
    piomas_month_idx = np.abs(piomas_times.values[None, :] - sample_times[:, None]).argmin(axis=1)
    gap_days = np.abs(piomas_times.values[piomas_month_idx] - sample_times) / np.timedelta64(1, "D")
    regridded = piomas_hi_flat[piomas_month_idx][:, space_idx].reshape(
        len(sample_times), 1, len(target_lat), len(target_lon)
    ).astype(np.float32)
    out_of_coverage = gap_days > max_gap_days
    regridded[out_of_coverage] = np.nan
    n_out = int(out_of_coverage.sum())
    coverage_end = pd.Timestamp(piomas_times.values.max()).strftime("%Y-%m-%d")
    print(f"  PIOMAS coverage: {len(sample_times) - n_out}/{len(sample_times)} test samples overlap "
          f"PIOMAS's record (through {coverage_end}).")
    return regridded


def rolling_window_samples(times, window_days):
    """Convert a time window (days) to an integer rolling-window size in samples, based
    on the median spacing between consecutive sample times.

    Multi-member datasets (MESA) carry several rows per calendar date (one per ensemble
    member) -- after sorting by time, most consecutive gaps are 0 seconds, which used to
    collapse the median gap to 0 and silently fall back to window=1 (no smoothing at all).
    Using the spacing between *unique* dates for dt_days, then scaling the row-count window
    by rows-per-date, keeps the window a true `window_days`-wide span of calendar time
    regardless of how many members/rows share each date."""
    times = pd.to_datetime(pd.Series(times)).sort_values().reset_index(drop=True)
    uniq = times.drop_duplicates()
    if len(uniq) < 2:
        return 1
    dt_days = (uniq.iloc[-1] - uniq.iloc[0]).total_seconds() / 86400.0 / (len(uniq) - 1)
    rows_per_date = len(times) / len(uniq)
    return max(1, int(round(window_days / dt_days * rows_per_date))) if dt_days > 0 else 1


ROLLING_DAYS_BATCH = 7  # centered running mean (days) applied to daily time-series plots below
ROLLING_DAYS_TIMESERIES = 10  # centered running mean (days) for the domain-mean SIT (section 1)
                               # and candidate-point (section 2) time series specifically -- these
                               # plot raw daily values (vs. section 8's already-differenced bias),
                               # so they read noisier at the same window and got a wider one


def run_all_sections(eval_dir, save_dir, show=False):
    """Regenerate every figure/table in this notebook (sections 00-16) for one run,
    using each section's notebook default parameters, and save into save_dir."""
    eval_dir = Path(eval_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(fig, name):
        fig.savefig(save_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_table(df, name):
        df.to_csv(save_dir / f"{name}.csv")

    # ---- Load everything ----
    fields = np.load(eval_dir / "fields.npz")
    X_test_sit_phys = fields["X_test_sit_phys"]
    Y_base_phys = fields["Y_base_phys"]
    Y_pred_det_phys = fields["Y_pred_det_phys"]
    preds_all_phys = fields["preds_all_phys"]
    Y_pred_phys = fields["Y_pred_phys"]
    Y_test_phys = fields["Y_test_phys"]
    mask_test = fields["mask_test"]
    test_tile_ids = fields["test_tile_ids"]
    land_mask = fields["land_mask"]
    hlat, hlon = fields["hlat"], fields["hlon"]
    llat, llon = fields["llat"], fields["llon"]

    with open(eval_dir / "tile_geometry.pkl", "rb") as f:
        tile_geometry = pickle.load(f)
    with open(eval_dir / "meta.json") as f:
        meta = json.load(f)

    bbox = meta["bbox"]
    use_patches = meta["use_patches"]
    candidate_points = meta["candidate_points"]
    candidate_points = {**candidate_points, "Point Hope": {"lat": POINT_HOPE_LAT, "lon": POINT_HOPE_LON_360}}
    sample_times_df = pd.read_csv(eval_dir / "sample_times.csv", parse_dates=["time"])

    proj, boundary_path, central_lon = make_polar_proj(bbox)

    def _style(ax, lon_=None, lat_=None, points=None):
        style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_, lat_, points=points)

    # ---- 00. Metrics ----
    metrics_path = eval_dir.parent / "metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        save_table(metrics_df, "00_metrics")

    # ---- 00b. Pattern corr check ----
    pattern_corr_check = []
    for method_name, pred_field in [
        ("Bilinear", Y_base_phys), ("Deterministic UNet", Y_pred_det_phys), ("Stochastic UNet Mean", Y_pred_phys),
    ]:
        unweighted = unweighted_pattern_corr(pred_field, Y_test_phys)
        weighted = coslat_weighted_pattern_corr(pred_field, Y_test_phys, test_tile_ids, tile_geometry)
        pattern_corr_check.append({
            "Method": method_name,
            "Pattern Corr (unweighted, recomputed)": round(unweighted, 4),
            "Pattern Corr (cos-lat weighted)": round(weighted, 4),
            "Difference": round(weighted - unweighted, 4),
        })
    save_table(pd.DataFrame(pattern_corr_check), "00b_pattern_corr_check")


    # ---- 14. Rank histogram & reliability diagram ----
    ocean_bool = mask_test[:, 0] <= 0.5
    ens_hwk = np.moveaxis(preds_all_phys[:, :, 0], 1, -1)
    K_eval_here = ens_hwk.shape[-1]
    ens_ocean = ens_hwk[ocean_bool]
    truth_ocean = Y_test_phys[:, 0][ocean_bool]

    rng = np.random.default_rng(0)
    below = (ens_ocean < truth_ocean[:, None]).sum(axis=1)
    ties = (ens_ocean == truth_ocean[:, None]).sum(axis=1)
    ranks = below + rng.integers(0, ties + 1)
    rank_counts = np.bincount(ranks, minlength=K_eval_here + 1)
    rank_freq = rank_counts / rank_counts.sum()
    expected_freq = 1.0 / (K_eval_here + 1)

    def calibration_score(ratio):
        if not np.isfinite(ratio) or ratio <= 0:
            return np.nan
        return min(ratio, 1.0 / ratio)

    rank_calibration = np.array([calibration_score(f / expected_freq) for f in rank_freq])

    rank_hist_df = pd.DataFrame({
        "rank": np.arange(K_eval_here + 1), "count": rank_counts,
        "frequency": rank_freq, "expected_frequency": expected_freq,
        "Calibration Score (1.0 = ideal)": rank_calibration,
    })
    save_table(rank_hist_df, "14_rank_histogram_data")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ax1.bar(rank_hist_df["rank"], rank_hist_df["frequency"], width=0.9, color="tab:blue", alpha=0.85, label="Observed")
    ax1.axhline(expected_freq, color="black", linestyle="--", linewidth=1.2, label="Perfect calibration (flat)")
    ax1.set_xlabel(f"Rank of truth among {K_eval_here} sorted ensemble members")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Rank histogram (Talagrand diagram)")
    ax1.legend(fontsize=9, frameon=False)

    ax2.bar(rank_hist_df["rank"], rank_hist_df["Calibration Score (1.0 = ideal)"], width=0.9, color="tab:green", alpha=0.85)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax2.set_xlabel(f"Rank of truth among {K_eval_here} sorted ensemble members")
    ax2.set_ylabel("Calibration score (min(ratio, 1/ratio))")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Per-rank calibration score")
    ax2.legend(fontsize=9, frameon=False, loc="lower right")
    plt.tight_layout()
    save_fig(fig, "14_rank_histogram")

    RELIABILITY_THRESHOLDS = [0.15, 0.5, 1.0]
    N_PROB_BINS = 10
    bin_edges = np.linspace(0, 1, N_PROB_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    threshold_colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]

    reliability_rows = []
    curves = {}
    for thr in RELIABILITY_THRESHOLDS:
        p_forecast = (ens_ocean > thr).mean(axis=1)
        outcome = (truth_ocean > thr).astype(float)
        bin_idx = np.clip(np.digitize(p_forecast, bin_edges[1:-1]), 0, N_PROB_BINS - 1)
        mean_fc = np.full(N_PROB_BINS, np.nan)
        obs_freq = np.full(N_PROB_BINS, np.nan)
        counts = np.zeros(N_PROB_BINS, dtype=int)
        for b in range(N_PROB_BINS):
            sel = bin_idx == b
            counts[b] = sel.sum()
            if counts[b] > 0:
                mean_fc[b] = p_forecast[sel].mean()
                obs_freq[b] = outcome[sel].mean()
        brier = float(np.mean((p_forecast - outcome) ** 2))
        curves[thr] = {"mean_forecast": mean_fc, "obs_freq": obs_freq, "count": counts}
        reliability_rows.append({
            "Threshold (m)": thr, "Brier Score": round(brier, 4),
            "Climatological frequency": round(float(outcome.mean()), 4),
        })
    reliability_df = pd.DataFrame(reliability_rows)
    save_table(reliability_df, "14_reliability_diagram_data")

    fig, (ax_main, ax_hist) = plt.subplots(2, 1, figsize=(6.5, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax_main.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Perfectly reliable")
    for thr, color in zip(RELIABILITY_THRESHOLDS, threshold_colors):
        c = curves[thr]
        valid = c["count"] > 0
        sizes = 20 + 200 * c["count"][valid] / c["count"].max()
        ax_main.plot(c["mean_forecast"][valid], c["obs_freq"][valid], color=color, linewidth=1.5, zorder=2)
        ax_main.scatter(c["mean_forecast"][valid], c["obs_freq"][valid], s=sizes, color=color, zorder=3, label=f"SIT > {thr} m")
        ax_hist.bar(bin_centers, c["count"] / c["count"].sum(), width=0.8 / N_PROB_BINS, color=color, alpha=0.5, label=f"SIT > {thr} m")
    ax_main.set_ylabel("Observed frequency")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.legend(fontsize=9, frameon=False, loc="upper left")
    ax_main.set_title("Reliability diagram -- ensemble-fraction forecast of SIT exceedance")
    ax_hist.set_xlabel("Forecast probability")
    ax_hist.set_ylabel("Fraction of\nsamples")
    ax_hist.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    save_fig(fig, "14_reliability_diagram")

    # ---- 15. Spread-skill by SIT regime ----
    SIT_BIN_EDGES = [0.0, 0.15, 0.5, 1.0, np.inf]
    SIT_BIN_LABELS = ["Open water/frazil (0-0.15 m)", "Thin ice (0.15-0.5 m)", "Moderate ice (0.5-1.0 m)", "Thick ice (>1.0 m)"]
    ens_mean_ocean = ens_ocean.mean(axis=1)
    ens_spread_ocean = ens_ocean.std(axis=1, ddof=1)
    abs_error_ocean = np.abs(ens_mean_ocean - truth_ocean)
    bin_idx_sit = np.digitize(truth_ocean, SIT_BIN_EDGES[1:-1])

    def calibration_score(ratio):
        if not np.isfinite(ratio) or ratio <= 0:
            return np.nan
        return min(ratio, 1.0 / ratio)

    spread_skill_rows = []
    for b, label in enumerate(SIT_BIN_LABELS):
        sel = bin_idx_sit == b
        n = int(sel.sum())
        mean_spread = float(ens_spread_ocean[sel].mean()) if n > 0 else np.nan
        mean_error = float(abs_error_ocean[sel].mean()) if n > 0 else np.nan
        ratio = mean_spread / mean_error if mean_error > 0 else np.nan
        spread_skill_rows.append({
            "SIT regime": label, "N": n,
            "Mean true SIT (m)": round(float(truth_ocean[sel].mean()), 4) if n > 0 else np.nan,
            "Mean ensemble spread (m)": round(mean_spread, 4),
            "Mean |error| (m)": round(mean_error, 4),
            "Spread/Error": round(ratio, 4) if np.isfinite(ratio) else np.nan,
            "Calibration Score (1.0 = ideal)": round(calibration_score(ratio), 4),
        })
    spread_skill_df = pd.DataFrame(spread_skill_rows)
    save_table(spread_skill_df, "15_spread_skill_by_regime_data")

    x = np.arange(len(SIT_BIN_LABELS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.35
    ax1.bar(x - width / 2, spread_skill_df["Mean ensemble spread (m)"], width, color="tab:blue", label="Ensemble spread (std)")
    ax1.bar(x + width / 2, spread_skill_df["Mean |error| (m)"], width, color="tab:orange", label="|Ensemble mean - truth|")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SIT_BIN_LABELS, fontsize=8)
    ax1.set_ylabel("m")
    ax1.set_title("Spread vs. error by true-SIT regime")
    ax1.legend(fontsize=9, frameon=False)

    ax2.bar(x, spread_skill_df["Calibration Score (1.0 = ideal)"], color="tab:green", alpha=0.85)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SIT_BIN_LABELS, fontsize=8)
    ax2.set_ylabel("Calibration score (min(ratio, 1/ratio))")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Normalized calibration score by true-SIT regime")
    ax2.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    save_fig(fig, "15_spread_skill_by_regime")

    # ---- 16. CRPS (proper scoring rule) by SIT regime ----
    # NRG/energy-form CRPS for an ensemble forecast against a single truth (Hersbach 2000):
    #   CRPS = mean_k|x_k - y| - (1/(2K^2)) * sum_i sum_j |x_i - x_j|
    # The double sum is computed via the standard order-statistic identity (sort ensemble
    # ascending, weight member i (1-indexed) by (2i - K - 1)) instead of an O(K^2) double loop --
    # this is the same trick used to make Kendall-tau-style pairwise sums tractable, and matters
    # here since ens_ocean can be tens of millions of (sample, cell) rows.
    K_crps = ens_ocean.shape[1]
    crps_term1 = np.abs(ens_ocean - truth_ocean[:, None]).mean(axis=1)
    sorted_ens = np.sort(ens_ocean, axis=1)
    crps_weights = (2 * np.arange(1, K_crps + 1) - K_crps - 1)
    crps_term2 = (sorted_ens * crps_weights[None, :]).sum(axis=1) / (K_crps ** 2)
    crps_per_sample = crps_term1 - crps_term2

    crps_rows = [{
        "SIT regime": "All (domain-wide)", "N": int(crps_per_sample.size),
        "Mean true SIT (m)": round(float(truth_ocean.mean()), 4),
        "CRPS (m)": round(float(crps_per_sample.mean()), 4),
    }]
    for b, label in enumerate(SIT_BIN_LABELS):
        sel = bin_idx_sit == b
        n = int(sel.sum())
        crps_rows.append({
            "SIT regime": label, "N": n,
            "Mean true SIT (m)": round(float(truth_ocean[sel].mean()), 4) if n > 0 else np.nan,
            "CRPS (m)": round(float(crps_per_sample[sel].mean()), 4) if n > 0 else np.nan,
        })
    crps_df = pd.DataFrame(crps_rows)
    save_table(crps_df, "16_crps_by_regime_data")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_rows = crps_df.iloc[1:]  # regime bars only, "All" reported separately as a reference line
    ax.bar(np.arange(len(plot_rows)), plot_rows["CRPS (m)"], color="tab:purple", alpha=0.85)
    ax.axhline(crps_df.iloc[0]["CRPS (m)"], color="black", linestyle="--", linewidth=1.2,
               label=f"Domain-wide CRPS ({crps_df.iloc[0]['CRPS (m)']:.4f} m)")
    ax.set_xticks(np.arange(len(plot_rows)))
    ax.set_xticklabels(plot_rows["SIT regime"], fontsize=8)
    ax.set_ylabel("CRPS (m, lower = better)")
    ax.set_title("CRPS by true-SIT regime")
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    save_fig(fig, "16_crps_by_regime")

    # ---- 00. Domain overview ----
    geo0 = tile_geometry[0] if not use_patches else None
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    cf = ax.pcolormesh(hlon, hlat, land_mask[0, 0], cmap="Greys", vmin=0, vmax=1, shading="auto", transform=ccrs.PlateCarree())
    if geo0 is not None:
        ax.plot(
            [geo0["target_lon"].min(), geo0["target_lon"].max(), geo0["target_lon"].max(), geo0["target_lon"].min(), geo0["target_lon"].min()],
            [geo0["target_lat"].min(), geo0["target_lat"].min(), geo0["target_lat"].max(), geo0["target_lat"].max(), geo0["target_lat"].min()],
            color="tab:blue", linewidth=1.5, transform=ccrs.PlateCarree(), label="test sub-domain",
        )
    _style(ax)
    ax.set_title("Land mask + candidate points" + (" + test sub-domain" if geo0 is not None else ""))
    fig.colorbar(cf, ax=ax, shrink=0.7, label="1 = land")
    plt.tight_layout()
    save_fig(fig, "00_domain_overview")

    if use_patches:
        return  # sections 1, 2, 5-10 need one coherent domain; nothing else to do

    piomas_regridded_phys = build_piomas_regridded(
        geo0["target_lat"], geo0["target_lon"], sample_times_df["time"]
    )

    ocean_weight = (1.0 - mask_test[0, 0]).clip(0, 1)
    weight_sum = ocean_weight.sum()

    def domain_mean(field_phys):
        return (field_phys[:, 0] * ocean_weight[None, :, :]).sum(axis=(1, 2)) / weight_sum

    # ---- 1. Domain-mean SIT time series ----
    ts_df = sample_times_df.copy()
    ts_df["truth"] = domain_mean(Y_test_phys)
    ts_df["stochastic_unet_mean"] = domain_mean(Y_pred_phys)
    ts_df["deterministic_unet"] = domain_mean(Y_pred_det_phys)
    ts_df["bilinear"] = domain_mean(Y_base_phys)
    ts_df["piomas"] = domain_mean(piomas_regridded_phys)
    ts_df = ts_df.sort_values("time").reset_index(drop=True)
    save_table(ts_df, "01_domain_mean_sit_timeseries_data")

    ts_plot_cols = ["truth", "stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"]
    ts_plot_df = ts_df.copy()
    if ROLLING_DAYS_TIMESERIES:
        win = rolling_window_samples(ts_plot_df["time"], ROLLING_DAYS_TIMESERIES)
        ts_plot_df[ts_plot_cols] = ts_plot_df[ts_plot_cols].rolling(win, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, col, color, ls in [
        ("Truth", "truth", "black", "-"), ("Stochastic UNet Mean", "stochastic_unet_mean", "tab:blue", "-"),
        ("Deterministic UNet", "deterministic_unet", "tab:orange", "-"), ("Bilinear", "bilinear", "tab:green", "--"),
        ("PIOMAS (obs)", "piomas", "tab:red", ":"),
    ]:
        if ts_plot_df[col].notna().sum() == 0:
            continue  # e.g. PIOMAS has zero overlap with this run's test period
        ax.plot(ts_plot_df["time"], ts_plot_df[col], label=label, color=color, ls=ls, lw=2 if col == "truth" else 1.5)
    ax.set_ylabel("Domain-mean SIT (m)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()
    save_fig(fig, "01_domain_mean_sit_timeseries")

    # ---- 1b. Domain-mean pattern correlation time series ----
    def spatial_pattern_corr_series(pred_phys, truth_phys, weight_hw):
        p, t = pred_phys[:, 0], truth_phys[:, 0]
        w = weight_hw[None, :, :]
        p_mean = (w * p).sum(axis=(1, 2))
        t_mean = (w * t).sum(axis=(1, 2))
        p_anom, t_anom = p - p_mean[:, None, None], t - t_mean[:, None, None]
        num = (w * p_anom * t_anom).sum(axis=(1, 2))
        den = np.sqrt((w * p_anom ** 2).sum(axis=(1, 2)) * (w * t_anom ** 2).sum(axis=(1, 2)))
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-8)

    land_hw_ts = mask_test[0, 0] > 0.5
    ocean_w_hw_ts = np.where(land_hw_ts, 0.0, np.cos(np.deg2rad(geo0["target_lat"]))[:, None] * np.ones((1, len(geo0["target_lon"]))))
    ocean_w_hw_ts = ocean_w_hw_ts / ocean_w_hw_ts.sum()

    pc_df = sample_times_df.copy()
    pc_df["bilinear"] = spatial_pattern_corr_series(Y_base_phys, Y_test_phys, ocean_w_hw_ts)
    pc_df["deterministic_unet"] = spatial_pattern_corr_series(Y_pred_det_phys, Y_test_phys, ocean_w_hw_ts)
    pc_df["stochastic_unet_mean"] = spatial_pattern_corr_series(Y_pred_phys, Y_test_phys, ocean_w_hw_ts)
    pc_df["piomas"] = spatial_pattern_corr_series(piomas_regridded_phys, Y_test_phys, ocean_w_hw_ts)
    pc_df = pc_df.sort_values("time").reset_index(drop=True)
    save_table(pc_df, "01b_domain_mean_pattern_corr_timeseries_data")

    pc_plot_cols = ["stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"]
    pc_plot_df = pc_df.copy()
    if ROLLING_DAYS_TIMESERIES:
        win = rolling_window_samples(pc_plot_df["time"], ROLLING_DAYS_TIMESERIES)
        pc_plot_df[pc_plot_cols] = pc_plot_df[pc_plot_cols].rolling(win, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, col, color, ls in [
        ("Stochastic UNet Mean", "stochastic_unet_mean", "tab:blue", "-"),
        ("Deterministic UNet", "deterministic_unet", "tab:orange", "-"),
        ("Bilinear", "bilinear", "tab:green", "--"),
        ("PIOMAS (obs)", "piomas", "tab:red", ":"),
    ]:
        if pc_plot_df[col].notna().sum() == 0:
            continue  # e.g. PIOMAS has zero overlap with this run's test period
        ax.plot(pc_plot_df["time"], pc_plot_df[col], label=label, color=color, ls=ls, linewidth=1.5)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_ylabel("Domain-mean pattern correlation vs. Truth")
    ax.set_xlabel("Time")
    ax.set_ylim(top=1.02)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()
    save_fig(fig, "01b_domain_mean_pattern_corr_timeseries")

    # ---- 2. Candidate-point time series ----
    geo = tile_geometry[0]
    target_lat, target_lon = geo["target_lat"], geo["target_lon"]
    land_mask_hw = mask_test[0, 0]
    FLAT_STD_TOL = 1e-6
    flat_truth_hw = Y_test_phys[:, 0].std(axis=0) < FLAT_STD_TOL
    flat_piomas_hw = np.nanstd(piomas_regridded_phys[:, 0], axis=0) < FLAT_STD_TOL
    exclude_hw = (land_mask_hw > 0.5) | flat_truth_hw | flat_piomas_hw

    fields_phys = {
        "truth": Y_test_phys, "bilinear": Y_base_phys,
        "deterministic_unet": Y_pred_det_phys, "stochastic_unet_mean": Y_pred_phys,
        "piomas": piomas_regridded_phys,
    }
    time_index = sample_times_df["time"]
    locations = {}
    rows = []
    for point_name, pt in candidate_points.items():
        iy, ix, dist_km = nearest_valid_index(target_lat, target_lon, exclude_hw, pt["lat"], pt["lon"])
        locations[point_name] = (iy, ix, dist_km)
        for method_name, field in fields_phys.items():
            values = field[:, 0, iy, ix]
            for t, v in zip(time_index, values):
                rows.append({"point": point_name, "method": method_name, "time": t, "value": float(v),
                             "grid_iy": iy, "grid_ix": ix, "dist_km": dist_km})
    point_df = pd.DataFrame(rows).sort_values(["point", "method", "time"]).reset_index(drop=True)
    save_table(point_df, "02_candidate_point_timeseries_data")

    points_to_plot = list(candidate_points.keys())
    method_style = {
        "truth": ("Truth", "black", "-", 2),
        "stochastic_unet_mean": ("Stochastic UNet Mean", "tab:blue", "-", 1.5),
        "deterministic_unet": ("Deterministic UNet", "tab:orange", "-", 1.5),
        "bilinear": ("Bilinear", "tab:green", "--", 1.5),
        "piomas": ("PIOMAS (obs)", "tab:red", ":", 1.5),
    }
    fig, axs = plt.subplots(len(points_to_plot), 1, figsize=(10, 3 * len(points_to_plot)), sharex=True)
    if len(points_to_plot) == 1:
        axs = [axs]
    for ax, point_name in zip(axs, points_to_plot):
        sub = point_df[point_df["point"] == point_name]
        iy, ix, dist_km = locations[point_name]
        for method, (label, color, ls, lw) in method_style.items():
            m = sub[sub["method"] == method].sort_values("time")
            if m["value"].notna().sum() == 0:
                continue  # e.g. PIOMAS has zero overlap with this run's test period
            y = m["value"]
            if ROLLING_DAYS_TIMESERIES:
                y = y.rolling(rolling_window_samples(m["time"], ROLLING_DAYS_TIMESERIES), center=True, min_periods=1).mean()
            ax.plot(m["time"], y, label=label, color=color, linestyle=ls, linewidth=lw)
        ax.set_title(f"{point_name} (nearest valid ocean grid cell ~{dist_km:.1f} km away)", fontsize=11)
        ax.set_ylabel("SIT (m)")
        if point_name == "Kivalina":
            ax.legend(fontsize=8, loc="upper left", frameon=False)
    axs[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    plt.tight_layout()
    save_fig(fig, "02_candidate_point_timeseries")

    # ---- 3. Ensemble figure ----
    N_SAMPLES = min(3, Y_test_phys.shape[0])
    MIN_ICE_THICKNESS = 0.5
    MIN_ICE_FRAC = 0.25
    ice_frac = (Y_test_phys[:, 0] > MIN_ICE_THICKNESS).mean(axis=(1, 2))
    valid_idxs = np.where(ice_frac > MIN_ICE_FRAC)[0]
    if len(valid_idxs) < N_SAMPLES:
        raise ValueError(f"Only {len(valid_idxs)} samples meet the ice criteria, need {N_SAMPLES}")
    SAMPLE_IDXS = np.random.default_rng(0).choice(valid_idxs, N_SAMPLES, replace=False)
    MEMBER_IDX = min(4, preds_all_phys.shape[1] - 1)
    VMIN, VMAX = 0, 3
    CMAP = "Blues"

    panel_titles = ["Low-Res Input", "Bilinear", "Deterministic", "One Member", "Ensemble Mean", "High-Res Truth"]
    fig, axs = plt.subplots(
        len(SAMPLE_IDXS), 6, figsize=(18, 3.3 * len(SAMPLE_IDXS)), constrained_layout=True, dpi=150,
        subplot_kw={"projection": proj},
    )
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
                ax.set_title(panel_titles[col], fontsize=13)
        axs[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=13)
    cbar = fig.colorbar(im, ax=axs, aspect=30, shrink=0.8, pad=0.02)
    cbar.set_label("Sea ice thickness (m)", fontsize=13)
    save_fig(fig, "03_ensemble_figure")

    # ---- 4. Error figure ----
    ERR_VMIN, ERR_VMAX = 0, 2
    ERR_CMAP = "viridis"
    panel_titles_err = ["Bilinear", "Deterministic", "Ensemble Mean"]
    fig, axs = plt.subplots(
        len(SAMPLE_IDXS), 3, figsize=(9, 2.7 * len(SAMPLE_IDXS)), constrained_layout=True, dpi=150,
        subplot_kw={"projection": proj},
    )
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
                ax.set_title(panel_titles_err[col], fontsize=13)
        axs[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=13)
    cbar = fig.colorbar(im, ax=axs, aspect=20, shrink=0.9, pad=0.02)
    cbar.set_label("|Y - Truth| Absolute Error (m)", fontsize=12)
    save_fig(fig, "04_error_figure")

    # ---- 5. Pattern correlation maps (March) ----
    MONTH = 3
    MIN_SAMPLES = 3
    month_idx = np.where(sample_times_df["time"].dt.month.values == MONTH)[0]
    if len(month_idx) >= MIN_SAMPLES:
        geo = tile_geometry[0]
        tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]
        land_hw = mask_test[0, 0] > 0.5
        methods_maps = {}
        for label, field in [
            ("Bilinear", Y_base_phys), ("Deterministic UNet", Y_pred_det_phys), ("Stochastic UNet Mean", Y_pred_phys),
        ]:
            corr_map = temporal_pixel_corr(field, Y_test_phys, month_idx)
            methods_maps[label] = np.where(land_hw, np.nan, corr_map)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
        for ax, (label, corr_map) in zip(axs, methods_maps.items()):
            im = ax.pcolormesh(tgt_lon, tgt_lat, corr_map, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
            _style(ax, tgt_lon, tgt_lat)
            ax.set_title(label, fontsize=13)
        cbar = fig.colorbar(im, ax=axs, aspect=30, shrink=0.8, pad=0.02)
        cbar.set_label(f"Temporal Pearson corr. vs. truth (month={MONTH})", fontsize=12)
        save_fig(fig, "05_pattern_corr_maps")

        ocean_w = np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon)))
        ocean_w = np.where(land_hw, 0.0, ocean_w)
        ocean_w = ocean_w / ocean_w.sum()
        summary = [
            {"Method": label, "Domain mean temporal corr. (cos-lat weighted)": round(float(np.nansum(ocean_w * corr_map)), 4)}
            for label, corr_map in methods_maps.items()
        ]
        save_table(pd.DataFrame(summary), "05_pattern_corr_summary")

    # ---- 6. Bias & std-ratio maps ----
    geo = tile_geometry[0]
    tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]
    land_hw = mask_test[0, 0] > 0.5
    BIAS_LIM = 0.5
    STD_RATIO_LIM = (0, 2)
    method_fields = {
        "Bilinear": Y_base_phys, "Deterministic UNet": Y_pred_det_phys, "Stochastic UNet Mean": Y_pred_phys,
    }
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
        ax.set_title(f"{label}: time-mean bias", fontsize=12)
    for col, (label, rmap) in enumerate(std_ratio_maps.items()):
        ax = axs[1, col]
        im1 = ax.pcolormesh(tgt_lon, tgt_lat, rmap, transform=ccrs.PlateCarree(), cmap="RdBu_r",
                             norm=TwoSlopeNorm(vcenter=1, vmin=STD_RATIO_LIM[0], vmax=STD_RATIO_LIM[1]), shading="auto")
        _style(ax, tgt_lon, tgt_lat)
        ax.set_title(f"{label}: std ratio (pred/truth)", fontsize=12)
    fig.colorbar(im0, ax=axs[0, :], shrink=0.8, pad=0.02, label="Time-mean bias (m)")
    fig.colorbar(im1, ax=axs[1, :], shrink=0.8, pad=0.02, label="Temporal std ratio")
    save_fig(fig, "06_bias_std_ratio_maps")

    ocean_w = np.where(land_hw, 0.0, np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon))))
    ocean_w = ocean_w / ocean_w.sum()
    summary = [{
        "Method": label,
        "Domain mean bias (cos-lat weighted, m)": round(float(np.nansum(ocean_w * bias_maps[label])), 4),
        "Domain mean std ratio (cos-lat weighted)": round(float(np.nansum(ocean_w * std_ratio_maps[label])), 4),
    } for label in method_fields]
    save_table(pd.DataFrame(summary), "06_bias_std_ratio_summary")

    # ---- 7. Trend maps ----
    MIN_YEARS_FOR_TREND = 5
    years = (sample_times_df["time"].dt.year + (sample_times_df["time"].dt.month - 1) / 12.0).values
    if years.max() - years.min() >= MIN_YEARS_FOR_TREND:
        def pixel_trend_per_decade(field_phys, years):
            t = years - years.mean()
            y = field_phys[:, 0]
            y_anom = y - y.mean(axis=0, keepdims=True)
            slope_per_year = (t[:, None, None] * y_anom).sum(axis=0) / (t ** 2).sum()
            return slope_per_year * 10.0

        method_fields_trend = {
            "Truth": Y_test_phys, "Bilinear": Y_base_phys,
            "Deterministic UNet": Y_pred_det_phys, "Stochastic UNet Mean": Y_pred_phys,
        }
        trend_maps = {
            label: np.where(land_hw, np.nan, pixel_trend_per_decade(field, years))
            for label, field in method_fields_trend.items()
        }
        TREND_LIM = float(np.nanmax(np.abs(trend_maps["Truth"])) * 1.2) or 0.1
        fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
        for ax, (label, tmap) in zip(axs, trend_maps.items()):
            im = ax.pcolormesh(tgt_lon, tgt_lat, tmap, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-TREND_LIM, vmax=TREND_LIM, shading="auto")
            _style(ax, tgt_lon, tgt_lat)
            ax.set_title(label, fontsize=12)
        cbar = fig.colorbar(im, ax=axs, aspect=30, shrink=0.8, pad=0.02)
        cbar.set_label("SIT trend (m / decades)", fontsize=12)
        save_fig(fig, "07_trend_maps")

        ocean_w2 = np.where(land_hw, 0.0, np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon))))
        ocean_w2 = ocean_w2 / ocean_w2.sum()
        truth_domain_trend = float(np.nansum(ocean_w2 * trend_maps["Truth"]))
        trend_summary = [{
            "Method": label,
            "Domain mean trend (cos-lat weighted, m/decade)": round(float(np.nansum(ocean_w2 * tmap)), 4),
            "Trend bias vs. Truth (m/decade)": (0.0 if label == "Truth" else round(float(np.nansum(ocean_w2 * tmap)) - truth_domain_trend, 4)),
        } for label, tmap in trend_maps.items()]
        save_table(pd.DataFrame(trend_summary), "07_trend_summary")

    # ---- 8. Domain-mean bias time series ----
    bias_df = ts_df.copy()
    bias_df["bias_stochastic_unet_mean"] = bias_df["stochastic_unet_mean"] - bias_df["truth"]
    bias_df["bias_deterministic_unet"] = bias_df["deterministic_unet"] - bias_df["truth"]
    bias_df["bias_bilinear"] = bias_df["bilinear"] - bias_df["truth"]
    bias_df["bias_piomas"] = bias_df["piomas"] - bias_df["truth"]

    bias_cols = ["bias_stochastic_unet_mean", "bias_deterministic_unet", "bias_bilinear", "bias_piomas"]
    bias_plot_df = bias_df.copy()
    if ROLLING_DAYS_BATCH:
        win = rolling_window_samples(bias_plot_df["time"], ROLLING_DAYS_BATCH)
        bias_plot_df[bias_cols] = bias_plot_df[bias_cols].rolling(win, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    for label, col, color in [
        ("Stochastic UNet Mean", "bias_stochastic_unet_mean", "tab:blue"),
        ("Deterministic UNet", "bias_deterministic_unet", "tab:orange"),
        ("Bilinear", "bias_bilinear", "tab:green"),
        ("PIOMAS (obs)", "bias_piomas", "tab:red"),
    ]:
        if bias_plot_df[col].notna().sum() == 0:
            continue  # e.g. PIOMAS has zero overlap with this run's test period
        ax.plot(bias_plot_df["time"], bias_plot_df[col], label=label, color=color, linewidth=1.5)
    ax.set_ylabel("Domain-mean bias (pred - truth, m)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()
    save_fig(fig, "08_domain_bias_timeseries")

    save_table(bias_df[bias_cols].agg(["mean", "std"]).round(4), "08_domain_bias_stats")

    # ---- 9. Seasonal climatology ----
    clim_df = ts_df.copy()
    clim_df["month"] = clim_df["time"].dt.month
    monthly_clim = clim_df.groupby("month")[["truth", "stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label, color, ls in [
        ("truth", "Truth", "black", "-"), ("stochastic_unet_mean", "Stochastic UNet Mean", "tab:blue", "-"),
        ("deterministic_unet", "Deterministic UNet", "tab:orange", "-"), ("bilinear", "Bilinear", "tab:green", "--"),
        ("piomas", "PIOMAS (obs)", "tab:red", ":"),
    ]:
        if monthly_clim[col].notna().sum() == 0:
            continue  # e.g. PIOMAS has zero overlap with this run's test period
        ax.plot(monthly_clim.index, monthly_clim[col], label=label, color=color, linestyle=ls, linewidth=1.5, marker="o", markersize=3)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Domain-mean SIT climatology (m)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "09_seasonal_climatology")

    amplitude_df = (monthly_clim.max() - monthly_clim.min()).rename("Seasonal amplitude (m)").round(4).to_frame()
    save_table(amplitude_df, "09_seasonal_amplitude")

    # ---- 10. Anomaly correlation ----
    months = sample_times_df["time"].dt.month.values
    all_idx = np.arange(len(months))
    truth_anom = remove_monthly_climatology(Y_test_phys, months)
    raw_maps, acc_maps = {}, {}
    for label, field in method_fields.items():
        raw_maps[label] = np.where(land_hw, np.nan, temporal_pixel_corr(field, Y_test_phys, all_idx))
        pred_anom = remove_monthly_climatology(field, months)
        acc_maps[label] = np.where(land_hw, np.nan, temporal_pixel_corr(pred_anom, truth_anom, all_idx))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
    for ax, (label, amap) in zip(axs, acc_maps.items()):
        im = ax.pcolormesh(tgt_lon, tgt_lat, amap, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        _style(ax, tgt_lon, tgt_lat)
        ax.set_title(label, fontsize=13)
    cbar = fig.colorbar(im, ax=axs, aspect=30, shrink=0.8, pad=0.02)
    cbar.set_label("Anomaly correlation (deseasonalized), all months", fontsize=12)
    save_fig(fig, "10_anomaly_corr_maps")

    ocean_w3 = np.where(land_hw, 0.0, np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon))))
    ocean_w3 = ocean_w3 / ocean_w3.sum()
    acc_summary = [{
        "Method": label,
        "Raw temporal corr., all months (cos-lat weighted)": round(float(np.nansum(ocean_w3 * raw_maps[label])), 4),
        "Anomaly corr., all months (cos-lat weighted)": round(float(np.nansum(ocean_w3 * acc_maps[label])), 4),
    } for label in method_fields]
    save_table(pd.DataFrame(acc_summary), "10_anomaly_corr_summary")

    # ---- 11. Taylor diagram ----
    def weighted_stats(pred, truth, weight_hw):
        N = pred.shape[0]
        w = (weight_hw / N)[None, :, :]
        p, t = pred[:, 0], truth[:, 0]
        p_mean, t_mean = np.nansum(w * p), np.nansum(w * t)
        p_anom, t_anom = p - p_mean, t - t_mean
        var_p, var_t = np.nansum(w * p_anom ** 2), np.nansum(w * t_anom ** 2)
        cov = np.nansum(w * p_anom * t_anom)
        R = cov / np.sqrt(var_p * var_t)
        std_ratio = np.sqrt(var_p / var_t)
        crmse = np.sqrt(np.nansum(w * (p_anom - t_anom) ** 2))
        return float(R), float(std_ratio), float(crmse)

    ocean_w_hw = np.where(land_hw, 0.0, np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon))))
    ocean_w_hw = ocean_w_hw / ocean_w_hw.sum()
    taylor_stats = {label: weighted_stats(field, Y_test_phys, ocean_w_hw) for label, field in method_fields.items()}

    def rms_circle(center, radius, theta_lim=(0, np.pi / 2), n=200):
        t = np.linspace(0, 2 * np.pi, n)
        x, y = center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)
        theta, r = np.arctan2(y, x), np.sqrt(x ** 2 + y ** 2)
        valid = (theta >= theta_lim[0]) & (theta <= theta_lim[1])
        return theta[valid], r[valid]

    max_std = max(sr for _, sr, _ in taylor_stats.values())
    r_lim = max(1.3, max_std * 1.2)
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, r_lim)
    corr_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_xticks(np.arccos(corr_ticks))
    ax.set_xticklabels([str(c) for c in corr_ticks])
    ax.set_xlabel("Correlation", labelpad=10)
    ax.set_ylabel("Normalized std. dev. (relative to Truth)", labelpad=30)
    for rms in np.arange(0.5, r_lim, 0.5):
        th, r = rms_circle((1, 0), rms)
        ax.plot(th, r, color="gray", linestyle=":", linewidth=0.8)
    std_circle_theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(std_circle_theta, np.ones_like(std_circle_theta), color="black", linestyle="--", linewidth=0.8)
    ax.plot([0], [1], marker="*", color="black", markersize=16, linestyle="none", label="Truth (reference)")
    colors = {"Bilinear": "tab:green", "Deterministic UNet": "tab:orange", "Stochastic UNet Mean": "tab:blue"}
    for label, (R, std_ratio, crmse) in taylor_stats.items():
        ax.plot([np.arccos(np.clip(R, -1, 1))], [std_ratio], marker="o", markersize=10,
                color=colors.get(label, "tab:red"), linestyle="none", label=label)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    save_fig(fig, "11_taylor_diagram")

    taylor_df = pd.DataFrame([
        {"Method": label, "Correlation (R)": round(R, 4), "Std Ratio": round(std_ratio, 4), "Centered RMSE (m)": round(crmse, 4)}
        for label, (R, std_ratio, crmse) in taylor_stats.items()
    ])
    save_table(taylor_df, "11_taylor_stats")


    # ---- 12. PIOMAS metrics & spatial snapshot ----
    piomas_valid = ~np.isnan(piomas_regridded_phys[:, 0]).all(axis=(1, 2))
    n_piomas_valid = int(piomas_valid.sum())

    def weighted_mae_rmse_bias(pred_phys, ref_phys, weight_hw, valid):
        if valid.sum() == 0:
            return np.nan, np.nan, np.nan
        diff = pred_phys[valid, 0] - ref_phys[valid, 0]
        w = weight_hw[None, :, :]
        mae = float(np.nansum(w * np.abs(diff)) / valid.sum())
        rmse = float(np.sqrt(np.nansum(w * diff ** 2) / valid.sum()))
        bias = float(np.nansum(w * diff) / valid.sum())
        return mae, rmse, bias

    if n_piomas_valid == 0:
        print("  No test samples overlap PIOMAS's record -- skipping PIOMAS metrics/snapshot.")
    else:
        methods_vs_piomas = {
            "Truth": Y_test_phys, "Bilinear": Y_base_phys,
            "Deterministic UNet": Y_pred_det_phys, "Stochastic UNet Mean": Y_pred_phys,
        }
        piomas_metrics = []
        for label, field in methods_vs_piomas.items():
            mae, rmse, bias = weighted_mae_rmse_bias(field, piomas_regridded_phys, ocean_w_hw, piomas_valid)
            piomas_metrics.append({
                "Method": label, "MAE vs. PIOMAS (m)": round(mae, 4),
                "RMSE vs. PIOMAS (m)": round(rmse, 4), "Bias vs. PIOMAS (m)": round(bias, 4),
            })
        save_table(pd.DataFrame(piomas_metrics), "12_piomas_metrics")

        valid_idxs = np.where(piomas_valid)[0]
        if int(SAMPLE_IDXS[0]) in valid_idxs:
            piomas_sample_idx = int(SAMPLE_IDXS[0])
        else:
            ice_valid = valid_idxs[(Y_test_phys[valid_idxs, 0] > 0.5).mean(axis=(1, 2)) > 0.25]
            piomas_sample_idx = int(ice_valid[0]) if len(ice_valid) else int(valid_idxs[0])
        panel_fields = [Y_test_phys[piomas_sample_idx, 0], Y_pred_phys[piomas_sample_idx, 0], piomas_regridded_phys[piomas_sample_idx, 0]]
        panel_titles_piomas = ["Truth", "Ensemble Mean", "PIOMAS (obs, regridded)"]
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
        for ax, field, title in zip(axs, panel_fields, panel_titles_piomas):
            im = ax.pcolormesh(tgt_lon, tgt_lat, field, transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=3, shading="auto")
            _style(ax, tgt_lon, tgt_lat)
            ax.set_title(title, fontsize=13)
        sample_time = sample_times_df["time"].iloc[piomas_sample_idx]
        fig.suptitle(f"Sample {piomas_sample_idx} ({sample_time:%Y-%m})", fontsize=12)
        cbar = fig.colorbar(im, ax=axs, aspect=30, shrink=0.8, pad=0.02)
        cbar.set_label("Sea ice thickness (m)", fontsize=13)
        save_fig(fig, "12_piomas_spatial_snapshot")

    # ---- 13. Power spectral density comparison ----
    ocean_hw = ~land_hw
    dlat_deg = float(np.diff(tgt_lat).mean())
    dlon_deg = float(np.diff(tgt_lon).mean())
    mean_lat_deg = float(tgt_lat.mean())
    N_PSD_BINS = 40

    def isotropic_psd(field_phys, ocean_hw, dlat_deg, dlon_deg, mean_lat_deg, n_bins=N_PSD_BINS):
        N, H, W = field_phys.shape[0], field_phys.shape[2], field_phys.shape[3]
        km_per_deg = 111.0
        dy = km_per_deg * dlat_deg
        dx = km_per_deg * dlon_deg * np.cos(np.deg2rad(mean_lat_deg))
        window = np.outer(np.hanning(H), np.hanning(W))
        ky = np.fft.fftfreq(H, d=dy)
        kx = np.fft.fftfreq(W, d=dx)
        kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        k_nyq = min(ky.max(), kx.max())
        bin_edges = np.linspace(0, k_nyq, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_idx = np.clip(np.digitize(kr.ravel(), bin_edges) - 1, 0, n_bins - 1)
        psd_samples = np.full((N, n_bins), np.nan)
        for i in range(N):
            f = field_phys[i, 0].astype(np.float64).copy()
            f[ocean_hw] -= f[ocean_hw].mean()
            f[~ocean_hw] = 0.0
            spec = np.abs(np.fft.fft2(f * window)) ** 2 / (H * W)
            flat_spec = spec.ravel()
            for b in range(n_bins):
                sel = bin_idx == b
                if sel.any():
                    psd_samples[i, b] = flat_spec[sel].mean()
        return bin_centers, np.nanmean(psd_samples, axis=0)

    psd_fields = {
        "Truth": Y_test_phys, "Stochastic UNet Mean": Y_pred_phys,
        "Deterministic UNet": Y_pred_det_phys, "Bilinear": Y_base_phys,
        "PIOMAS (obs)": piomas_regridded_phys,
    }
    psd_style = {
        "Truth": ("black", "-"), "Stochastic UNet Mean": ("tab:blue", "-"),
        "Deterministic UNet": ("tab:orange", "-"), "Bilinear": ("tab:green", "--"),
        "PIOMAS (obs)": ("tab:red", ":"),
    }
    psd_results = {}
    for label, field in psd_fields.items():
        wavenumber, psd = isotropic_psd(field, ocean_hw, dlat_deg, dlon_deg, mean_lat_deg)
        psd_results[label] = psd

    psd_df = pd.DataFrame({
        "wavenumber_cycles_per_km": wavenumber,
        "wavelength_km": 1.0 / np.where(wavenumber > 0, wavenumber, np.nan),
    })
    for label, psd in psd_results.items():
        psd_df[label] = psd
    save_table(psd_df, "13_psd_data")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, psd in psd_results.items():
        color, ls = psd_style[label]
        valid = (wavenumber > 0) & (psd > 0)
        ax.plot(wavenumber[valid], psd[valid], label=label, color=color, linestyle=ls, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber (cycles/km)")
    ax.set_ylabel("Isotropic PSD (m$^2$, arb. spectral units)")
    ax.set_title("Domain-wide power spectral density, averaged over test record")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "13_psd_comparison")

    # ---- 13b. Per-member PSD spread -- is the ensemble's spatial texture calibrated? ----
    # The comparison above uses the ensemble MEAN field, so it only tests whether the central
    # estimate is realistically sharp -- it says nothing about the ensemble's distributional
    # properties. This computes the same isotropic PSD separately for every stochastic draw and
    # asks whether the *spread* of member spectra brackets truth's spectrum at each wavenumber --
    # a spectral-domain analog of the physical-domain spread-skill check above. Uses bincount
    # instead of isotropic_psd's per-bin boolean-mask loop (N x K calls here vs. N there) purely
    # for speed.
    def isotropic_psd_per_member(preds_all_hwk, ocean_hw, dlat_deg, dlon_deg, mean_lat_deg, n_bins=N_PSD_BINS):
        N, K, H, W = preds_all_hwk.shape
        km_per_deg = 111.0
        dy = km_per_deg * dlat_deg
        dx = km_per_deg * dlon_deg * np.cos(np.deg2rad(mean_lat_deg))
        window = np.outer(np.hanning(H), np.hanning(W))
        ky = np.fft.fftfreq(H, d=dy)
        kx = np.fft.fftfreq(W, d=dx)
        kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        k_nyq = min(ky.max(), kx.max())
        bin_edges = np.linspace(0, k_nyq, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_idx_flat = np.clip(np.digitize(kr.ravel(), bin_edges) - 1, 0, n_bins - 1)
        bin_counts = np.bincount(bin_idx_flat, minlength=n_bins).astype(np.float64)
        bin_counts[bin_counts == 0] = np.nan

        psd_samples = np.full((N, K, n_bins), np.nan)
        for i in range(N):
            for k in range(K):
                f = preds_all_hwk[i, k].astype(np.float64).copy()
                f[ocean_hw] -= f[ocean_hw].mean()
                f[~ocean_hw] = 0.0
                spec = np.abs(np.fft.fft2(f * window)) ** 2 / (H * W)
                sums = np.bincount(bin_idx_flat, weights=spec.ravel(), minlength=n_bins)
                psd_samples[i, k] = sums / bin_counts
        return bin_centers, psd_samples  # (n_bins,), (N, K, n_bins)

    _, member_psd_samples = isotropic_psd_per_member(
        preds_all_phys[:, :, 0], ocean_hw, dlat_deg, dlon_deg, mean_lat_deg
    )
    member_psd_time_mean = np.nanmean(member_psd_samples, axis=0)  # (K, n_bins)

    psd_spread_df = pd.DataFrame({
        "wavenumber_cycles_per_km": wavenumber,
        "wavelength_km": 1.0 / np.where(wavenumber > 0, wavenumber, np.nan),
        "Truth": psd_results["Truth"],
        "Member mean": member_psd_time_mean.mean(axis=0),
        "Member p10": np.percentile(member_psd_time_mean, 10, axis=0),
        "Member p90": np.percentile(member_psd_time_mean, 90, axis=0),
    })
    psd_spread_df["Relative member spread pct"] = (
        (psd_spread_df["Member p90"] - psd_spread_df["Member p10"]) / psd_spread_df["Member mean"] * 100
    )
    save_table(psd_spread_df, "13b_psd_per_member_spread_data")

    # Two panels: the absolute-PSD view (left) plus a relative-spread view (right). The p10-p90
    # band IS being drawn on the left panel (fill_between never silently no-ops) -- but on a
    # log-log absolute-PSD axis, a band that's genuinely only ~0.5-1.5% of the mean value (this
    # project's actual member-to-member spectral texture, confirmed by direct inspection of the
    # underlying percentiles) renders thinner than the line stroke itself, so it reads as "no
    # spread" even though it's really there. The right panel makes that same (still small)
    # relative spread legible on its own linear axis instead of relying on visibility against
    # the absolute PSD's many-orders-of-magnitude range.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    valid = (wavenumber > 0) & (psd_spread_df["Truth"] > 0)
    ax1.plot(wavenumber[valid], psd_spread_df["Truth"][valid], label="Truth", color="black", linewidth=1.8)
    ax1.fill_between(wavenumber[valid], psd_spread_df["Member p10"][valid], psd_spread_df["Member p90"][valid],
                      color="tab:blue", alpha=0.3, label="Stochastic UNet members (10th-90th pct)")
    ax1.plot(wavenumber[valid], psd_spread_df["Member mean"][valid], color="tab:blue", linewidth=1.2, linestyle="--",
             label="Stochastic UNet member mean")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Wavenumber (cycles/km)")
    ax1.set_ylabel("Isotropic PSD (m$^2$, arb. spectral units)")
    ax1.set_title("Per-member PSD spread vs. truth")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(wavenumber[valid], psd_spread_df["Relative member spread pct"][valid], color="tab:blue", linewidth=1.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Wavenumber (cycles/km)")
    ax2.set_ylabel("Relative member spread,\n(p90 - p10) / mean (%)")
    ax2.set_title("Inter-member spectral-texture spread\n(near-zero = members look nearly identical)")
    ax2.grid(True, which="both", alpha=0.3)
    fig.suptitle("Per-member PSD spread -- calibration of spatial texture", y=1.02)
    plt.tight_layout()
    save_fig(fig, "13b_psd_per_member_spread")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True, help="Path to a results/<BATCH_NAME>/ directory")
    parser.add_argument("--save-root", default="evaluation/saved_figs", help="Root dir for output figs/tables (default: evaluation/saved_figs)")
    parser.add_argument("--only", default=None, help="Only process this one run (subdirectory name), for testing")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    run_dirs = sorted(d for d in batch_dir.iterdir() if (d / "eval_data").exists())
    if args.only:
        run_dirs = [d for d in run_dirs if d.name == args.only]
        if not run_dirs:
            raise SystemExit(f"--only={args.only!r} not found (with eval_data/) under {batch_dir}")

    print(f"Found {len(run_dirs)} run(s) with eval_data/ under {batch_dir}:")
    for d in run_dirs:
        print(" ", d.name)
    if not run_dirs:
        print("Nothing to do.")
        return

    for run_dir in run_dirs:
        run_save_dir = Path(args.save_root) / batch_dir.name / run_dir.name
        print(f"\n--- {run_dir.name} -> {run_save_dir} ---", flush=True)
        run_all_sections(run_dir / "eval_data", run_save_dir, show=False)

    print("\nBatch done.")


if __name__ == "__main__":
    main()
