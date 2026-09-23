#!/usr/bin/env python3
"""
build_4window_aggregate_figures.py

All 4 runs in a recommended-config-style batch (results/<batch>/) train on different
years but evaluate on the *same* test year (2021) -- so a genuine 4-window-pooled figure
is more representative for the paper than picking one window arbitrarily, and matches
the "4-window average" convention every table in recommended_config.md already uses for
scalar metrics. This does the same thing for the distributional-evaluation *figures*
(rank histogram, reliability diagram, spread-skill by regime, CRPS by regime, PSD).

Pools the four windows' raw per-cell ensemble/truth data via additive sufficient
statistics (bincounts, sums, counts) accumulated one window at a time -- NOT by loading
all 4 windows' preds_all_phys into memory simultaneously (four ~4-5GB arrays at once
would be a real OOM risk on a shared node) and NOT by naively averaging the four windows'
already-computed summary numbers (which can distort nonlinear quantities like Brier
scores). Mathematically equivalent to true pooling for every metric here, since bincount/
sum/count are inherently additive.

PSD (section 13/13b) is the one exception: it's already a mean over samples *within* a
window, so a further unweighted mean of the 4 windows' saved PSD curves is a valid
additional averaging step and much cheaper than re-deriving it from raw FFTs.

Usage:
    python evaluation/build_4window_aggregate_figures.py \\
        --batch-dir results/MESA_stochastic_refine_sweep_avg_sharedbias \\
        --save-dir evaluation/saved_figs/MESA_stochastic_refine_sweep_avg_sharedbias/_4window_aggregate
"""
import argparse
import glob
import json
import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Sibling module, same directory -- reuses its plotting helpers (make_polar_proj,
# style_polar_ax, temporal_pixel_corr, remove_monthly_climatology, build_piomas_regridded,
# rolling_window_samples) and constants (ROLLING_DAYS_*, POINT_HOPE_*) instead of
# re-deriving them here, so the two files' math can't quietly drift apart. Safe to import:
# everything in run_daily_eval_batch.py outside run_all_sections()/main() is a plain
# function/constant definition, no side effects at import time.
import run_daily_eval_batch as rdeb

SIT_BIN_EDGES = [0.0, 0.15, 0.5, 1.0, np.inf]
SIT_BIN_LABELS = ["Open water/frazil (0-0.15 m)", "Thin ice (0.15-0.5 m)", "Moderate ice (0.5-1.0 m)", "Thick ice (>1.0 m)"]
RELIABILITY_THRESHOLDS = [0.15, 0.5, 1.0]
N_PROB_BINS = 10


def accumulate_window(run_dir, acc, seed):
    print(f"    loading fields.npz...", flush=True)
    fields = np.load(os.path.join(run_dir, "eval_data", "fields.npz"))
    preds_all_phys = fields["preds_all_phys"]
    print(f"    preds_all_phys loaded, shape={preds_all_phys.shape}, "
          f"{preds_all_phys.nbytes / 1e9:.2f} GB", flush=True)
    Y_test_phys = fields["Y_test_phys"]
    mask_test = fields["mask_test"]

    ocean_bool = mask_test[:, 0] <= 0.5
    ens_hwk = np.ascontiguousarray(np.moveaxis(preds_all_phys[:, :, 0], 1, -1))
    print(f"    ens_hwk built, shape={ens_hwk.shape}, {ens_hwk.nbytes / 1e9:.2f} GB", flush=True)
    K = ens_hwk.shape[-1]
    ens_ocean = ens_hwk[ocean_bool]  # (M, K)
    del ens_hwk, preds_all_phys
    truth_ocean = Y_test_phys[:, 0][ocean_bool]  # (M,)
    acc["K"] = K
    print(f"    ens_ocean built, shape={ens_ocean.shape}, {ens_ocean.nbytes / 1e9:.2f} GB", flush=True)

    # ---- Rank histogram: accumulate rank_counts (additive) ----
    print(f"    computing rank histogram...", flush=True)
    rng = np.random.default_rng(seed)
    below = (ens_ocean < truth_ocean[:, None]).sum(axis=1)
    ties = (ens_ocean == truth_ocean[:, None]).sum(axis=1)
    ranks = below + rng.integers(0, ties + 1)
    rank_counts = np.bincount(ranks, minlength=K + 1)
    acc["rank_counts"] = acc.get("rank_counts", np.zeros(K + 1, dtype=np.int64)) + rank_counts
    # Per-regime rank histograms: the pooled histogram is dominated by whichever regime has the
    # most cells (open water), which can hide opposite-signed miscalibration in the others.
    # Regimes for calibration diagnostics are assigned by ensemble-mean SIT, not truth:
    # conditioning on the observation distorts rank histograms/spread-skill even for a
    # perfectly calibrated ensemble (e.g. spurious rank-K excess in the thick bin).
    bin_idx_sit = np.digitize(truth_ocean, SIT_BIN_EDGES[1:-1])
    bin_idx_fc = np.digitize(ens_ocean.mean(axis=1), SIT_BIN_EDGES[1:-1])
    rank_by_regime = acc.setdefault("rank_counts_by_regime", np.zeros((len(SIT_BIN_LABELS), K + 1), dtype=np.int64))
    for b in range(len(SIT_BIN_LABELS)):
        rank_by_regime[b] += np.bincount(ranks[bin_idx_fc == b], minlength=K + 1)

    # ---- Reliability diagram: accumulate per-bin sums/counts/sq-error per threshold ----
    print(f"    computing reliability diagram...", flush=True)
    bin_edges = np.linspace(0, 1, N_PROB_BINS + 1)
    for thr in RELIABILITY_THRESHOLDS:
        p_forecast = (ens_ocean > thr).mean(axis=1)
        outcome = (truth_ocean > thr).astype(np.float64)
        bin_idx = np.clip(np.digitize(p_forecast, bin_edges[1:-1]), 0, N_PROB_BINS - 1)
        key = f"reliab_{thr}"
        d = acc.setdefault(key, {
            "count": np.zeros(N_PROB_BINS, dtype=np.int64),
            "sum_fc": np.zeros(N_PROB_BINS), "sum_obs": np.zeros(N_PROB_BINS),
            "sq_err_sum": 0.0, "n_total": 0, "outcome_sum": 0.0,
        })
        for b in range(N_PROB_BINS):
            sel = bin_idx == b
            d["count"][b] += int(sel.sum())
            d["sum_fc"][b] += float(p_forecast[sel].sum())
            d["sum_obs"][b] += float(outcome[sel].sum())
        d["sq_err_sum"] += float(np.sum((p_forecast - outcome) ** 2))
        d["n_total"] += len(outcome)
        d["outcome_sum"] += float(outcome.sum())

    # ---- Spread-skill by regime: accumulate N/sum_spread/sum_error per regime ----
    print(f"    computing spread-skill by regime...", flush=True)
    ens_mean_ocean = ens_ocean.mean(axis=1)
    ens_spread_ocean = ens_ocean.std(axis=1, ddof=1)
    abs_error_ocean = np.abs(ens_mean_ocean - truth_ocean)
    n_rows = len(SIT_BIN_LABELS) + 1  # last row = all ocean cells
    ss = acc.setdefault("spread_skill", {
        k: np.zeros(n_rows) for k in ["n", "sum_truth", "sum_spread", "sum_error", "sum_var", "sum_sq_error", "sum_signed_error"]
    })
    for b in range(n_rows):
        sel = (bin_idx_fc == b) if b < len(SIT_BIN_LABELS) else slice(None)
        ss["n"][b] += int(truth_ocean[sel].size)
        ss["sum_truth"][b] += float(truth_ocean[sel].sum())
        ss["sum_spread"][b] += float(ens_spread_ocean[sel].sum())
        ss["sum_error"][b] += float(abs_error_ocean[sel].sum())
        ss["sum_var"][b] += float((ens_spread_ocean[sel] ** 2).sum())
        ss["sum_sq_error"][b] += float(((ens_mean_ocean[sel] - truth_ocean[sel]) ** 2).sum())
        ss["sum_signed_error"][b] += float((ens_mean_ocean[sel] - truth_ocean[sel]).sum())

    # ---- CRPS by regime: accumulate N/sum_crps per regime + domain-wide ----
    print(f"    computing CRPS by regime...", flush=True)
    K_crps = ens_ocean.shape[1]
    crps_term1 = np.abs(ens_ocean - truth_ocean[:, None]).mean(axis=1)
    sorted_ens = np.sort(ens_ocean, axis=1)
    crps_weights = (2 * np.arange(1, K_crps + 1) - K_crps - 1)
    crps_term2 = (sorted_ens * crps_weights[None, :]).sum(axis=1) / (K_crps ** 2)
    crps_per_sample = crps_term1 - crps_term2
    crps_acc = acc.setdefault("crps", {
        "n_all": 0, "sum_all": 0.0,
        "n": np.zeros(len(SIT_BIN_LABELS), dtype=np.int64), "sum": np.zeros(len(SIT_BIN_LABELS)),
    })
    crps_acc["n_all"] += len(crps_per_sample)
    crps_acc["sum_all"] += float(crps_per_sample.sum())
    for b in range(len(SIT_BIN_LABELS)):
        sel = bin_idx_sit == b
        crps_acc["n"][b] += int(sel.sum())
        crps_acc["sum"][b] += float(crps_per_sample[sel].sum())


def load_window_full(run_dir):
    """Load everything one window's eval_data/ has for the full-period (non-distributional)
    sections -- mirrors run_all_sections()'s own "Load everything" block."""
    eval_dir = os.path.join(run_dir, "eval_data")
    fields = np.load(os.path.join(eval_dir, "fields.npz"))
    with open(os.path.join(eval_dir, "tile_geometry.pkl"), "rb") as f:
        tile_geometry = pickle.load(f)
    with open(os.path.join(eval_dir, "meta.json")) as f:
        meta = json.load(f)
    sample_times_df = pd.read_csv(os.path.join(eval_dir, "sample_times.csv"), parse_dates=["time"])
    return fields, tile_geometry, meta, sample_times_df


def pool_full_period_sections(run_dirs, save_dir):
    """Pools sections 1, 2, 5, 6, 8, 9, 10, 11, and 12's numeric PIOMAS table across all
    windows in run_dirs. Unlike the distributional sections above (rank histogram, etc.),
    which pool raw per-cell/per-sample data via additive sufficient statistics, these
    sections are each already a full-test-period statistic *per window* (a time series, a
    time-mean map, a single Taylor-diagram point) -- so the pooling here is the same
    approach already used for PSD above: recompute each window's own statistic exactly as
    run_all_sections() does, then average the four windows' results. Truth, the bilinear
    baseline, the land mask/grid, and the PIOMAS regrid are identical across windows (they
    don't depend on which years the model trained on), so only the *model-dependent* fields
    (Deterministic UNet, Stochastic UNet Mean) actually differ window to window and need
    averaging; the rest are taken once from the first window.

    Deliberately excludes the spatial-snapshot figures (sections 3, 4, and 12's map panel),
    which show a single date's field -- per the existing project convention (see
    recommended_config.md), pixel-averaging four differently-trained models' single-day maps
    would blur real structure rather than usefully summarize it, so a representative single
    window is used for those instead, unchanged by this function. Section 7 (trend maps) is
    also skipped: it requires >= 5 years of test data (MIN_YEARS_FOR_TREND in
    run_daily_eval_batch.py) and this project's runs all use a single test year (2021), so no
    window ever produces trend output to pool in the first place.
    """
    def save_fig(fig, name):
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_table(df, name):
        df.to_csv(os.path.join(save_dir, f"{name}.csv"))

    fields0, tile_geometry0, meta0, sample_times_df0 = load_window_full(run_dirs[0])
    if meta0["use_patches"]:
        print("  use_patches=True -- sections 1/2/5-11 need one coherent domain, skipping full-period pooling.")
        return

    Y_test_phys = fields0["Y_test_phys"]
    Y_base_phys = fields0["Y_base_phys"]
    mask_test = fields0["mask_test"]
    land_mask = fields0["land_mask"]
    bbox = meta0["bbox"]
    candidate_points = {**meta0["candidate_points"], "Point Hope": {"lat": rdeb.POINT_HOPE_LAT, "lon": rdeb.POINT_HOPE_LON_360}}
    geo0 = tile_geometry0[0]
    tgt_lat, tgt_lon = geo0["target_lat"], geo0["target_lon"]
    land_hw = mask_test[0, 0] > 0.5

    proj, boundary_path, _ = rdeb.make_polar_proj(bbox)

    def _style(ax, lon_=None, lat_=None, points=None):
        rdeb.style_polar_ax(ax, proj, boundary_path, bbox, candidate_points, lon_, lat_, points=points)

    print("  building pooled PIOMAS regrid (identical across windows, computed once)...", flush=True)
    piomas_regridded_phys = rdeb.build_piomas_regridded(tgt_lat, tgt_lon, sample_times_df0["time"])

    ocean_weight = (1.0 - mask_test[0, 0]).clip(0, 1)
    weight_sum = ocean_weight.sum()

    def domain_mean(field_phys):
        return (field_phys[:, 0] * ocean_weight[None, :, :]).sum(axis=(1, 2)) / weight_sum

    ocean_w_hw = np.where(land_hw, 0.0, np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon))))
    ocean_w_hw = ocean_w_hw / ocean_w_hw.sum()

    print("  loading per-window model-dependent fields (Deterministic/Stochastic UNet)...", flush=True)
    per_window = []
    for d in run_dirs:
        f, _, _, stdf = load_window_full(d)
        per_window.append({"Y_pred_det_phys": f["Y_pred_det_phys"], "Y_pred_phys": f["Y_pred_phys"],
                            "sample_times_df": stdf})
    n_win = len(per_window)

    # ============ Section 1: domain-mean SIT time series ============
    print("  section 1: domain-mean SIT time series...", flush=True)
    # Built in sample_times_df0's own (unsorted) row order first, so each computed column
    # stays position-aligned with the raw per-sample arrays (Y_test_phys etc.) it was
    # computed from -- sorting *before* attaching these columns (the original approach)
    # silently detached the sorted "time" axis from the still-originally-ordered values.
    # For MESA that original order is per-member blocks (member 0's full year, then member
    # 1's, ...), so the old code plotted each member's full annual cycle back-to-back
    # compressed across the real one-year x-axis -- visible as several repeated seasonal
    # cycles within a single nominal year. Grouping by "time" and averaging (real intent of
    # a "domain-mean" plot -- one curve per method per date) both fixes the alignment and
    # collapses MESA's 6 member-samples-per-date into the single averaged value the plot
    # was always meant to show.
    raw_ts_df = sample_times_df0.copy()
    raw_ts_df["truth"] = domain_mean(Y_test_phys)
    raw_ts_df["bilinear"] = domain_mean(Y_base_phys)
    raw_ts_df["piomas"] = domain_mean(piomas_regridded_phys)
    raw_ts_df["deterministic_unet"] = np.mean([domain_mean(w["Y_pred_det_phys"]) for w in per_window], axis=0)
    raw_ts_df["stochastic_unet_mean"] = np.mean([domain_mean(w["Y_pred_phys"]) for w in per_window], axis=0)
    ts_value_cols = ["truth", "bilinear", "piomas", "deterministic_unet", "stochastic_unet_mean"]
    ts_df = raw_ts_df.groupby("time", as_index=False)[ts_value_cols].mean().sort_values("time").reset_index(drop=True)
    save_table(ts_df, "01_domain_mean_sit_timeseries_data_4window")

    ts_plot_cols = ["truth", "stochastic_unet_mean", "deterministic_unet", "bilinear", "piomas"]
    ts_plot_df = ts_df.copy()
    if rdeb.ROLLING_DAYS_TIMESERIES:
        win = rdeb.rolling_window_samples(ts_plot_df["time"], rdeb.ROLLING_DAYS_TIMESERIES)
        ts_plot_df[ts_plot_cols] = ts_plot_df[ts_plot_cols].rolling(win, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, col, color, ls in [
        ("Truth", "truth", "black", "-"), ("Stochastic UNet Mean", "stochastic_unet_mean", "tab:blue", "-"),
        ("Deterministic UNet", "deterministic_unet", "tab:orange", "-"), ("Bilinear", "bilinear", "tab:green", "--"),
        ("PIOMAS (obs)", "piomas", "tab:red", ":"),
    ]:
        if ts_plot_df[col].notna().sum() == 0:
            continue
        ax.plot(ts_plot_df["time"], ts_plot_df[col], label=label, color=color, ls=ls, lw=2 if col == "truth" else 1.5)
    ax.set_ylabel("Domain-mean SIT (m)"); ax.set_xlabel("Time"); ax.legend(fontsize=9)
    fig.autofmt_xdate(); plt.tight_layout()
    save_fig(fig, "01_domain_mean_sit_timeseries_4window")

    # ============ Section 1b: domain-mean pattern correlation time series ============
    print("  section 1b: domain-mean pattern correlation time series...", flush=True)
    def spatial_pattern_corr_series(pred_phys, truth_phys, weight_hw):
        p, t = pred_phys[:, 0], truth_phys[:, 0]
        w = weight_hw[None, :, :]
        p_mean = (w * p).sum(axis=(1, 2))
        t_mean = (w * t).sum(axis=(1, 2))
        p_anom, t_anom = p - p_mean[:, None, None], t - t_mean[:, None, None]
        num = (w * p_anom * t_anom).sum(axis=(1, 2))
        den = np.sqrt((w * p_anom ** 2).sum(axis=(1, 2)) * (w * t_anom ** 2).sum(axis=(1, 2)))
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-8)

    # Pattern correlation is nonlinear in the predicted field, unlike section 1's domain mean --
    # so (matching section 11's Taylor-diagram convention for the same reason) the model-dependent
    # methods are correlated *per window first*, then averaged across windows, rather than
    # averaging the raw predicted fields across windows before correlating. Computed at the raw
    # (unsorted, per-member-block) sample order first, same as section 1, then grouped by "time"
    # to collapse MESA's multi-member rows per date down to one value per date.
    raw_pc_df = sample_times_df0.copy()
    raw_pc_df["bilinear"] = spatial_pattern_corr_series(Y_base_phys, Y_test_phys, ocean_w_hw)
    raw_pc_df["piomas"] = spatial_pattern_corr_series(piomas_regridded_phys, Y_test_phys, ocean_w_hw)
    det_pc_per_win = np.stack([spatial_pattern_corr_series(w["Y_pred_det_phys"], Y_test_phys, ocean_w_hw) for w in per_window])
    sto_pc_per_win = np.stack([spatial_pattern_corr_series(w["Y_pred_phys"], Y_test_phys, ocean_w_hw) for w in per_window])
    raw_pc_df["deterministic_unet"] = np.nanmean(det_pc_per_win, axis=0)
    raw_pc_df["stochastic_unet_mean"] = np.nanmean(sto_pc_per_win, axis=0)
    pc_value_cols = ["bilinear", "piomas", "deterministic_unet", "stochastic_unet_mean"]
    pc_df = raw_pc_df.groupby("time", as_index=False)[pc_value_cols].mean().sort_values("time").reset_index(drop=True)
    save_table(pc_df, "01b_domain_mean_pattern_corr_timeseries_data_4window")

    pc_plot_df = pc_df.copy()
    if rdeb.ROLLING_DAYS_TIMESERIES:
        win = rdeb.rolling_window_samples(pc_plot_df["time"], rdeb.ROLLING_DAYS_TIMESERIES)
        pc_plot_df[pc_value_cols] = pc_plot_df[pc_value_cols].rolling(win, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, col, color, ls in [
        ("Stochastic UNet Mean", "stochastic_unet_mean", "tab:blue", "-"),
        ("Deterministic UNet", "deterministic_unet", "tab:orange", "-"),
        ("Bilinear", "bilinear", "tab:green", "--"),
        ("PIOMAS (obs)", "piomas", "tab:red", ":"),
    ]:
        if pc_plot_df[col].notna().sum() == 0:
            continue
        ax.plot(pc_plot_df["time"], pc_plot_df[col], label=label, color=color, ls=ls, linewidth=1.5)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_ylabel("Domain-mean pattern correlation vs. Truth, pooled across 4 windows")
    ax.set_xlabel("Time")
    ax.set_ylim(top=1.02)
    ax.legend(fontsize=9)
    fig.autofmt_xdate(); plt.tight_layout()
    save_fig(fig, "01b_domain_mean_pattern_corr_timeseries_4window")

    # ============ Section 2: candidate-point time series ============
    print("  section 2: candidate-point time series...", flush=True)
    FLAT_STD_TOL = 1e-6
    flat_truth_hw = Y_test_phys[:, 0].std(axis=0) < FLAT_STD_TOL
    flat_piomas_hw = np.nanstd(piomas_regridded_phys[:, 0], axis=0) < FLAT_STD_TOL
    exclude_hw = (land_hw) | flat_truth_hw | flat_piomas_hw
    # Raw (unsorted) time index, position-aligned with the per-sample field arrays below --
    # see section 1's comment above for why zipping against the *sorted* ts_df["time"]
    # (the original approach) was wrong, and would additionally now be a hard length
    # mismatch (365 vs. 2190) since section 1's ts_df is grouped/averaged.
    raw_time_index = sample_times_df0["time"]
    rows = []
    for point_name, pt in candidate_points.items():
        iy, ix, dist_km = rdeb.nearest_valid_index(tgt_lat, tgt_lon, exclude_hw, pt["lat"], pt["lon"])
        fields_phys = {
            "truth": Y_test_phys, "bilinear": Y_base_phys, "piomas": piomas_regridded_phys,
            "deterministic_unet": np.mean([w["Y_pred_det_phys"] for w in per_window], axis=0),
            "stochastic_unet_mean": np.mean([w["Y_pred_phys"] for w in per_window], axis=0),
        }
        for method_name, field in fields_phys.items():
            values = field[:, 0, iy, ix]
            for t, v in zip(raw_time_index, values):
                rows.append({"point": point_name, "method": method_name, "time": t, "value": float(v),
                             "grid_iy": iy, "grid_ix": ix, "dist_km": dist_km})
    # Average multiple member-samples per (point, method, date) down to one value, same as
    # section 1 -- a candidate-point time series is meant to be one curve per method.
    point_df = (pd.DataFrame(rows)
                .groupby(["point", "method", "time"], as_index=False)
                .agg(value=("value", "mean"), grid_iy=("grid_iy", "first"),
                     grid_ix=("grid_ix", "first"), dist_km=("dist_km", "first"))
                .sort_values(["point", "method", "time"]).reset_index(drop=True))
    save_table(point_df, "02_candidate_point_timeseries_data_4window")

    points_to_plot = list(candidate_points.keys())
    fig, axs = plt.subplots(len(points_to_plot), 1, figsize=(10, 2.6 * len(points_to_plot)), sharex=True)
    if len(points_to_plot) == 1:
        axs = [axs]
    for ax, point_name in zip(axs, points_to_plot):
        sub = point_df[point_df["point"] == point_name]
        dist_km = sub["dist_km"].iloc[0] if len(sub) else np.nan
        for label, col, color, ls, lw in [
            ("Truth", "truth", "black", "-", 1.8), ("Stochastic UNet Mean", "stochastic_unet_mean", "tab:blue", "-", 1.2),
            ("Deterministic UNet", "deterministic_unet", "tab:orange", "-", 1.2),
            ("Bilinear", "bilinear", "tab:green", "--", 1.2), ("PIOMAS (obs)", "piomas", "tab:red", ":", 1.2),
        ]:
            m = sub[sub["method"] == col]
            if m["value"].notna().sum() == 0:
                continue
            y = m["value"]
            if rdeb.ROLLING_DAYS_TIMESERIES:
                y = y.rolling(rdeb.rolling_window_samples(m["time"], rdeb.ROLLING_DAYS_TIMESERIES), center=True, min_periods=1).mean()
            ax.plot(m["time"], y, label=label, color=color, linestyle=ls, linewidth=lw)
        ax.set_title(f"{point_name} (nearest valid ocean grid cell ~{dist_km:.1f} km away), pooled across 4 windows", fontsize=10)
        ax.set_ylabel("SIT (m)")
        if point_name == "Kivalina":
            ax.legend(fontsize=8, loc="upper left", frameon=False)
    axs[-1].set_xlabel("Time")
    fig.autofmt_xdate(); plt.tight_layout()
    save_fig(fig, "02_candidate_point_timeseries_4window")

    # ============ Section 5: pattern-corr maps (March), pooled by averaging each window's map ============
    print("  section 5: pattern-corr maps (March)...", flush=True)
    MONTH = 3
    MIN_SAMPLES = 3
    # Indices into the *raw* per-sample arrays (Y_test_phys, per_window's fields, etc.),
    # which are still in sample_times_df0's original (unsorted, per-member-block) order --
    # ts_df is now grouped/averaged to one row per date (see section 1), so it's the wrong
    # length/order to index these raw arrays with.
    month_idx = np.where(sample_times_df0["time"].dt.month.values == MONTH)[0]
    if len(month_idx) >= MIN_SAMPLES:
        corr_maps_by_method = {"Bilinear": [], "Deterministic UNet": [], "Stochastic UNet Mean": []}
        for w in per_window:
            for label, field in [("Bilinear", Y_base_phys), ("Deterministic UNet", w["Y_pred_det_phys"]),
                                  ("Stochastic UNet Mean", w["Y_pred_phys"])]:
                corr_maps_by_method[label].append(rdeb.temporal_pixel_corr(field, Y_test_phys, month_idx))
        methods_maps = {label: np.where(land_hw, np.nan, np.nanmean(maps, axis=0))
                         for label, maps in corr_maps_by_method.items()}
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
        for ax, (label, corr_map) in zip(axs, methods_maps.items()):
            im = ax.pcolormesh(tgt_lon, tgt_lat, corr_map, transform=rdeb.ccrs.PlateCarree(), cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
            _style(ax, tgt_lon, tgt_lat)
            ax.set_title(label, fontsize=13)
        cbar = fig.colorbar(im, ax=axs, aspect=30, shrink=0.8, pad=0.02)
        cbar.set_label(f"Temporal Pearson corr. vs. truth (month={MONTH}), pooled across 4 windows", fontsize=12)
        save_fig(fig, "05_pattern_corr_maps_4window")

        ocean_w = np.where(land_hw, 0.0, np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon))))
        ocean_w = ocean_w / ocean_w.sum()
        save_table(pd.DataFrame([
            {"Method": label, "Domain mean temporal corr. (cos-lat weighted)": round(float(np.nansum(ocean_w * corr_map)), 4)}
            for label, corr_map in methods_maps.items()
        ]), "05_pattern_corr_summary_4window")

    # ============ Section 6: bias & std-ratio maps ============
    print("  section 6: bias & std-ratio maps...", flush=True)
    truth_mean_hw = Y_test_phys[:, 0].mean(axis=0)
    truth_std_hw = Y_test_phys[:, 0].std(axis=0)
    safe_truth_std = np.where(truth_std_hw > 1e-8, truth_std_hw, np.nan)
    bias_maps, std_ratio_maps = {}, {}
    for label, fields_per_window in [
        ("Bilinear", [Y_base_phys] * n_win),
        ("Deterministic UNet", [w["Y_pred_det_phys"] for w in per_window]),
        ("Stochastic UNet Mean", [w["Y_pred_phys"] for w in per_window]),
    ]:
        per_win_bias = [np.where(land_hw, np.nan, f[:, 0].mean(axis=0) - truth_mean_hw) for f in fields_per_window]
        per_win_ratio = [np.where(land_hw, np.nan, f[:, 0].std(axis=0) / safe_truth_std) for f in fields_per_window]
        bias_maps[label] = np.nanmean(per_win_bias, axis=0)
        std_ratio_maps[label] = np.nanmean(per_win_ratio, axis=0)

    BIAS_LIM = 0.5
    STD_RATIO_LIM = (0, 2)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
    for col, (label, bmap) in enumerate(bias_maps.items()):
        ax = axs[0, col]
        im0 = ax.pcolormesh(tgt_lon, tgt_lat, bmap, transform=rdeb.ccrs.PlateCarree(), cmap="RdBu_r", vmin=-BIAS_LIM, vmax=BIAS_LIM, shading="auto")
        _style(ax, tgt_lon, tgt_lat)
        ax.set_title(f"{label}: time-mean bias", fontsize=12)
    for col, (label, rmap) in enumerate(std_ratio_maps.items()):
        ax = axs[1, col]
        im1 = ax.pcolormesh(tgt_lon, tgt_lat, rmap, transform=rdeb.ccrs.PlateCarree(), cmap="RdBu_r",
                             norm=TwoSlopeNorm(vcenter=1, vmin=STD_RATIO_LIM[0], vmax=STD_RATIO_LIM[1]), shading="auto")
        _style(ax, tgt_lon, tgt_lat)
        ax.set_title(f"{label}: std ratio (pred/truth)", fontsize=12)
    fig.colorbar(im0, ax=axs[0, :], shrink=0.8, pad=0.02, label="Time-mean bias (m), pooled across 4 windows")
    fig.colorbar(im1, ax=axs[1, :], shrink=0.8, pad=0.02, label="Temporal std ratio, pooled across 4 windows")
    save_fig(fig, "06_bias_std_ratio_maps_4window")

    ocean_w = np.where(land_hw, 0.0, np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon))))
    ocean_w = ocean_w / ocean_w.sum()
    save_table(pd.DataFrame([{
        "Method": label,
        "Domain mean bias (cos-lat weighted, m)": round(float(np.nansum(ocean_w * bias_maps[label])), 4),
        "Domain mean std ratio (cos-lat weighted)": round(float(np.nansum(ocean_w * std_ratio_maps[label])), 4),
    } for label in bias_maps]), "06_bias_std_ratio_summary_4window")

    # ============ Section 8: domain-mean bias time series ============
    print("  section 8: domain-mean bias time series...", flush=True)
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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    for label, col, color in [
        ("Stochastic UNet Mean", "bias_stochastic_unet_mean", "tab:blue"),
        ("Deterministic UNet", "bias_deterministic_unet", "tab:orange"),
        ("Bilinear", "bias_bilinear", "tab:green"),
        ("PIOMAS (obs)", "bias_piomas", "tab:red"),
    ]:
        if bias_plot_df[col].notna().sum() == 0:
            continue
        ax.plot(bias_plot_df["time"], bias_plot_df[col], label=label, color=color, linewidth=1.5)
    ax.set_ylabel("Domain-mean bias (pred - truth, m)"); ax.set_xlabel("Time"); ax.legend(fontsize=9)
    fig.autofmt_xdate(); plt.tight_layout()
    save_fig(fig, "08_domain_bias_timeseries_4window")
    save_table(bias_df[bias_cols].agg(["mean", "std"]).round(4), "08_domain_bias_stats_4window")

    # ============ Section 9: seasonal climatology ============
    print("  section 9: seasonal climatology...", flush=True)
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
            continue
        ax.plot(monthly_clim.index, monthly_clim[col], label=label, color=color, linestyle=ls, linewidth=1.5, marker="o", markersize=3)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylabel("Domain-mean SIT climatology (m), pooled across 4 windows")
    ax.legend(fontsize=9); plt.tight_layout()
    save_fig(fig, "09_seasonal_climatology_4window")
    save_table((monthly_clim.max() - monthly_clim.min()).rename("Seasonal amplitude (m)").round(4).to_frame(),
               "09_seasonal_amplitude_4window")

    # ============ Section 10: anomaly correlation maps ============
    print("  section 10: anomaly correlation maps...", flush=True)
    # Same reasoning as section 5's month_idx fix above: must be raw-array-aligned
    # (sample_times_df0's original order/length), not ts_df's grouped/averaged one.
    months = sample_times_df0["time"].dt.month.values
    all_idx = np.arange(len(months))
    truth_anom = rdeb.remove_monthly_climatology(Y_test_phys, months)
    raw_maps, acc_maps = {}, {}
    for label, fields_per_window in [
        ("Bilinear", [Y_base_phys] * n_win),
        ("Deterministic UNet", [w["Y_pred_det_phys"] for w in per_window]),
        ("Stochastic UNet Mean", [w["Y_pred_phys"] for w in per_window]),
    ]:
        raw_per_win = [np.where(land_hw, np.nan, rdeb.temporal_pixel_corr(f, Y_test_phys, all_idx)) for f in fields_per_window]
        acc_per_win = [np.where(land_hw, np.nan, rdeb.temporal_pixel_corr(rdeb.remove_monthly_climatology(f, months), truth_anom, all_idx))
                       for f in fields_per_window]
        raw_maps[label] = np.nanmean(raw_per_win, axis=0)
        acc_maps[label] = np.nanmean(acc_per_win, axis=0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, dpi=150, subplot_kw={"projection": proj})
    for ax, (label, amap) in zip(axs, acc_maps.items()):
        im = ax.pcolormesh(tgt_lon, tgt_lat, amap, transform=rdeb.ccrs.PlateCarree(), cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        _style(ax, tgt_lon, tgt_lat)
        ax.set_title(label, fontsize=13)
    cbar = fig.colorbar(im, ax=axs, aspect=30, shrink=0.8, pad=0.02)
    cbar.set_label("Anomaly correlation (deseasonalized), all months, pooled across 4 windows", fontsize=12)
    save_fig(fig, "10_anomaly_corr_maps_4window")
    ocean_w3 = np.where(land_hw, 0.0, np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones((1, len(tgt_lon))))
    ocean_w3 = ocean_w3 / ocean_w3.sum()
    save_table(pd.DataFrame([{
        "Method": label,
        "Raw temporal corr., all months (cos-lat weighted)": round(float(np.nansum(ocean_w3 * raw_maps[label])), 4),
        "Anomaly corr., all months (cos-lat weighted)": round(float(np.nansum(ocean_w3 * acc_maps[label])), 4),
    } for label in raw_maps]), "10_anomaly_corr_summary_4window")

    # ============ Section 11: Taylor diagram ============
    print("  section 11: Taylor diagram...", flush=True)
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

    taylor_stats = {}
    for label, fields_per_window in [
        ("Bilinear", [Y_base_phys] * n_win),
        ("Deterministic UNet", [w["Y_pred_det_phys"] for w in per_window]),
        ("Stochastic UNet Mean", [w["Y_pred_phys"] for w in per_window]),
    ]:
        stats_per_win = [weighted_stats(f, Y_test_phys, ocean_w_hw) for f in fields_per_window]
        taylor_stats[label] = tuple(np.mean(stats_per_win, axis=0))

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
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_ylim(0, r_lim)
    corr_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_xticks(np.arccos(corr_ticks)); ax.set_xticklabels([str(c) for c in corr_ticks])
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
    save_fig(fig, "11_taylor_diagram_4window")
    save_table(pd.DataFrame([
        {"Method": label, "Correlation (R)": round(R, 4), "Std Ratio": round(std_ratio, 4), "Centered RMSE (m)": round(crmse, 4)}
        for label, (R, std_ratio, crmse) in taylor_stats.items()
    ]), "11_taylor_stats_4window")

    # ============ Section 12 (numeric only): PIOMAS metrics ============
    print("  section 12: PIOMAS metrics...", flush=True)
    piomas_valid = ~np.isnan(piomas_regridded_phys[:, 0]).all(axis=(1, 2))
    if piomas_valid.sum() == 0:
        print("  No test samples overlap PIOMAS's record -- skipping PIOMAS metrics.")
        return

    def weighted_mae_rmse_bias(pred_phys, ref_phys, weight_hw, valid):
        if valid.sum() == 0:
            return np.nan, np.nan, np.nan
        diff = pred_phys[valid, 0] - ref_phys[valid, 0]
        w = weight_hw[None, :, :]
        mae = float(np.nansum(w * np.abs(diff)) / valid.sum())
        rmse = float(np.sqrt(np.nansum(w * diff ** 2) / valid.sum()))
        bias = float(np.nansum(w * diff) / valid.sum())
        return mae, rmse, bias

    piomas_metrics = []
    for label, fields_per_window in [
        ("Truth", [Y_test_phys] * n_win), ("Bilinear", [Y_base_phys] * n_win),
        ("Deterministic UNet", [w["Y_pred_det_phys"] for w in per_window]),
        ("Stochastic UNet Mean", [w["Y_pred_phys"] for w in per_window]),
    ]:
        stats_per_win = [weighted_mae_rmse_bias(f, piomas_regridded_phys, ocean_w_hw, piomas_valid) for f in fields_per_window]
        mae, rmse, bias = np.mean(stats_per_win, axis=0)
        piomas_metrics.append({"Method": label, "MAE vs. PIOMAS (m)": round(mae, 4),
                                "RMSE vs. PIOMAS (m)": round(rmse, 4), "Bias vs. PIOMAS (m)": round(bias, 4)})
    save_table(pd.DataFrame(piomas_metrics), "12_piomas_metrics_4window")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def save_fig(fig, name):
        fig.savefig(os.path.join(args.save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_table(df, name):
        df.to_csv(os.path.join(args.save_dir, f"{name}.csv"))

    # Only directories with real eval_data/fields.npz -- a batch dir can accumulate crashed/
    # incomplete run attempts (e.g. from earlier bug-hunting) that only got as far as writing
    # run_config.json before failing. Same defensive filter run_daily_eval_batch.py's own
    # main() already uses, for the same reason.
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(args.batch_dir, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "eval_data", "fields.npz"))
    )
    print(f"Pooling {len(run_dirs)} windows from {args.batch_dir}:", flush=True)
    acc = {}
    for i, d in enumerate(run_dirs):
        print(f"  {os.path.basename(d)}", flush=True)
        accumulate_window(d, acc, seed=i)
        print(f"  done with {os.path.basename(d)}", flush=True)
    K = acc["K"]

    # ---- Rank histogram figure/table ----
    rank_counts = acc["rank_counts"]
    rank_freq = rank_counts / rank_counts.sum()
    expected_freq = 1.0 / (K + 1)
    rank_hist_df = pd.DataFrame({
        "rank": np.arange(K + 1), "count": rank_counts, "frequency": rank_freq,
        "expected_frequency": expected_freq,
    })
    save_table(rank_hist_df, "14_rank_histogram_data_4window")

    rank_by_regime = acc["rank_counts_by_regime"]
    regime_rows = []
    for b, label in enumerate(SIT_BIN_LABELS):
        freq_b = rank_by_regime[b] / max(rank_by_regime[b].sum(), 1)
        for r in range(K + 1):
            regime_rows.append({"SIT regime": label, "rank": r, "count": int(rank_by_regime[b, r]),
                                "frequency": freq_b[r], "relative_frequency": freq_b[r] / expected_freq})
    rank_regime_df = pd.DataFrame(regime_rows)
    save_table(rank_regime_df, "14b_rank_histogram_by_regime_data_4window")

    regime_colors = ["tab:cyan", "tab:green", "tab:orange", "tab:red"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ax1.bar(rank_hist_df["rank"], rank_hist_df["frequency"], width=0.9, color="tab:blue", alpha=0.85, label="Observed")
    ax1.axhline(expected_freq, color="black", linestyle="--", linewidth=1.2, label="Perfect calibration (flat)")
    ax1.set_xlabel(f"Rank of truth among {K} sorted ensemble members")
    ax1.set_ylabel("Frequency")
    ax1.set_title("(A) Rank histogram, all ocean cells")
    ax1.legend(fontsize=9, frameon=False)
    for (label, grp), color in zip(rank_regime_df.groupby("SIT regime", sort=False), regime_colors):
        share = grp["count"].sum() / rank_counts.sum()
        ax2.plot(grp["rank"], grp["relative_frequency"], marker="o", markersize=3, color=color,
                 label=f"{label} ({share:.0%} of cells)")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Perfect calibration (flat)")
    ax2.set_xlabel(f"Rank of truth among {K} sorted ensemble members")
    ax2.set_ylabel("Frequency / expected frequency")
    ax2.set_title("(B) Rank histogram by ensemble-mean SIT regime\n(U-shape = under-dispersed, hump = over-dispersed)")
    ax2.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    save_fig(fig, "14_rank_histogram_4window")

    # ---- Reliability diagram figure/table ----
    bin_edges = np.linspace(0, 1, N_PROB_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    threshold_colors = ["tab:blue", "tab:orange", "tab:red"]
    reliability_rows, curves = [], {}
    for thr in RELIABILITY_THRESHOLDS:
        d = acc[f"reliab_{thr}"]
        counts = d["count"]
        mean_fc = np.where(counts > 0, d["sum_fc"] / np.maximum(counts, 1), np.nan)
        obs_freq = np.where(counts > 0, d["sum_obs"] / np.maximum(counts, 1), np.nan)
        brier = d["sq_err_sum"] / d["n_total"]
        climo = d["outcome_sum"] / d["n_total"]
        curves[thr] = {"mean_forecast": mean_fc, "obs_freq": obs_freq, "count": counts}
        reliability_rows.append({"Threshold (m)": thr, "Brier Score": round(brier, 4), "Climatological frequency": round(climo, 4)})
    save_table(pd.DataFrame(reliability_rows), "14_reliability_diagram_data_4window")
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
    ax_main.set_title("Reliability diagram, pooled across 4 train windows (test=2021)")
    ax_hist.set_xlabel("Forecast probability")
    ax_hist.set_ylabel("Fraction of\nsamples")
    ax_hist.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    save_fig(fig, "14_reliability_diagram_4window")

    # ---- Spread-skill by regime figure/table ----
    # Spread-skill ratio = sqrt((K+1)/K * mean ens variance) / RMSE(ens mean), which is 1.0 for
    # a calibrated K-member ensemble (Fortin et al. 2014). mean(std)/mean|err| is kept for
    # reference only: its calibrated value is ~1.21 for K=20 (Gaussian), not 1.0.
    ss = acc["spread_skill"]
    c4 = np.sqrt(2.0 / (K - 1)) * np.exp(math.lgamma(K / 2) - math.lgamma((K - 1) / 2))
    mae_ratio_calibrated = c4 * np.sqrt(np.pi / 2) / np.sqrt(1 + 1.0 / K)
    rows = []
    for b, label in enumerate(SIT_BIN_LABELS + ["All ocean cells"]):
        n = ss["n"][b]
        rms_spread = np.sqrt((K + 1) / K * ss["sum_var"][b] / n)
        rmse_b = np.sqrt(ss["sum_sq_error"][b] / n)
        rows.append({
            "SIT regime": label, "N": int(n), "Mean true SIT (m)": round(ss["sum_truth"][b] / n, 4),
            "Ensemble-mean bias (m)": round(ss["sum_signed_error"][b] / n, 4),
            "RMS spread, (K+1)/K-corrected (m)": round(rms_spread, 4), "RMSE (m)": round(rmse_b, 4),
            "Spread-skill ratio (1.0 = calibrated)": round(rms_spread / rmse_b, 4),
            "Mean ensemble spread (m)": round(ss["sum_spread"][b] / n, 4), "Mean |error| (m)": round(ss["sum_error"][b] / n, 4),
            "Mean spread / mean |error|": round(ss["sum_spread"][b] / ss["sum_error"][b], 4),
            "Mean spread / mean |error|, calibrated Gaussian value": round(mae_ratio_calibrated, 4),
        })
    spread_skill_df = pd.DataFrame(rows)
    save_table(spread_skill_df, "15_spread_skill_by_regime_data_4window")
    regime_df = spread_skill_df.iloc[:len(SIT_BIN_LABELS)]
    x = np.arange(len(SIT_BIN_LABELS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.35
    ax1.bar(x - width / 2, regime_df["RMS spread, (K+1)/K-corrected (m)"], width, color="tab:blue", label="Ensemble spread (RMS, $(K+1)/K$-corrected)")
    ax1.bar(x + width / 2, regime_df["RMSE (m)"], width, color="tab:orange", label="RMSE of ensemble mean")
    ax1.set_xticks(x); ax1.set_xticklabels(SIT_BIN_LABELS, fontsize=8, rotation=20, ha="right")
    ax1.set_ylabel("m"); ax1.set_title("(A) Spread vs. error by ensemble-mean SIT regime")
    ax1.legend(fontsize=9, frameon=False)
    ax2.bar(x, regime_df["Spread-skill ratio (1.0 = calibrated)"], color="tab:green", alpha=0.85)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Calibrated")
    ax2.set_xticks(x); ax2.set_xticklabels(SIT_BIN_LABELS, fontsize=8, rotation=20, ha="right")
    ax2.set_ylabel("Spread-skill ratio\n(<1 under-dispersed, >1 over-dispersed)")
    ax2.set_title("(B) Spread-skill ratio by regime"); ax2.legend(fontsize=9, frameon=False, loc="lower right")
    plt.tight_layout()
    save_fig(fig, "15_spread_skill_by_regime_4window")

    # ---- CRPS by regime figure/table ----
    crps_acc = acc["crps"]
    crps_rows = [{"SIT regime": "All (domain-wide)", "N": crps_acc["n_all"],
                  "CRPS (m)": round(crps_acc["sum_all"] / crps_acc["n_all"], 4)}]
    for b, label in enumerate(SIT_BIN_LABELS):
        n = crps_acc["n"][b]
        crps_rows.append({"SIT regime": label, "N": int(n),
                           "CRPS (m)": round(crps_acc["sum"][b] / n, 4) if n > 0 else np.nan})
    crps_df = pd.DataFrame(crps_rows)
    save_table(crps_df, "16_crps_by_regime_data_4window")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_rows = crps_df.iloc[1:]
    ax.bar(np.arange(len(plot_rows)), plot_rows["CRPS (m)"], color="tab:purple", alpha=0.85)
    ax.axhline(crps_df.iloc[0]["CRPS (m)"], color="black", linestyle="--", linewidth=1.2,
               label=f"Domain-wide CRPS ({crps_df.iloc[0]['CRPS (m)']:.4f} m)")
    ax.set_xticks(np.arange(len(plot_rows))); ax.set_xticklabels(plot_rows["SIT regime"], fontsize=8)
    ax.set_ylabel("CRPS (m, lower = better)")
    ax.set_title("CRPS by true-SIT regime, pooled across 4 train windows (test=2021)")
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    save_fig(fig, "16_crps_by_regime_4window")

    # ---- PSD: average the 4 windows' already-computed per-window curves ----
    psd_frames = [pd.read_csv(f, index_col=0) for f in
                  glob.glob(os.path.join("evaluation", "saved_figs", os.path.basename(args.batch_dir), "*", "13_psd_data.csv"))]
    if psd_frames:
        method_cols = [c for c in psd_frames[0].columns if c not in ("wavenumber_cycles_per_km", "wavelength_km")]
        psd_avg = psd_frames[0][["wavenumber_cycles_per_km", "wavelength_km"]].copy()
        for c in method_cols:
            psd_avg[c] = np.mean([f[c].values for f in psd_frames], axis=0)
        save_table(psd_avg, "13_psd_data_4window")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        wavenumber = psd_avg["wavenumber_cycles_per_km"].values
        for c in method_cols:
            valid = (wavenumber > 0) & (psd_avg[c] > 0)
            ax.plot(wavenumber[valid], psd_avg[c][valid], label=c, linewidth=1.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Wavenumber (cycles/km)"); ax.set_ylabel("Isotropic PSD (m$^2$, arb. spectral units)")
        ax.set_title("Domain-wide PSD, averaged across 4 train windows (test=2021)")
        ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        save_fig(fig, "13_psd_comparison_4window")

    psdb_frames = [pd.read_csv(f, index_col=0) for f in
                   glob.glob(os.path.join("evaluation", "saved_figs", os.path.basename(args.batch_dir), "*", "13b_psd_per_member_spread_data.csv"))]
    if psdb_frames:
        cols = [c for c in psdb_frames[0].columns if c not in ("wavenumber_cycles_per_km", "wavelength_km")]
        psdb_avg = psdb_frames[0][["wavenumber_cycles_per_km", "wavelength_km"]].copy()
        for c in cols:
            psdb_avg[c] = np.mean([f[c].values for f in psdb_frames], axis=0)
        # Recomputed from the pooled p10/p90/mean rather than averaged directly from each window's
        # own already-saved "Relative member spread pct" column (older saved_figs from before this
        # column existed won't have it) -- cheap and exactly equivalent either way since it's a
        # simple ratio of already-pooled quantities.
        psdb_avg["Relative member spread pct"] = (psdb_avg["Member p90"] - psdb_avg["Member p10"]) / psdb_avg["Member mean"] * 100
        save_table(psdb_avg, "13b_psd_per_member_spread_data_4window")
        # See run_daily_eval_batch.py's matching comment: the left panel's p10-p90 band is real,
        # just too thin (relative to the many-orders-of-magnitude log-log PSD axis) to see -- the
        # right panel makes that same small relative spread legible on its own linear axis.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        wavenumber = psdb_avg["wavenumber_cycles_per_km"].values
        valid = (wavenumber > 0) & (psdb_avg["Truth"] > 0)
        ax1.plot(wavenumber[valid], psdb_avg["Truth"][valid], label="Truth", color="black", linewidth=1.8)
        ax1.fill_between(wavenumber[valid], psdb_avg["Member p10"][valid], psdb_avg["Member p90"][valid],
                          color="tab:blue", alpha=0.3, label="Stochastic UNet members (10th-90th pct)")
        ax1.plot(wavenumber[valid], psdb_avg["Member mean"][valid], color="tab:blue", linewidth=1.2, linestyle="--",
                 label="Stochastic UNet member mean")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("Wavenumber (cycles/km)"); ax1.set_ylabel("Isotropic PSD (m$^2$, arb. spectral units)")
        ax1.set_title("Per-member PSD spread vs. truth")
        ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)

        ax2.plot(wavenumber[valid], psdb_avg["Relative member spread pct"][valid], color="tab:blue", linewidth=1.8)
        ax2.set_xscale("log")
        ax2.set_xlabel("Wavenumber (cycles/km)")
        ax2.set_ylabel("Relative member spread,\n(p90 - p10) / mean (%)")
        ax2.set_title("Inter-member spectral-texture spread\n(near-zero = members look nearly identical)")
        ax2.grid(True, which="both", alpha=0.3)
        fig.suptitle("Per-member PSD spread, averaged across 4 train windows", y=1.02)
        plt.tight_layout()
        save_fig(fig, "13b_psd_per_member_spread_4window")

    # ---- Sections 1, 2, 5, 6, 8, 9, 10, 11, 12 (numeric): full-period statistics, pooled
    # by recomputing each window's own statistic and averaging (see pool_full_period_sections'
    # docstring). Sections 3, 4, and 12's map panel are spatial snapshots, deliberately left
    # to a single representative window; section 7 (trend) needs >= 5 test years, which no
    # window here has, so it produces nothing to pool. ----
    print("\nPooling full-period sections (1, 2, 5, 6, 8, 9, 10, 11, 12)...", flush=True)
    pool_full_period_sections(run_dirs, args.save_dir)

    print(f"\nDone. 4-window-aggregate figures/tables written to: {args.save_dir}")


if __name__ == "__main__":
    main()
