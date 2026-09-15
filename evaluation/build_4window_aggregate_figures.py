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
        --save-dir saved_figs/MESA_stochastic_refine_sweep_avg_sharedbias/_4window_aggregate
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SIT_BIN_EDGES = [0.0, 0.15, 0.5, 1.0, np.inf]
SIT_BIN_LABELS = ["Open water/frazil (0-0.15 m)", "Thin ice (0.15-0.5 m)", "Moderate ice (0.5-1.0 m)", "Thick ice (>1.0 m)"]
RELIABILITY_THRESHOLDS = [0.15, 0.5, 1.0]
N_PROB_BINS = 10


def calibration_score(ratio):
    if not np.isfinite(ratio) or ratio <= 0:
        return np.nan
    return min(ratio, 1.0 / ratio)


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
    bin_idx_sit = np.digitize(truth_ocean, SIT_BIN_EDGES[1:-1])
    ss = acc.setdefault("spread_skill", {
        "n": np.zeros(len(SIT_BIN_LABELS), dtype=np.int64),
        "sum_truth": np.zeros(len(SIT_BIN_LABELS)),
        "sum_spread": np.zeros(len(SIT_BIN_LABELS)), "sum_error": np.zeros(len(SIT_BIN_LABELS)),
    })
    for b in range(len(SIT_BIN_LABELS)):
        sel = bin_idx_sit == b
        ss["n"][b] += int(sel.sum())
        ss["sum_truth"][b] += float(truth_ocean[sel].sum())
        ss["sum_spread"][b] += float(ens_spread_ocean[sel].sum())
        ss["sum_error"][b] += float(abs_error_ocean[sel].sum())

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
    rank_calibration = np.array([calibration_score(f / expected_freq) for f in rank_freq])
    rank_hist_df = pd.DataFrame({
        "rank": np.arange(K + 1), "count": rank_counts, "frequency": rank_freq,
        "expected_frequency": expected_freq, "Calibration Score (1.0 = ideal)": rank_calibration,
    })
    save_table(rank_hist_df, "14_rank_histogram_data_4window")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ax1.bar(rank_hist_df["rank"], rank_hist_df["frequency"], width=0.9, color="tab:blue", alpha=0.85, label="Observed")
    ax1.axhline(expected_freq, color="black", linestyle="--", linewidth=1.2, label="Perfect calibration (flat)")
    ax1.set_xlabel(f"Rank of truth among {K} sorted ensemble members")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Rank histogram, pooled across 4 train windows (test=2021)")
    ax1.legend(fontsize=9, frameon=False)
    ax2.bar(rank_hist_df["rank"], rank_hist_df["Calibration Score (1.0 = ideal)"], width=0.9, color="tab:green", alpha=0.85)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax2.set_xlabel(f"Rank of truth among {K} sorted ensemble members")
    ax2.set_ylabel("Calibration score (min(ratio, 1/ratio))")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Per-rank calibration score, pooled")
    ax2.legend(fontsize=9, frameon=False, loc="lower right")
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
    ss = acc["spread_skill"]
    rows = []
    for b, label in enumerate(SIT_BIN_LABELS):
        n = ss["n"][b]
        mean_truth = ss["sum_truth"][b] / n if n > 0 else np.nan
        mean_spread = ss["sum_spread"][b] / n if n > 0 else np.nan
        mean_error = ss["sum_error"][b] / n if n > 0 else np.nan
        ratio = mean_spread / mean_error if mean_error > 0 else np.nan
        rows.append({
            "SIT regime": label, "N": int(n), "Mean true SIT (m)": round(mean_truth, 4),
            "Mean ensemble spread (m)": round(mean_spread, 4), "Mean |error| (m)": round(mean_error, 4),
            "Spread/Error": round(ratio, 4) if np.isfinite(ratio) else np.nan,
            "Calibration Score (1.0 = ideal)": round(calibration_score(ratio), 4),
        })
    spread_skill_df = pd.DataFrame(rows)
    save_table(spread_skill_df, "15_spread_skill_by_regime_data_4window")
    x = np.arange(len(SIT_BIN_LABELS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.35
    ax1.bar(x - width / 2, spread_skill_df["Mean ensemble spread (m)"], width, color="tab:blue", label="Ensemble spread (std)")
    ax1.bar(x + width / 2, spread_skill_df["Mean |error| (m)"], width, color="tab:orange", label="|Ensemble mean - truth|")
    ax1.set_xticks(x); ax1.set_xticklabels(SIT_BIN_LABELS, fontsize=8)
    ax1.set_ylabel("m"); ax1.set_title("Spread vs. error by true-SIT regime, pooled across 4 windows")
    ax1.legend(fontsize=9, frameon=False)
    ax2.bar(x, spread_skill_df["Calibration Score (1.0 = ideal)"], color="tab:green", alpha=0.85)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax2.set_xticks(x); ax2.set_xticklabels(SIT_BIN_LABELS, fontsize=8)
    ax2.set_ylabel("Calibration score (min(ratio, 1/ratio))"); ax2.set_ylim(0, 1.05)
    ax2.set_title("Normalized calibration score by regime, pooled"); ax2.legend(fontsize=9, frameon=False)
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
                  glob.glob(os.path.join("saved_figs", os.path.basename(args.batch_dir), "*", "13_psd_data.csv"))]
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
                   glob.glob(os.path.join("saved_figs", os.path.basename(args.batch_dir), "*", "13b_psd_per_member_spread_data.csv"))]
    if psdb_frames:
        cols = [c for c in psdb_frames[0].columns if c not in ("wavenumber_cycles_per_km", "wavelength_km")]
        psdb_avg = psdb_frames[0][["wavenumber_cycles_per_km", "wavelength_km"]].copy()
        for c in cols:
            psdb_avg[c] = np.mean([f[c].values for f in psdb_frames], axis=0)
        save_table(psdb_avg, "13b_psd_per_member_spread_data_4window")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        wavenumber = psdb_avg["wavenumber_cycles_per_km"].values
        valid = (wavenumber > 0) & (psdb_avg["Truth"] > 0)
        ax.plot(wavenumber[valid], psdb_avg["Truth"][valid], label="Truth", color="black", linewidth=1.8)
        ax.fill_between(wavenumber[valid], psdb_avg["Member p10"][valid], psdb_avg["Member p90"][valid],
                         color="tab:blue", alpha=0.3, label="Stochastic UNet members (10th-90th pct)")
        ax.plot(wavenumber[valid], psdb_avg["Member mean"][valid], color="tab:blue", linewidth=1.2, linestyle="--",
                label="Stochastic UNet member mean")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Wavenumber (cycles/km)"); ax.set_ylabel("Isotropic PSD (m$^2$, arb. spectral units)")
        ax.set_title("Per-member PSD spread vs. truth, averaged across 4 train windows")
        ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        save_fig(fig, "13b_psd_per_member_spread_4window")

    print(f"\nDone. 4-window-aggregate figures/tables written to: {args.save_dir}")


if __name__ == "__main__":
    main()
