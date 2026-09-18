#!/usr/bin/env python3
"""
Sensitivity-test heatmaps for the paper appendix, in the same style as
Version5/evaluation/compare_all_batches.ipynb's plot_badness_heatmap: rows = tested
configurations, columns = metrics, color = per-column "badness" (0 = best row on that
metric among the rows shown here, 1 = worst; direction-aware), raw values annotated.

Two SEPARATE figures (A1 = MESA, A2 = FOSI), not one combined table: FOSI's absolute
error scale differs enough from MESA's (independent low-res source) that mixing them
into one column-normalized "badness" scale would misleadingly paint every FOSI row red
regardless of how FOSI actually performs relative to its own alternatives. A2 has far
fewer rows than A1 because most cascade-specific ablations (seed variance, mask-fusion
timing, noise channels, kernel size, the isolated no-correlated-noise/global-z
comparisons) were only ever run on MESA -- this is a real, honestly-reported gap in
sweep coverage, not an omission from this figure.

All ten metrics from functions_engressnet.compute_metrics_table are included. Historical
old-refiner-era rows (both A1 and A2) only have the columns that were actually recorded
in recommended_config.md at the time -- MAE/Bias/Grad MAE/Coastal MAE are blank for those
rows since the underlying eval_data no longer exists to recover them, and IIEE is blank
for all but the two isolated-mechanism rows (no correlated noise, global z) since those
predate this session's IIEE bugfix and area-weighting fix and can't be made consistent
with the rest of the table.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_DIRECTION = {
    "MAE": "lower", "RMSE": "lower", "Bias": "zero", "Grad MAE": "lower",
    "Pattern Corr": "higher", "SSIM": "higher", "IIEE": "lower",
    "Coastal MAE": "lower", "Coastal RMSE": "lower", "Spread/Error": "one",
}
METRIC_COLS = list(METRIC_DIRECTION.keys())
NAN = np.nan

# ============================================================
# A1: MESA
# ============================================================
mesa_rows = {
    # -- old refiner-era historical (recommended_config.md; MAE/Bias/GradMAE/CoastalMAE
    #    were never recorded for these rows; IIEE unrecoverable except where noted) --
    "avg, pre-fix (old refiner)":                  [NAN, 0.1111, NAN, NAN, 0.8783, 0.9344, 0.3312, NAN, 0.1938, 1.034],
    "interp, pre-fix (old refiner)":                [NAN, 0.1332, NAN, NAN, 0.8468, 0.9201, NAN,    NAN, 0.2516, 0.916],
    "avg, shared-bias fix (old refiner)":           [NAN, 0.1080, NAN, NAN, NAN,    NAN,    0.2936, NAN, 0.1888, 0.972],
    "avg, deep mask head (old refiner)":            [NAN, 0.1091, NAN, NAN, NAN,    NAN,    0.2431, NAN, 0.1884, 1.129],
    "avg, noise-mix none (old refiner)":            [NAN, 0.1109, NAN, NAN, NAN,    NAN,    0.3944, NAN, 0.1885, 1.370],
    "avg, noise-mix gaussian_fixed (old refiner)":  [NAN, 0.1112, NAN, NAN, NAN,    NAN,    0.2367, NAN, 0.1868, 1.369],
    # -- isolated noise-mechanism comparisons (current architecture generation, IIEE
    #    recomputed 2026-09-18 with the current, corrected methodology) --
    "no correlated noise":            [0.0418, 0.1196, 0.0010, 0.0454, 0.8662, 0.9117, 0.2132, 0.0924, 0.1970, 1.8720],
    "single global noise field (\"global z\")": [0.0414, 0.1184, 0.0014, 0.0454, 0.8612, 0.9097, 0.2842, 0.0936, 0.1986, 1.8787],
    # -- finalized per-stage cascade --
    "cascade (used in paper), 4-window avg": [0.0369, 0.1126, -0.0008, 0.0407, 0.8786, 0.9269, 0.2392, 0.0816, 0.1872, 1.3056],
    "cascade, 2000-2005 window, seed 0":     [0.0367, 0.1115, -0.0009, 0.0394, 0.8748, 0.9273, 0.2566, 0.0808, 0.1846, 1.3492],
    "cascade, 2000-2005 window, seed 1":     [0.0369, 0.1114, -0.0005, 0.0405, 0.8736, 0.9260, 0.1500, 0.0801, 0.1819, 1.4596],
    "cascade, 2000-2005 window, seed 2":     [0.0377, 0.1141, -0.0005, 0.0433, 0.8744, 0.9240, 0.3136, 0.0842, 0.1908, 1.3375],
    "cascade, 2000-2005 window, seed 3":     [0.0371, 0.1114,  0.0002, 0.0415, 0.8687, 0.9237, 0.3386, 0.0829, 0.1869, 1.3384],
    "cascade, 2000-2005 window, seed 4":     [0.0368, 0.1115, -0.0003, 0.0400, 0.8694, 0.9260, 0.3205, 0.0815, 0.1855, 1.2921],
    "cascade, late mask fusion":             [0.0370, 0.1128, -0.0001, 0.0426, 0.8755, 0.9247, 0.2087, 0.0819, 0.1884, 1.3413],
    "cascade, noise channels 1→2":       [0.0373, 0.1134, -0.0006, 0.0413, 0.8757, 0.9249, 0.2729, 0.0828, 0.1889, 1.3304],
    "cascade, kernel size 5→3":          [0.0362, 0.1110, -0.0014, 0.0411, 0.8798, 0.9271, 0.1910, 0.0803, 0.1842, 1.2835],
    "cascade, kernel size 5→9":          [0.0364, 0.1118, -0.0003, 0.0393, 0.8763, 0.9276, 0.2621, 0.0807, 0.1851, 1.3085],
}

# ============================================================
# A2: FOSI
# ============================================================
fosi_rows = {
    # -- old refiner-era historical, "smooth-noise-fixed EnScale-lite" checkpoint
    #    generation (recommended_config.md) -- a different, earlier historical stage
    #    than MESA's shared-bias/deep-mask-head ablations, which were MESA-only --
    "avg, smooth-noise-fixed refiner (historical)":    [NAN, 0.1804, NAN, NAN, 0.9458, 0.8968, NAN, NAN, 0.2960, 0.833],
    "interp, smooth-noise-fixed refiner (historical)": [NAN, 0.2160, NAN, NAN, 0.9208, 0.8826, NAN, NAN, 0.3916, 0.750],
    "trained full-year (historical refiner)":          [NAN, 0.2155, NAN, NAN, 0.9722, 0.8489, NAN, NAN, 0.3758, 1.3563],
    "trained season-only (historical refiner)":        [NAN, 0.2244, NAN, NAN, 0.9697, 0.8319, NAN, NAN, 0.3917, 1.2713],
    # -- finalized per-stage cascade --
    "cascade (used in paper), 4-window avg": [0.0707, 0.1811, -0.0045, 0.0707, 0.9453, 0.8980, 0.1739, 0.1527, 0.2956, 1.2045],
    "cascade, MESA-warm-started":            [0.0686, 0.1784, -0.0031, 0.0687, 0.9499, 0.9037, 0.1638, 0.1470, 0.2895, 0.8903],
}

mesa_df = pd.DataFrame.from_dict(mesa_rows, orient="index", columns=METRIC_COLS)
fosi_df = pd.DataFrame.from_dict(fosi_rows, orient="index", columns=METRIC_COLS)


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
    """Ported from Version5/evaluation/compare_all_batches.ipynb."""
    badness_df = pd.DataFrame(index=value_df.index, columns=value_df.columns, dtype=float)
    for col in value_df.columns:
        direction = METRIC_DIRECTION.get(col, "lower")
        badness_df[col] = normalize_0_1(badness(value_df[col], direction))

    if figsize is None:
        figsize = (1.15 * len(value_df.columns) + 2, 0.5 * len(value_df.index) + 2)
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


OUT_DIR = "/glade/work/skygale/projects/SeaIceDownscaling/Version6/sensitivity/analysis"

fig1, ax1 = plot_badness_heatmap(
    mesa_df,
    "A1: Sensitivity tests, MESA -- configuration × metric\n"
    "color = relative badness within this table (lightest = best, darkest = worst per column); blank = not measured",
)
fig1.savefig(f"{OUT_DIR}/sensitivity_heatmap_A1_mesa.png", dpi=220, bbox_inches="tight")
fig1.savefig(f"{OUT_DIR}/sensitivity_heatmap_A1_mesa.pdf", bbox_inches="tight")

fig2, ax2 = plot_badness_heatmap(
    fosi_df,
    "A2: Sensitivity tests, FOSI -- configuration × metric\n"
    "color = relative badness within this table (lightest = best, darkest = worst per column); blank = not measured",
)
fig2.savefig(f"{OUT_DIR}/sensitivity_heatmap_A2_fosi.png", dpi=220, bbox_inches="tight")
fig2.savefig(f"{OUT_DIR}/sensitivity_heatmap_A2_fosi.pdf", bbox_inches="tight")

print("wrote A1 (MESA) and A2 (FOSI) heatmaps")
