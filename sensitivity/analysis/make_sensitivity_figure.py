#!/usr/bin/env python3
"""
Sensitivity-test summary figure for the paper appendix.

Two groups of rows:
  1. "Version5 carry-over" -- toggle/data sensitivity tests run on the earlier
     (global-z / LocallyConnected2d-refiner) architecture, all still relevant since
     they test data-side or loss-side choices independent of the decoder's
     noise-injection mechanism (see recommended_config.md). Numbers verified
     directly against that file, not from memory.
  2. "Architecture progression" -- the three decoder noise-injection designs tried
     in this paper, all vs. the same reference (the original, now-removed
     LocallyConnected2d refiner).
  3. "Per-stage cascade sensitivity tests" -- six additional tests run against the
     finalized cascade itself (seed variance, mask-fusion timing, noise_channels,
     kernel size x2, FOSI vs. MESA), single seed per config except the seed-variance
     row. Real numbers as of 2026-09-18, all pending jobs finished.

All numbers are MESA, 4-window-averaged (train in {2000-05,05-10,10-15,15-20},
test=2021), DATA_VARIANT=avg unless noted otherwise in the row label itself.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ---- validated palette (dataviz skill reference instance) ----
BLUE = "#2a78d6"     # improvement (RMSE down / Spread-Error toward 1.0)
RED = "#e34948"      # cost (RMSE up / Spread-Error away from 1.0)
GRAY_TEXT = "#52514e"
GRAY_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": "#0b0b0b",
    "axes.edgecolor": GRID,
    "axes.labelcolor": GRAY_TEXT,
    "xtick.color": GRAY_TEXT,
    "ytick.color": "#0b0b0b",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
})

# ============================================================
# Group 1: Version5 carry-over sensitivity tests (MESA, matched baselines)
# ============================================================
g1_labels = [
    "Data variant: interp → avg",
    "Land threshold: 0.1 → 0.5",
    "Truth regrid: bilinear → conservative",
    "Training data: full-year → season-restricted",
    "Noise mixing: learned (LocallyConnected2d) → none",
    "Deep mask head: off → on",
]
g1_rmse_pct = [-16.6, 10.0, 12.0, 2.8, 2.7, 1.0]
# Spread/Error (before, after); None where no clean matched pair exists
g1_spread = [(0.916, 1.034), None, None, None, (0.972, 1.370), (0.972, 1.129)]

# ============================================================
# Group 2: architecture progression (MESA, avg, vs. the original refiner)
# ============================================================
g2_labels = [
    "Refiner (patchy, removed) → global z only",
    "Refiner (patchy, removed) → per-stage noise cascade",
]
# RMSE: 0.1080 -> 0.1184 (+9.6%), 0.1080 -> 0.1126 (+4.3%)
g2_rmse_pct = [9.6, 4.3]
g2_spread = [(0.972, 1.879), (0.972, 1.306)]

# ============================================================
# Group 3: per-stage cascade sensitivity tests (MESA unless noted; single seed
# per config except the seed-variance row itself, which reports the 5-seed range
# via its own before/after pair: (min, max) rather than (baseline, toggled))
# ============================================================
g3_labels = [
    "Per-stage cascade: seed variance (5 seeds)",
    "Per-stage cascade: mask fusion timing (early → late)",
    "Per-stage cascade: noise channels (1 → 2)",
    "Per-stage cascade: kernel size (5 → 3)",
    "Per-stage cascade: kernel size (5 → 9)",
    "Per-stage cascade: FOSI vs. MESA (Δ improvement over own bilinear baseline)",
]
# seed-variance row's "% change" is span-over-mean, not a directional before/after.
# The FOSI row is NOT a raw RMSE comparison (FOSI and MESA have very different absolute
# error scales from different low-res source data) -- it's the difference in each
# dataset's OWN %-improvement over its OWN bilinear baseline (MESA: -56.4%, FOSI:
# -59.4%), i.e. does the cascade's relative benefit transfer across datasets.
g3_rmse_pct = [(0.1141 - 0.1114) / 0.1126 * 100, 0.2, 0.7, -1.4, -0.7, -59.4 - (-56.4)]
g3_spread = [(1.284, 1.460), (1.306, 1.341), (1.306, 1.330), (1.306, 1.284), (1.306, 1.309), (1.306, 1.205)]

all_labels = g1_labels + g2_labels + g3_labels
n_g1, n_g2, n_pending = len(g1_labels), len(g2_labels), len(g3_labels)
n_total = len(all_labels)
y = np.arange(n_total)[::-1]  # top-to-bottom reading order

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11.5, 0.52 * n_total + 1.6),
    gridspec_kw={"width_ratios": [1.15, 1.0]},
)

# ---------------- Panel A: RMSE % change ----------------
rmse_vals = g1_rmse_pct + g2_rmse_pct + g3_rmse_pct
seedvar_idx = n_g1 + n_g2  # first g3 row: a dispersion measure, not a directional effect
bar_h = 0.52
for i, (yi, val) in enumerate(zip(y, rmse_vals)):
    if val is None:
        ax1.barh(yi, 1, left=-0.5, height=bar_h, facecolor="none",
                  edgecolor=GRAY_MUTED, hatch="////", linewidth=0.8)
        continue
    if i == seedvar_idx:
        # span-over-mean across 5 seeds, not a before/after toggle effect -- neutral
        # color and "range" label so it doesn't read as a worse/better comparison.
        ax1.barh(yi, val, height=bar_h, color=GRAY_MUTED, edgecolor="none")
        ax1.text(val + 0.6, yi, f"±{val:.1f}% range", va="center", ha="left",
                  fontsize=9.5, color="#0b0b0b")
        continue
    color = RED if val > 0 else BLUE
    ax1.barh(yi, val, height=bar_h, color=color, edgecolor="none")
    label_x = val + (0.6 if val > 0 else -0.6)
    ha = "left" if val > 0 else "right"
    ax1.text(label_x, yi, f"{val:+.1f}%", va="center", ha=ha,
              fontsize=9.5, color="#0b0b0b")

ax1.axvline(0, color=GRAY_TEXT, linewidth=1.1)
ax1.set_xlim(-26, 22)
ax1.set_yticks(y)
ax1.set_yticklabels(all_labels, fontsize=9.5)
ax1.set_xlabel("RMSE change vs. reference (%)\n← improves        worsens →")
ax1.set_title("Domain-mean RMSE")
ax1.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax1.spines[spine].set_visible(False)

# ---------------- Panel B: Spread/Error (calibration) ----------------
spread_vals = g1_spread + g2_spread + g3_spread
ax2.axvline(1.0, color=GRAY_TEXT, linewidth=1.1, linestyle="--", zorder=1)
for i, (yi, pair) in enumerate(zip(y, spread_vals)):
    if pair is None:
        continue  # row spacing already fixed by ax2.set_yticks(y) below
    before, after = pair
    if i == seedvar_idx:
        # (min, max) across 5 seeds, not a before/after toggle -- both points gray,
        # en-dash range label, no directional color.
        ax2.plot([before, after], [yi, yi], color=GRAY_MUTED, linewidth=1.6, zorder=2)
        ax2.scatter([before, after], [yi, yi], s=42, color=GRAY_MUTED, zorder=3, label="_nolegend_")
        ax2.text(max(before, after) + 0.05, yi, f"{before:.2f}–{after:.2f} (range)",
                  va="center", ha="left", fontsize=9, color="#0b0b0b")
        continue
    ax2.plot([before, after], [yi, yi], color=GRAY_MUTED, linewidth=1.6, zorder=2)
    ax2.scatter([before], [yi], s=42, color=GRAY_MUTED, zorder=3, label="_nolegend_")
    color_after = RED if abs(after - 1.0) > abs(before - 1.0) else BLUE
    ax2.scatter([after], [yi], s=58, color=color_after, zorder=4, label="_nolegend_")
    ax2.text(max(before, after) + 0.05, yi, f"{before:.2f}→{after:.2f}",
              va="center", ha="left", fontsize=9, color="#0b0b0b")

ax2.set_xlim(0.6, 2.15)
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.tick_params(axis="y", length=0)
ax2.set_xlabel("Spread/Error (1.0 = ideal calibration)")
ax2.set_title("Ensemble calibration")
ax2.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax2.spines[spine].set_visible(False)

for ax in (ax1, ax2):
    ax.set_ylim(-0.7, n_total - 0.3)

# Group separators + labels
def group_band(ax, y0, y1, text):
    ax.axhspan(y0, y1, color=GRAY_MUTED, alpha=0.06, zorder=0)

y_g1_top, y_g1_bot = y[0] + bar_h / 2 + 0.05, y[n_g1 - 1] - bar_h / 2 - 0.05
y_g2_top, y_g2_bot = y[n_g1] + bar_h / 2 + 0.05, y[n_g1 + n_g2 - 1] - bar_h / 2 - 0.05
y_g3_top, y_g3_bot = y[n_g1 + n_g2] + bar_h / 2 + 0.05, y[-1] - bar_h / 2 - 0.05
for ax in (ax1, ax2):
    group_band(ax, y_g1_bot, y_g1_top, "")
    group_band(ax, y_g3_bot, y_g3_top, "")

fig.suptitle("Sensitivity tests", fontsize=13, fontweight="bold", y=0.995)
fig.text(0.01, 0.965,
         "MESA, 4-window average (train ∈ {2000–05, 05–10, 10–15, 15–20}, test=2021), DATA_VARIANT=avg unless noted",
         fontsize=8.8, color=GRAY_TEXT)

plt.tight_layout(rect=[0, 0.02, 1, 0.955])
out_png = "/glade/work/skygale/projects/SeaIceDownscaling/sensitivity/analysis/sensitivity_tests.png"
out_pdf = "/glade/work/skygale/projects/SeaIceDownscaling/sensitivity/analysis/sensitivity_tests.pdf"
fig.savefig(out_png, dpi=220, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print("wrote", out_png)
print("wrote", out_pdf)
