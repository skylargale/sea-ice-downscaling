"""
CICE sea-ice-thickness variability ceiling, as a simple bar chart (2026-10-08, advisor meeting; revised same day).

Quantity: sub-grid variability = how much 0.1 deg thickness varies inside one 1 deg cell on one day. 1 deg is EngressNet's input
grid (the coarse PIOMAS / CICE cells it downscales), so this is exactly the variability the downscaling has to add: the input itself
carries none of it (bilinear PIOMAS ~ 0).

Method (like-for-like): QC'd ICESat-2 pass-cell means (same QC as the evaluation notebooks), Oct 2018 - Dec 2021 (the years the CICE
truth covers), grouped by (day, 1 deg input cell) with >= 6 pass-cells. Every source is sampled at exactly those pass-cells on the
same day. Variability = pooled within-cell standard deviation, the standard within-group statistic:
    s_pooled^2 = sum_i (n_i - 1) s_i^2 / sum_i (n_i - 1)
over all groups i (n_i pass-cells, sample variance s_i^2). Ensembles / multi-window models (EngressNet 80 members, det-MSE 4 training
windows, MESACLIP 6 members): pooled variance of each member, averaged over members, then sqrt (the variability of one realization).
ICESat-2: its random measurement-noise variance (per thickness bin, from the along-track structure function) is pooled the same way
and subtracted, so the bar is real thickness variability. Percentages = ratio to ICESat-2.
Output: figures/final_figures/cice_ceiling_bar.png (copy: observations/figures/15b_cice_ceiling_bar.png).
"""
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, f"{V6}/observations/processing")
import calibration_options_compare as c  # noqa: E402  (shared ICESat-2 pass-cell tools: QC masks, noise model)

OUT = f"{V6}/figures/final_figures/cice_ceiling_bar.png"
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
    eng, engb = np.load(f"{ev}/cal1a_piomas_cells.npz"), np.load(f"{ev}/cal1a_piomas_bins.npz")
    det = np.load(f"{ev}/detmse_FOSI_cells.npz")
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
        f"{V6}/observations/data/alongtrack_eval/cice_ceiling_bar.csv")

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
             "day. ICESat-2: measurement noise removed.\nEngressNet: average over its 80 ensemble members; det-MSE: over its 4 "
             "training windows; MESACLIP: over its 6 members.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT, dpi=250, bbox_inches="tight")
    shutil.copy(OUT, f"{V6}/observations/figures/15b_cice_ceiling_bar.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
