"""
Paper Figure 7 (Observational application), final EngressNet (2026-10-07; T2 amplitude maps + 10 px energy tiles +
noise-parameter lr x10, FOSI-trained, 4 windows x 20 members) driven by PIOMAS, with ICESat-2 along-track data as truth.
Per user decision 2026-10-07: final model on PIOMAS input only, compared to ICESat-2 (supersedes build_fig7_piomas_application.py).

  (a) ensemble mean, 10 Mar 2021, with that day's ICESat-2 pass-cells (QC'd 0.1 deg cell means) overlaid on the same colour scale
  (b) one ensemble member, same day and overlay (what a single realization looks like)
  (c) along-track transect of the strongest-coverage pass that day (gt1r): ICESat-2 pass-cell mean vs ensemble 5-95 % / 25-75 %,
      ensemble mean, two members and bilinear PIOMAS
  (d) weekly mean over all co-located pass-cells in the domain, winters 2018/19-2021/22: ICESat-2 vs ensemble (member spread of the
      weekly mean, 5-95 %) and bilinear PIOMAS
  (e) all pass-cells: ensemble mean vs ICESat-2 (hexbin) with bias / RMSE / r / CRPS, bilinear PIOMAS for reference
ICESat-2 is floe thickness and EngressNet a grid-cell mean (see observations/README.md); same QC as the evaluation notebooks
(extent >= 2 km, >= 20 segments, median segment uncertainty <= 1 m).
Output: figures/final_figures/07b_piomas_vs_icesat2_evaluation.png (supplementary since 2026-10-08).
"""
import glob
import os
import shutil

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
BATCH = f"{V6}/results/PIOMAS_2018-2022_CAL1a_noiselr_FOSI"
IS2C = "/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells"
NPZ = f"{IS2C}/eval/cal1a_piomas_cells.npz"
OUT = f"{V6}/figures/final_figures/07b_piomas_vs_icesat2_evaluation.png"  # supplementary; Fig 7 = build_fig7_final_piomas_application.py
DAY = pd.Timestamp("2021-03-10")
GRANULE = "IS2SITDAT4_01_20210310204006_11741001_005_01_bnum5gt1r_001"
MEMBER_WINDOW, MEMBER = "2015-2020", 0
LAT = np.round(60.0 + 0.1 * np.arange(150), 1)
LON = np.round(178.1 + 0.1 * np.arange(310), 1)
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
C_ENS, C_BIL = "#2a78d6", "#eb6834"
CMAP, VMAX = "viridis", 3.0
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def crps_fair(X, y):
    K = X.shape[1]
    Xs = np.sort(X, 1)
    w = 2 * np.arange(1, K + 1) - K - 1
    return np.abs(X - y[:, None]).mean(1) - (Xs * w).sum(1) / (K * (K - 1))


def day_maps():
    """4-window ensemble mean and one member for DAY on the 150 x 310 sub-domain; land mask."""
    runs = sorted(glob.glob(f"{BATCH}/*/eval_data/fields.npz"))
    assert len(runs) == 4, runs
    ens, member, land = [], None, None
    for f in runs:
        t = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(f.replace("fields.npz", "sample_times.csv"))["time"]))
        k = int(np.where(t.normalize() == DAY)[0][0])
        z = np.load(f)
        ens.append(z["Y_pred_phys"][k, 0])
        if land is None:
            land = z["land_mask"][0, 0, 0:150, 80:390] > 0.5
        if MEMBER_WINDOW in f:
            member = z["preds_all_phys"][k, MEMBER, 0]
        del z
    E = np.mean(ens, 0)
    return np.where(land, np.nan, E), np.where(land, np.nan, member)


def main():
    pcd = pd.read_parquet(f"{IS2C}/pass_cells.parquet").query("in_domain").reset_index(drop=True)
    qc = ((pcd.extent_km >= 2.0) & (pcd.n_seg >= 20) & (pcd.unc_med <= 1.0)).values
    z = np.load(NPZ)
    M, bil = z["members"].astype(float), z["bilinear"].astype(float)
    ok = qc & np.isfinite(M).all(1) & np.isfinite(bil)
    mu = M.mean(1)

    E, m1 = day_maps()
    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], width_ratios=[1, 1, 1.25], hspace=0.28, wspace=0.18)
    proj = ccrs.NorthPolarStereo(central_longitude=-170)
    day = ok & (pcd.date.values == np.datetime64(DAY))
    for n, (A, ttl) in enumerate(((E, "(a) EngressNet ensemble mean"), (m1, "(b) one ensemble member"))):
        ax = fig.add_subplot(gs[0, n], projection=proj)
        ax.set_extent([178, 209, 60, 75], crs=ccrs.PlateCarree())
        pm = ax.pcolormesh(LON, LAT, A, transform=ccrs.PlateCarree(), cmap=CMAP, vmin=0, vmax=VMAX, shading="auto", rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor="#d9d8d3", zorder=2)
        ax.coastlines(lw=0.5, zorder=3)
        ax.scatter(pcd.lon.values[day], pcd.lat.values[day], c=pcd.h_mean.values[day], cmap=CMAP, vmin=0, vmax=VMAX, s=9,
                   edgecolors=INK, linewidths=0.35, transform=ccrs.PlateCarree(), zorder=4)
        ax.set_title(f"{ttl}, PIOMAS input, {DAY:%d %b %Y}\n(dots: ICESat-2 0.1° pass-cell means, same colour scale)", loc="left", fontsize=9)
    cb = fig.colorbar(pm, ax=[fig.axes[0], fig.axes[1]], orientation="horizontal", shrink=0.6, pad=0.04, aspect=40)
    cb.set_label("sea ice thickness (m)")

    # (c) transect
    ax = fig.add_subplot(gs[0, 2])
    g = ok & (pcd.granule.values == GRANULE)
    lat = pcd.lat.values[g]
    o = np.argsort(lat)
    lat, X, y, b = lat[o], M[g][o], pcd.h_mean.values[g][o], bil[g][o]
    ax.fill_between(lat, *np.percentile(X, [5, 95], 1), color=C_ENS, alpha=0.18, lw=0, label="EngressNet 5–95 %")
    ax.fill_between(lat, *np.percentile(X, [25, 75], 1), color=C_ENS, alpha=0.32, lw=0, label="EngressNet 25–75 %")
    for j, ls in ((0, "-"), (41, "-")):
        ax.plot(lat, X[:, j], color=C_ENS, lw=0.6, alpha=0.8, ls=ls, label="two members" if j == 0 else None)
    ax.plot(lat, X.mean(1), color=C_ENS, lw=1.8, label="EngressNet ensemble mean")
    ax.plot(lat, b, color=C_BIL, lw=1.4, ls="--", label="bilinear PIOMAS")
    ax.plot(lat, y, "o", color=INK, ms=2.8, label="ICESat-2 pass-cell mean")
    inside = ((y >= np.percentile(X, 5, 1)) & (y <= np.percentile(X, 95, 1))).mean()
    ax.set_xlabel("latitude along track (°N)")
    ax.set_ylabel("sea ice thickness (m)")
    ax.set_title(f"(c) ICESat-2 pass {DAY:%d %b %Y} (gt1r, {g.sum()} cells, 60.7–74.9°N)\n"
                 f"ICESat-2 inside the 5–95 % range at {inside:.0%} of cells", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    # (d) weekly co-located series
    ax = fig.add_subplot(gs[1, 0:2])
    d = pd.DataFrame({"date": pcd.date.values[ok], "is2": pcd.h_mean.values[ok], "bil": bil[ok]})
    Mo = pd.DataFrame(M[ok])
    d["week"] = pd.DatetimeIndex(d.date).to_period("W").start_time
    Mo["week"] = d["week"].values
    wk = d.groupby("week")[["is2", "bil"]].mean()
    mw = Mo.groupby("week").mean()  # per-member weekly means
    n = d.groupby("week").size()
    keep = n[n >= 50].index
    wk, mw = wk.loc[keep], mw.loc[keep]
    seas = np.where(wk.index.month >= 8, wk.index.year, wk.index.year - 1)
    for s_ in np.unique(seas):
        k = seas == s_
        t = wk.index[k]
        ax.fill_between(t, mw[k].quantile(0.05, axis=1), mw[k].quantile(0.95, axis=1), color=C_ENS, alpha=0.25, lw=0,
                        label="EngressNet 5–95 % (members)" if s_ == seas[0] else None)
        ax.plot(t, mw[k].mean(axis=1), color=C_ENS, lw=1.6, label="EngressNet ensemble mean" if s_ == seas[0] else None)
        ax.plot(t, wk.bil[k], color=C_BIL, lw=1.3, ls="--", label="bilinear PIOMAS" if s_ == seas[0] else None)
        ax.plot(t, wk.is2[k], "o-", color=INK, ms=2.5, lw=0.8, label="ICESat-2" if s_ == seas[0] else None)
    ax.set_ylabel("mean thickness over co-located cells (m)")
    ax.set_title("(d) Weekly mean over all co-located ICESat-2 pass-cells (weeks with ≥ 50 cells; ICESat-2 product Oct–Apr)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left", ncol=4)

    # (e) hexbin
    ax = fig.add_subplot(gs[1, 2])
    yy = pcd.h_mean.values[ok]
    hb = ax.hexbin(mu[ok], yy, gridsize=45, extent=(0, 5, 0, 5), mincnt=1, bins="log", cmap="Greys")
    ax.plot([0, 5], [0, 5], color=INK, lw=0.8)
    st = lambda x: (np.mean(x - yy), np.sqrt(np.mean((x - yy) ** 2)), np.corrcoef(x, yy)[0, 1])
    be, re, ce = st(mu[ok])
    bb, rb, cb_ = st(bil[ok])
    crps = crps_fair(M[ok], yy).mean()
    ax.text(0.03, 0.97, f"EngressNet ens. mean: bias {be:+.2f} m, RMSE {re:.2f} m, r {ce:.2f}\nEngressNet CRPS {crps:.3f} m"
            f"\nbilinear PIOMAS: bias {bb:+.2f} m, RMSE {rb:.2f} m, r {cb_:.2f}\n(MAE = its CRPS: {np.mean(np.abs(bil[ok] - yy)):.3f} m)",
            transform=ax.transAxes, va="top", fontsize=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlabel("EngressNet ensemble mean (m)")
    ax.set_ylabel("ICESat-2 pass-cell mean (m)")
    ax.set_title(f"(e) All pass-cells, Oct 2018–May 2022\n(N = {ok.sum():,})", loc="left", fontsize=9)
    fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02).set_label("cells per bin")

    fig.suptitle("Final EngressNet applied to PIOMAS, evaluated against ICESat-2 along-track thickness", x=0.01, ha="left",
                 fontsize=11, y=0.995)
    if os.path.exists(OUT) and not os.path.exists(OUT.replace(".png", "_previous.png")):
        shutil.copy(OUT, OUT.replace(".png", "_previous.png"))
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT, f"| CRPS {crps:.3f}, ens-mean bias {be:+.3f} RMSE {re:.3f} r {ce:.3f}; bilinear bias {bb:+.3f} RMSE {rb:.3f}")


if __name__ == "__main__":
    main()
