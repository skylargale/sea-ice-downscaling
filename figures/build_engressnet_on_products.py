"""
Final EngressNet (T2 amplitude maps + 10 px energy tiles + noise lr x10, FOSI-trained, 4 windows x 20 members) applied to each
observational thickness product as input, one winter day (2026-10-08, to look at).

Rows: input product (PIOMAS, CS2SMOS, TOPAZ4-ML, Petty GPSat). Columns: (1) the coarse input as EngressNet receives it (bilinear
from the 1 deg input grid), (2) ensemble mean of all 80 members with that day's QC'd ICESat-2 pass-cells overlaid (same colour
scale), (3) one member (2015-2020 window, member 0), (4) ensemble spread (std over all 80 members).
TOPAZ4-ML's model domain is closed at Bering Strait, so its input south of 66N was filled from PIOMAS (dotted line); Petty GPSat is
built from ICESat-2 freeboards, so it is not independent of the overlaid tracks.
Output: figures/final_figures/engressnet_on_products_<date>.png
"""
import glob
import os
import pickle
import sys

import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(BASE, "training"))
sys.path.insert(0, os.path.join(BASE, "evaluation"))
import functions_engressnet as fe  # noqa: E402

DAY = pd.Timestamp(os.environ.get("DAY", "2021-03-10"))
IS2C = "/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells"
INPUTS = [("PIOMAS", "PIOMAS_2018-2022_CAL1a_noiselr_FOSI"), ("CS2SMOS", "INPUTSWAP_c2s_smos_cal1a_4win"),
          ("TOPAZ4-ML", "INPUTSWAP_topaz4ml_cal1a_4win"), ("Petty GPSat (ICESat-2-based)", "INPUTSWAP_petty_gpsat_cal1a_4win")]
MEMBER_WINDOW = "2015-2020"


def load(batch):
    """Day DAY: bilinear input, 80 members (4 windows x 20), one member; None if the batch or day is missing."""
    runs = sorted(glob.glob(f"{BASE}/results/{batch}/*/eval_data/fields.npz"))
    if len(runs) != 4:
        print(f"{batch}: {len(runs)} runs, skipped"); return None
    mem, base, one, geo = [], None, None, None
    for f in runs:
        t = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(f.replace("fields.npz", "sample_times.csv"))["time"])).normalize()
        k = np.where(t == DAY)[0]
        if not len(k):
            print(f"{batch}: no {DAY:%Y-%m-%d}"); return None
        k = int(k[0])
        z = np.load(f)
        P = z["preds_all_phys"][k, :, 0]
        mem.append(P)
        if base is None:
            base = z["Y_base_phys"][k, 0]
            ocean = z["mask_test"][0, 0] <= 0.5
            with open(f.replace("fields.npz", "tile_geometry.pkl"), "rb") as g:
                tg = pickle.load(g)
            geo = (np.asarray(tg[0]["target_lat"]), np.asarray(tg[0]["target_lon"]), ocean)
        if MEMBER_WINDOW in f:
            one = P[0]
        del z
    M = np.concatenate(mem, 0)
    lat, lon, ocean = geo
    w = lambda a: np.where(ocean, a, np.nan)
    return dict(base=w(base), mean=w(M.mean(0)), one=w(one), std=w(M.std(0, ddof=1)), lat=lat, lon=lon)


def main():
    pc = pd.read_parquet(f"{IS2C}/pass_cells.parquet").query("in_domain")
    pc = pc[(pc.extent_km >= 2.0) & (pc.n_seg >= 20) & (pc.unc_med <= 1.0) & (pc.date == DAY)]
    data = [(name, load(b)) for name, b in INPUTS]
    data = [(n, d) for n, d in data if d is not None]
    plot_bbox = {"lon_min": -182, "lon_max": -151, "lat_min": 60, "lat_max": 75}
    proj, boundary, _ = fe.make_polar_proj(plot_bbox)
    cols = ["Input as EngressNet sees it\n(1° product, bilinear)", "EngressNet ensemble mean\n(dots: ICESat-2 that day)",
            "One ensemble member", "Ensemble spread (std)"]
    fig = plt.figure(figsize=(16, 4.1 * len(data)))
    for r, (name, d) in enumerate(data):
        lon2d, lat2d = np.meshgrid(d["lon"], d["lat"])
        for c, key in enumerate(["base", "mean", "one", "std"]):
            ax = fig.add_subplot(len(data), 4, 4 * r + c + 1, projection=proj)
            fe.style_polar_ax(ax, proj, boundary, plot_bbox)
            if key == "std":
                pm_s = ax.pcolormesh(lon2d, lat2d, d[key], transform=ccrs.PlateCarree(), cmap="Purples", vmin=0, vmax=0.6,
                                     shading="auto", rasterized=True)
            else:
                pm = ax.pcolormesh(lon2d, lat2d, d[key], transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=2.5,
                                   shading="auto", rasterized=True)
            if key == "mean" and len(pc):
                ax.scatter(((pc.lon.values + 180) % 360) - 180, pc.lat.values, c=pc.h_mean.values, cmap="Blues", vmin=0, vmax=2.5,
                           s=6, linewidths=0.25, edgecolors="#3a3a3a", transform=ccrs.PlateCarree(), zorder=5)
            if name.startswith("TOPAZ"):
                ax.plot(np.linspace(-182, -151, 50), np.full(50, 66.0), ls=":", color="#b3261e", lw=1.2, transform=ccrs.PlateCarree(), zorder=6)
                if c == 0:
                    ax.text(-160, 64.2, "PIOMAS-filled", color="#b3261e", fontsize=8, transform=ccrs.PlateCarree(), zorder=6)
            if r == 0:
                ax.set_title(cols[c], fontsize=12)
            if c == 0:
                ax.text(-0.06, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=13)
    axes = fig.axes
    cb = fig.colorbar(pm, ax=[a for i, a in enumerate(axes) if i % 4 != 3], orientation="horizontal", aspect=50, shrink=0.8, pad=0.02)
    cb.set_label("Sea ice thickness (m)", fontsize=13)
    cb2 = fig.colorbar(pm_s, ax=[a for i, a in enumerate(axes) if i % 4 == 3], orientation="horizontal", aspect=14, shrink=0.9, pad=0.02)
    cb2.set_label("Ensemble std (m)", fontsize=13)
    fig.suptitle(f"Final EngressNet applied to each observational product, {DAY:%d %B %Y} (80 members: 4 training windows × 20)",
                 fontsize=14, y=1.0)
    out = f"{BASE}/figures/final_figures/engressnet_on_products_{DAY:%Y-%m-%d}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out, "| rows:", [n for n, _ in data], "| ICESat-2 cells that day:", len(pc))


if __name__ == "__main__":
    main()
