"""
Input-swap test (2026-10-06): EngressNet run on observational SIT products instead of PIOMAS as the coarse thickness
input, scored against the ICESat-2 pass-cells. This builds the inputs.

X = the PIOMAS-obs X (X_PIOMAS_obs_2017-2022_cdrv6_daily_v4_avg_coastfill.nc: hi_d, CDR v6 aice_d, JRA55 u_10/v_10 on the
1 deg grid) with only hi_d replaced by the product, on the days the product covers (2018-2022):
  topaz4ml     TOPAZ4-ML sit x sic                        (floe -> grid-cell mean, the quantity the network was trained on)
  c2s_smos     CS2SMOS thickness x SIC (status 0), 0 over open ocean (status 1)   (same conversion)
  petty_gpsat  GPSat ice_thickness (already a grid-cell mean; negatives -> 0)
Product cells are averaged into the 1 deg X cells by cell centre (equal weights; the products are 12.5-25 km).
Gap filling per day, on FOSI-ocean 1 deg cells (FOSI aice ever > 0, the coastfill definition): CDR aice < 0.15 -> 0; else
3 x 3-neighbour means (3 passes); anything still missing -> the PIOMAS value (counted). A day is kept only if the product
itself covers >= 50 % of the ice-covered (CDR >= 0.15) cells of the context crop north of Bering Strait (lat 66-75,
lon 178-209; TOPAZ4-ML has no Bering Sea data, which is PIOMAS-filled).
Y = the PIOMAS-obs Y on the same days (only a placeholder truth; scoring is against ICESat-2).
Outputs (scratch downscaling/obs_eval_tests/): X_<p>_obs_2018-2022_daily_avg.nc, Y_<p>_obs_2018-2022_placeholder.nc,
input_swap_fill_<p>.csv (per-day coverage / fill fractions).

usage: python build_input_swap.py topaz4ml c2s_smos petty_gpsat
"""
import glob
import io
import os
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

D = "/glade/derecho/scratch/skygale/downscaling/obs_eval_tests"
RAW = "/glade/derecho/scratch/skygale/observations"
XP = f"{D}/X_PIOMAS_obs_2017-2022_cdrv6_daily_v4_avg_coastfill.nc"
YP = f"{D}/Y_PIOMAS_obs_2017-2022_cdrv6_daily_v2.nc"
FOSI_X = "/glade/derecho/scratch/skygale/downscaling/X_FOSI_HR_JRA55_daily_avg.nc"
LAT0, LON0, NJ, NI = 60.0, 170.0, 21, 51
# Coverage / fill accounting region: the context crop north of Bering Strait (lat 66-75, lon 178-209). TOPAZ4-ML's
# model domain is closed at Bering Strait (no data in the Bering Sea), so it is judged, and later scored, where every
# input is its own data; its Bering Sea cells are PIOMAS-filled (counted only outside this region).
CROP = (slice(6, 16), slice(8, 40))


def _bin(lat, lon, h):
    """Mean of finite product values whose cell centres fall in each 1 deg X cell -> (21, 51), count."""
    j = np.round(np.asarray(lat) - LAT0).astype(int)
    i = np.round(np.asarray(lon) % 360 - LON0).astype(int)
    ok = (j >= 0) & (j < NJ) & (i >= 0) & (i < NI) & np.isfinite(h)
    k = j[ok] * NI + i[ok]
    s = np.bincount(k, weights=h[ok], minlength=NJ * NI)
    n = np.bincount(k, minlength=NJ * NI)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(NJ, NI), n.reshape(NJ, NI)


def frames(product):
    """Yield (date, eff_thickness 2-D, lat, lon) on the product's native grid, 2018-2022."""
    if product == "topaz4ml":
        for z in sorted(glob.glob(f"{RAW}/TOPAZ4ML/20[12][0-9].zip")):
            if not 2018 <= int(os.path.basename(z)[:4]) <= 2022:
                continue
            import netCDF4
            zf = zipfile.ZipFile(z)
            for n in sorted(n for n in zf.namelist() if n.endswith(".nc")):
                ds = xr.open_dataset(xr.backends.NetCDF4DataStore(netCDF4.Dataset(n, memory=zf.read(n))))
                t = pd.Timestamp(datetime.strptime(os.path.basename(n)[-11:-3], "%Y%m%d"))
                sit, sic = ds["sit"].values, ds["sic"].values
                eff = np.where(np.isfinite(sit) & np.isfinite(sic), sit * sic, np.where(np.isfinite(sic) & (sic < 0.15), 0.0, np.nan))
                yield t, eff, ds["latitude"].values, ds["longitude"].values
    elif product == "c2s_smos":
        for f in sorted(glob.glob(f"{RAW}/C2S_SMOS_AWI_v300/20[12][0-9]/[0-9][0-9]/*.nc")):
            ds = xr.open_dataset(f)
            t = pd.Timestamp(ds["time"].values[0]).normalize()
            if not 2018 <= t.year <= 2022:
                continue
            h, sic, st = ds["sea_ice_thickness"].values[0], ds["sea_ice_concentration"].values[0] / 100.0, ds["status_flag"].values[0]
            eff = np.where(st == 0, h * sic, np.where(st == 1, 0.0, np.nan))
            yield t, eff, ds["lat"].values, ds["lon"].values
    elif product == "petty_gpsat":
        for f in sorted(glob.glob(f"{RAW}/Petty_IS2SMOS_GPSat/GPSat_multivar_*.nc")):
            if not 2018 <= int(os.path.basename(f)[-9:-5]) <= 2022:
                continue
            ds = xr.open_dataset(f)
            la, lo = ds["latitude"].values, ds["longitude"].values
            for k, t in enumerate(ds["time"].values):
                h = ds["ice_thickness"].values[k]
                yield pd.Timestamp(t).normalize(), np.where(h < 0, 0.0, h), la, lo
    else:
        raise ValueError(product)


def build(product):
    X = xr.open_dataset(XP)["X"]
    Y = xr.open_dataset(YP)["Y"]
    xkeys = {(t.year, t.month, t.day): k for k, t in enumerate(X["time"].values)}
    aice = X.isel(ensemble=0).sel(channel="aice_d").values           # (T, 21, 51) CDR v6
    hpio = X.isel(ensemble=0).sel(channel="hi_d").values             # PIOMAS fallback
    ocean = (xr.open_dataset(FOSI_X)["X"].isel(ensemble=0).sel(channel="aice_d").max("time") > 0).values
    keep, rows = [], []
    Xv = X.values.copy()
    for t, eff, la, lo in frames(product):
        key = (t.year, t.month, t.day)
        if key not in xkeys:  # Feb 29 / outside 2017-2022
            continue
        k = xkeys[key]
        hb, n = _bin(la, lo, eff)
        a = aice[k]
        ice = ocean & (a >= 0.15)
        direct = np.isfinite(hb) & ocean
        cov = (direct & ice)[CROP].sum() / max(ice[CROP].sum(), 1)
        h = np.where(ocean, hb, np.nan)
        h[ocean & ~np.isfinite(h) & (a < 0.15)] = 0.0
        nb = 0
        for _ in range(3):
            miss = ocean & ~np.isfinite(h)
            if not miss.any():
                break
            p = np.pad(h, 1, constant_values=np.nan)
            stack = np.stack([p[1 + dj:1 + dj + NJ, 1 + di:1 + di + NI] for dj in (-1, 0, 1) for di in (-1, 0, 1) if dj or di])
            with np.errstate(invalid="ignore"):
                m = np.nanmean(stack, 0)
            fill = miss & np.isfinite(m)
            h[fill] = m[fill]
            nb += int((fill & ice)[CROP].sum())
        rest = ocean & ~np.isfinite(h)
        npio = int((rest & ice)[CROP].sum())
        h[rest] = hpio[k][rest]
        h[~ocean] = hpio[k][~ocean]
        rows.append({"date": t, "coverage_ice_crop": cov, "n_ice_crop": int(ice[CROP].sum()), "n_neighbour_fill": nb,
                     "n_piomas_fill": npio, "kept": cov >= 0.5})
        if cov >= 0.5 and k not in keep:
            Xv[0, k, 0] = np.clip(h, 0, None)
            keep.append(k)
    keep = sorted(keep)
    stats = pd.DataFrame(rows)
    stats.to_csv(f"{D}/input_swap_fill_{product}.csv", index=False)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    Xo = X.copy(data=Xv).isel(time=keep)
    Xo.attrs.update(description=f"X_PIOMAS_obs cdrv6 coastfill with hi_d replaced by {product} (grid-cell-mean thickness, 1 deg "
                                f"cell-centre averages, gap-filled; see build_input_swap.py) on {len(keep)} covered days 2018-2022",
                    date_created=stamp)
    Xo.to_dataset(name="X").to_netcdf(f"{D}/X_{product}_obs_2018-2022_daily_avg.nc")
    Yo = Y.isel(time=keep)
    Yo.attrs.update(description="PIOMAS-obs Y on the same days; placeholder truth only (input-swap runs are scored against ICESat-2)",
                    date_created=stamp)
    Yo.to_dataset(name="Y").to_netcdf(f"{D}/Y_{product}_obs_2018-2022_placeholder.nc")
    k_ = stats[stats.kept]
    print(f"{product}: {len(stats)} product days, {len(keep)} kept (coverage >= 50%); median coverage of kept days "
          f"{k_.coverage_ice_crop.median():.2f}; neighbour-filled {k_.n_neighbour_fill.sum() / max(k_.n_ice_crop.sum(), 1):.1%}, "
          f"PIOMAS-filled {k_.n_piomas_fill.sum() / max(k_.n_ice_crop.sum(), 1):.1%} of ice-covered crop cells; "
          f"years {sorted({X['time'].values[i].year for i in keep})}", flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        build(p)
