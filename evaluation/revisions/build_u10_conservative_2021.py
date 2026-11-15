"""
Revisions 2026-11, C.2 follow-up (DJ: ne120 cell corners): an alternative MESACLIP test-year X file whose U10 channel is
conservatively remapped from the native ne120 mesh using the SCRIP cell corners, instead of the center-in-cell bin average
the paper used. Everything else is copied from X_MESA_HR_daily_avg.nc.

Conservative weights: area of overlap between each ne120 cell polygon (grid_corner_lat/lon of ne120np4_pentagons_100310.nc)
and each 1-deg destination cell, computed in the lon-lat plane (shapely) and multiplied by cos(lat) of the destination cell
centre (the cos factor cancels within a cell; the overlap is what matters). Destination value = overlap-weighted mean.
Before swapping, every member's center-in-cell average is recomputed and must match the existing X file (validates member
order and day alignment).
Output: /glade/derecho/scratch/skygale/downscaling/training_data/X_MESA_HR_daily_avg_2021_U10conservative.nc (2021 only, same layout).
"""
import glob

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
from shapely import STRtree
from shapely.geometry import Polygon, box

D = "/glade/derecho/scratch/skygale/downscaling"
X_PATH = f"{D}/training_data/X_MESA_HR_daily_avg.nc"
OUT = f"{D}/training_data/X_MESA_HR_daily_avg_2021_U10conservative.nc"
SCRIP = "/glade/p/cesmdata/cseg/inputdata/share/scripgrids/ne120np4_pentagons_100310.nc"
MEMBERS = ["004", "005", "006", "007", "008", "010"]  # RCP8.5 members used for X (build_X_Y_from_MESA-HR_daily_rcp85.py)

X = xr.open_dataset(X_PATH)
ti = np.array([i for i, t in enumerate(X.time.values) if t.year == 2021])
X21 = X.isel(time=ti).load()
lat_d, lon_d = X21.lat.values, X21.lon.values % 360
ci = list(X21.channel.values).index("U10")
print("X 2021:", dict(X21.sizes), "U10 channel", ci, "grid", lat_d.min(), lat_d.max(), lon_d.min(), lon_d.max(), flush=True)

s = xr.open_dataset(SCRIP)
clat, clon = s.grid_center_lat.values, s.grid_center_lon.values % 360
sel = (clat > lat_d.min() - 2) & (clat < lat_d.max() + 2) & (clon > lon_d.min() - 2) & (clon < lon_d.max() + 2)
src_idx = np.flatnonzero(sel)
cla, clo = s.grid_corner_lat.values[sel], s.grid_corner_lon.values[sel] % 360
polys = [Polygon(zip(o, a)) for o, a in zip(clo, cla)]
tree = STRtree(polys)
rows, cols, vals = [], [], []
for a, la0 in enumerate(lat_d):
    for b, lo0 in enumerate(lon_d):
        bx = box(lo0 - .5, la0 - .5, lo0 + .5, la0 + .5)
        for j in tree.query(bx):
            ar = polys[j].intersection(bx).area
            if ar > 0:
                rows.append(a * len(lon_d) + b); cols.append(j); vals.append(ar)
W = csr_matrix((vals, (rows, cols)), shape=(len(lat_d) * len(lon_d), len(polys)))
W = csr_matrix(W.multiply(1.0 / np.asarray(W.sum(1)).ravel()[:, None]))
print(f"conservative weights: {W.nnz} overlaps, {W.nnz / W.shape[0]:.1f} source cells per destination", flush=True)
# center-in-cell bin average operator (the paper's method), for the member/day check
li = np.clip(np.digitize(clat[sel], np.r_[lat_d - .5, lat_d[-1] + .5]) - 1, -1, len(lat_d))
lj = np.clip(np.digitize(clon[sel], np.r_[lon_d - .5, lon_d[-1] + .5]) - 1, -1, len(lon_d))
ok = (li >= 0) & (li < len(lat_d)) & (lj >= 0) & (lj < len(lon_d))
B = csr_matrix((np.ones(ok.sum()), (li[ok] * len(lon_d) + lj[ok], np.flatnonzero(ok))), shape=W.shape)
B = csr_matrix(B.multiply(1.0 / np.maximum(np.asarray(B.sum(1)).ravel(), 1)[:, None]))

new = X21.X.values.copy()
for m, tag in enumerate(MEMBERS):
    f = [p for p in sorted(glob.glob(f"/glade/campaign/collections/gdex/data/d651009/*-2006-2100.{tag}/atm/proc/tseries/day_1/*.U10.*.nc"))]
    days = []
    for p in f:
        ds = xr.open_dataset(p)
        yrs = np.array([t.year for t in ds.time.values])
        if (yrs == 2021).any():
            k = np.flatnonzero(yrs == 2021)
            days.append(ds.U10.isel(time=k).values[:, src_idx])
            dates = pd.to_datetime([str(t)[:10] for t in ds.time.values[k]])
    U = np.concatenate(days)
    assert U.shape[0] == 365, (tag, U.shape)
    binavg = (B @ U.T).T.reshape(365, len(lat_d), len(lon_d))
    cons = (W @ U.T).T.reshape(365, len(lat_d), len(lon_d))
    x_old = X21.X.values[m, :, ci]
    match = np.mean(np.abs(binavg - x_old) < 1e-3)
    print(f"member .{tag} (ensemble {m}): center-in-cell reproduces X on {100 * match:.2f}% of cells "
          f"(mean |diff| {np.abs(binavg - x_old).mean():.4f} m/s); conservative - X: mean {np.mean(cons - x_old):+.3f}, "
          f"mean |.| {np.abs(cons - x_old).mean():.3f}, max |.| {np.abs(cons - x_old).max():.2f} m/s", flush=True)
    assert match > 0.98, f"member order / day alignment check failed for .{tag}"
    new[m, :, ci] = cons.astype(np.float32)

# full time axis (the pipeline indexes test X and test Y with the same time mask), only 2021 U10 replaced
out = X.load().copy()
out["X"].values[:, ti, ci] = new[:, :, ci]
out.attrs.update({"description": "X_MESA_HR_daily_avg.nc with 2021 U10 conservatively remapped from ne120 using "
                                 "the SCRIP cell corners (revisions 2026-11 sensitivity check)",
                  "source_script": "evaluation/revisions/build_u10_conservative_2021.py"})
out.to_netcdf(OUT)
print("wrote", OUT)
