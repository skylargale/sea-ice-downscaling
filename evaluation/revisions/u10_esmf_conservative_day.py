"""
Revisions 2026-11, follow-up 3: one test day of MESACLIP U10 regridded with ESMF first-order conservative remapping (the
engine xESMF wraps; xESMF itself only builds conservative weights for structured grids) using the ne120 SCRIP file
/glade/campaign/cesm/cesmdata/cseg/inputdata/share/scripgrids/ne120np4_pentagons_100310.nc (native cell corners), onto the
paper's 1-deg grid (cell edges at +-0.5 deg), compared with the paper's center-in-cell average as stored in
X_MESA_HR_daily_avg.nc. Day 2021-03-01, member .004 (ensemble 0). Analysis only.
"""
import glob

import numpy as np
import xarray as xr

DAY = "2021-03-01"
SCRIP = "/glade/campaign/cesm/cesmdata/cseg/inputdata/share/scripgrids/ne120np4_pentagons_100310.nc"
X = xr.open_dataset("/glade/derecho/scratch/skygale/downscaling/training_data/X_MESA_HR_daily_avg.nc").X
ti = [i for i, t in enumerate(X.time.values) if str(t)[:10] == DAY][0]
xu = X.isel(time=ti, ensemble=0).sel(channel="U10").values
lat_c, lon_c = X.lat.values, X.lon.values % 360

f = sorted(glob.glob("/glade/campaign/collections/gdex/data/d651009/*-2006-2100.004/atm/proc/tseries/day_1/*.U10.2021*.nc"))[0]
ds = xr.open_dataset(f)
k = [i for i, t in enumerate(ds.time.values) if str(t)[:10] == DAY][0]
u = ds.U10.isel(time=k).values.astype(np.float64)  # (ncol,), SCRIP order (verified: centers reproduce X exactly)

# ESMF's own SCRIP reader segfaults on this NETCDF3 file (esmpy 8.9.1, conda build), so the same SCRIP cell polygons are passed
# to xESMF as an in-memory ESMF Mesh (xesmf.backend.Mesh.from_polygons) and xESMF builds the first-order conservative weights.
import xesmf as xe
from shapely.geometry import Polygon
s = xr.open_dataset(SCRIP)
clat, clon = s.grid_center_lat.values, s.grid_center_lon.values % 360
sel = (clat > lat_c.min() - 3) & (clat < lat_c.max() + 3) & (clon > lon_c.min() - 3) & (clon < lon_c.max() + 3)
cla, clo = s.grid_corner_lat.values[sel], s.grid_corner_lon.values[sel] % 360
polys = []
for o, a_ in zip(clo, cla):
    pts = list(dict.fromkeys(zip(o, a_)))  # pentagon-file quadrilaterals repeat their last corner
    polys.append(Polygon(pts))
src = xe.backend.Mesh.from_polygons(np.array(polys, dtype=object))
lon_e = np.r_[lon_c - 0.5, lon_c[-1] + 0.5]; lat_e = np.r_[lat_c - 0.5, lat_c[-1] + 0.5]
LON, LAT = np.meshgrid(lon_c, lat_c); LONB, LATB = np.meshgrid(lon_e, lat_e)
ds_out = xr.Dataset({"lon": (("y", "x"), LON), "lat": (("y", "x"), LAT),
                     "lon_b": (("y_b", "x_b"), LONB), "lat_b": (("y_b", "x_b"), LATB)})
dst, shape_out, _ = xe.frontend.ds_to_ESMFgrid(ds_out, need_bounds=True)
rg = xe.frontend.BaseRegridder(src, dst, "conservative", unmapped_to_nan=True)
W = rg.weights.data
W = W.tocsr() if hasattr(W, "tocsr") else __import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix(W.to_scipy_sparse())
nlat, nlon = len(lat_c), len(lon_c)
assert W.shape[0] == nlat * nlon, (W.shape, shape_out)
cons = np.asarray(W @ u[sel]).reshape(nlat, nlon)  # xESMF flattens the (y, x) destination in C order
cons[np.asarray(W.sum(1)).ravel().reshape(nlat, nlon) < 0.999] = np.nan
print(f"xESMF conservative weights: {W.nnz} overlaps from {sel.sum()} ne120 cells", flush=True)

d = cons - xu
print(f"{DAY}, member .004, {nlat}x{nlon} 1-deg cells (60-80N, 170-220E)")
print(f"ESMF conservative (SCRIP corners) vs paper center-in-cell average: RMSE {np.sqrt(np.nanmean(d ** 2)):.3f} m/s, "
      f"mean diff {np.nanmean(d):+.3f}, mean |diff| {np.nanmean(np.abs(d)):.3f}, max |diff| {np.nanmax(np.abs(d)):.3f} m/s "
      f"(domain-mean U10 {np.nanmean(xu):.2f} m/s, spatial std {np.nanstd(xu):.2f} m/s)")
a, b = np.unravel_index(np.nanargmax(np.abs(d)), d.shape)
print(f"largest difference at {lat_c[a]:.0f}N {lon_c[b]:.0f}E: paper {xu[a, b]:.2f}, conservative {cons[a, b]:.2f} m/s")
print(f"cells with |diff| > 0.5 m/s: {np.mean(np.abs(d) > 0.5) * 100:.1f}%, > 1 m/s: {np.mean(np.abs(d) > 1) * 100:.1f}%")
