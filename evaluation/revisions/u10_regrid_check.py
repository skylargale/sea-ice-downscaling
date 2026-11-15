import glob, numpy as np, xarray as xr, pandas as pd
from shapely.geometry import Polygon, box
DAY = "2021-03-01"
X = xr.open_dataset("/glade/derecho/scratch/skygale/downscaling/training_data/X_MESA_HR_daily_avg.nc").X
print("X dims", X.dims, "channels", X.channel.values if "channel" in X.coords else None)
ti = [i for i, t in enumerate(X.time.values) if str(t)[:10] == DAY]; assert len(ti) == 1, ti
xu = X.isel(time=ti[0], ensemble=0)
print("X time label", str(X.time.values[ti[0]]))
ci = list(X.channel.values).index("U10") if X.channel.dtype.kind in "OUS" else 2
xu = xu.isel(channel=ci).values
lat_d, lon_d = X.lat.values, X.lon.values
print("X grid", lat_d.min(), lat_d.max(), lon_d.min(), lon_d.max(), xu.shape)
d = glob.glob("/glade/campaign/collections/gdex/data/d651009/*-2006-2100.004")[0]
f = [p for p in sorted(glob.glob(f"{d}/atm/proc/tseries/day_1/*.U10.*.nc"))]
ds = xr.open_mfdataset(f, combine="by_coords") if False else None
for p in f:
    t = xr.open_dataset(p, decode_times=True).time
    if str(t.values[0])[:4] <= "2021" <= str(t.values[-1])[:4]:
        fp = p; break
print("native file", fp.split("/")[-1])
nat = xr.open_dataset(fp).U10
tt = pd.to_datetime([str(v)[:10] for v in nat.time.values])
k = int(np.where(tt == pd.Timestamp(DAY))[0][0]); print("native time label", str(nat.time.values[k]), "(CESM daily stamp = end of day)")
scrip = xr.open_dataset("/glade/p/cesmdata/cseg/inputdata/share/scripgrids/ne120np4_pentagons_100310.nc")
clat, clon = scrip.grid_center_lat.values, scrip.grid_center_lon.values % 360
sel = (clat > lat_d.min() - 2) & (clat < lat_d.max() + 2) & (clon > lon_d.min() - 2) & (clon < lon_d.max() + 2)
res = {}
for shift in (0, -1, 1):  # check neighbouring native days too (time-stamp convention)
    u = nat.isel(time=k + shift).values[sel]
    la, lo = clat[sel], clon[sel]
    # (1) center-in-cell bin average
    li = np.round(la - lat_d[0]).astype(int); oi = np.round(lo - lon_d[0]).astype(int)
    ok = (li >= 0) & (li < len(lat_d)) & (oi >= 0) & (oi < len(lon_d))
    s = np.zeros(xu.shape); n = np.zeros(xu.shape)
    np.add.at(s, (li[ok], oi[ok]), u[ok]); np.add.at(n, (li[ok], oi[ok]), 1)
    binavg = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    # (3) nearest neighbour to cell centre
    LA, LO = np.meshgrid(lat_d, lon_d, indexing="ij")
    nn = np.empty(xu.shape)
    for a in range(len(lat_d)):
        for b in range(len(lon_d)):
            dd = (la - LA[a, b]) ** 2 + ((lo - LO[a, b]) * np.cos(np.deg2rad(LA[a, b]))) ** 2
            nn[a, b] = u[np.argmin(dd)]
    res[shift] = (binavg, nn, u)
    print(f"day shift {shift:+d}: max|X - binavg| {np.nanmax(np.abs(xu - binavg)):.2e}  max|X - nearest| {np.nanmax(np.abs(xu - nn)):.2e}")
# (2) true conservative (area overlap of SCRIP corner polygons with 1-deg boxes, lon/lat plane) for the best day
binavg, nn, u = res[0]
cla, clo = scrip.grid_corner_lat.values[sel], scrip.grid_corner_lon.values[sel] % 360
polys = [Polygon(zip(o, a)) for o, a in zip(clo, cla)]
cons = np.full(xu.shape, np.nan)
for a, la0 in enumerate(lat_d):
    for b, lo0 in enumerate(lon_d):
        bx = box(lo0 - .5, la0 - .5, lo0 + .5, la0 + .5)
        w = np.array([p.intersection(bx).area if p.intersects(bx) else 0.0 for p in polys])
        if w.sum() > 0:
            cons[a, b] = (w * u).sum() / w.sum()
print(f"day shift +0: max|X - conservative(corners)| {np.nanmax(np.abs(xu - cons)):.3f} m/s, mean|.| {np.nanmean(np.abs(xu - cons)):.3f};"
      f" binavg vs conservative mean|diff| {np.nanmean(np.abs(binavg - cons)):.3f} m/s, max {np.nanmax(np.abs(binavg - cons)):.3f}; domain-mean U10 {np.nanmean(xu):.2f} m/s")
dB = np.abs(xu - binavg); dC = np.abs(xu - cons)
print(f"binavg: mean|diff| {np.nanmean(dB):.4f}, cells |diff|>1e-3: {(dB>1e-3).sum()} of {dB.size}, >0.05: {(dB>0.05).sum()}")
for a, b in zip(*np.where(dB > 0.05)):
    print(f"   lat {lat_d[a]:.0f} lon {lon_d[b]:.0f}: X {xu[a,b]:.3f} bin {binavg[a,b]:.3f} cons {cons[a,b]:.3f} n_centres {int(n[a,b])}")
# re-bin exactly like the code (np.digitize on half-degree edges, regrid domain 55-85 / 160-230, clip)
