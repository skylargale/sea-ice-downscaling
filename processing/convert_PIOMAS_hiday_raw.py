"""
convert_PIOMAS_hiday_raw.py

Converts the raw PIOMAS v2.1 daily sea ice thickness binaries
(hiday.H<year>, downloaded directly from
https://pscfiles.apl.washington.edu/zhang/PIOMAS/data/v2.1/hiday/, per
https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/model_grid)
into per-year NetCDF files on the PIOMAS native curvilinear grid.

Binary format (confirmed against file size: 120*360*4*365 bytes exactly):
little-endian float32, one (nlat=120, nlon=360) record per day, 365 records/
year (no Feb 29 -- same fixed 365-day convention as PIOMAS's own monthly
heff.H processing, see process_PIOMAS.ncl's daypermon; noleap calendar used
here for consistency with this project's other daily products).

Grid: reused directly from the existing official group mirror's grid.dat
(/glade/campaign/cgd/ccr/yeager/OBS/seaice/PIOMAS/grid.dat, same file
process_PIOMAS.ncl reads for the monthly product) -- ASCII, 2*120*360
values, reshape (2, nlat, nlon): block 0 = lon, block 1 = lat.

Output: /glade/derecho/scratch/skygale/observations/PIOMAS_daily/PIOMAS_hiday_<year>.nc,
dims (time=365, nlat=120, nlon=360), variable "hi" (m), plus 2D lat/lon.
"""

import sys
import numpy as np
import xarray as xr
from datetime import datetime, timezone

# Years from the command line, e.g. `python convert_PIOMAS_hiday_raw.py 2020 2021` (default: 2020-2022,
# the original conversion). Copied from Version5/processing 2026-09-24.
YEARS = [int(y) for y in sys.argv[1:]] or [2020, 2021, 2022]
RAW_DIR = "/glade/derecho/scratch/skygale/observations/PIOMAS_daily"
GRID_PATH = "/glade/campaign/cgd/ccr/yeager/OBS/seaice/PIOMAS/grid.dat"
NLAT, NLON = 120, 360
NDAYS = 365

grid_vals = np.loadtxt(GRID_PATH, dtype=np.float32).reshape(2, NLAT, NLON)
lon = grid_vals[0]
lat = grid_vals[1]
print("Grid loaded:", lat.shape, "lat range", lat.min(), lat.max(), "lon range", lon.min(), lon.max())

for year in YEARS:
    raw_path = f"{RAW_DIR}/hiday.H{year}"
    print(f"\n=== {year}: reading {raw_path} ===")
    data = np.fromfile(raw_path, dtype="<f4")
    expected = NDAYS * NLAT * NLON
    if data.size != expected:
        raise RuntimeError(f"{raw_path}: expected {expected} float32 values, got {data.size}")
    hi = data.reshape(NDAYS, NLAT, NLON)
    print("hi shape:", hi.shape, "min/max:", float(hi.min()), float(hi.max()),
          "mean:", float(hi.mean()))

    time = xr.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D",
                          calendar="noleap", use_cftime=True)
    if time.size != NDAYS:
        raise RuntimeError(f"noleap date range for {year} has {time.size} days, expected {NDAYS}")

    ds = xr.Dataset(
        {"hi": (["time", "nlat", "nlon"], hi.astype(np.float32))},
        coords={
            "time": time,
            "lat": (["nlat", "nlon"], lat),
            "lon": (["nlat", "nlon"], lon),
        },
    )
    ds["hi"].attrs = {"long_name": "PIOMAS v2.1 daily sea ice thickness", "units": "m"}
    ds.attrs["source"] = (
        "https://pscfiles.apl.washington.edu/zhang/PIOMAS/data/v2.1/hiday/hiday.H%d, "
        "downloaded directly (anonymous HTTP)" % year
    )
    ds.attrs["reference"] = "Zhang, Jinlun and D.A. Rothrock, Mon. Wea. Rev. 131(5), 681-697, 2003"
    ds.attrs["grid_source"] = GRID_PATH
    ds.attrs["calendar_note"] = (
        "365 fixed daily records/year, no Feb 29 in leap years -- noleap calendar assigned "
        "here for consistency with this project's other daily datasets (FOSI, JRA55, CDR)."
    )
    ds.attrs["created_by"] = "Sky Gale"
    ds.attrs["date_created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    out_path = f"{RAW_DIR}/PIOMAS_hiday_{year}.nc"
    ds.to_netcdf(out_path)
    print("Saved:", out_path)

print("\nDone.")
