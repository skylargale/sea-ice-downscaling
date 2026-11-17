"""
Final round (2026-11-15): conservative-truth targets that also cover the 2021 test year, for the sensitivity test
"conservative truth" (training/submit_sensitivity_final2.sh, config construth).

training_data/conservative_targets/Y_*_conservative_2000_2020.nc are conservative only for 2000-2020 (bilinear elsewhere,
so 2021 = bilinear). The retired archive/retired/conservative_targets_2015_2021/Y_*_conservative_2015_2021.nc are
conservative for 2015-2021 with the same members, grid and time axis (identical values in 2016, checked). This splices
2021 from the 2015-2021 file into a copy of the 2000-2020 file:
  training_data/conservative_targets/Y_{MESA_HR,FOSI_HR_JRA55}_daily_conservative_2000_2021.nc
Usage: python build_conservative_2021.py [MESA FOSI]
"""
import os
import sys

import numpy as np
import xarray as xr

D = "/glade/derecho/scratch/skygale/downscaling"
NAMES = {"MESA": "MESA_HR", "FOSI": "FOSI_HR_JRA55"}
for ds in (sys.argv[1:] or ["MESA", "FOSI"]):
    n = NAMES[ds]
    a = xr.open_dataset(f"{D}/training_data/conservative_targets/Y_{n}_daily_conservative_2000_2020.nc", chunks={"time": 365})
    b = xr.open_dataset(f"{D}/archive/retired/conservative_targets_2015_2021/Y_{n}_daily_conservative_2015_2021.nc", chunks={"time": 365})
    assert np.array_equal(a.time.values, b.time.values) and a.Y.shape == b.Y.shape
    chk = a.Y.sel(time="2016-03").load().values, b.Y.sel(time="2016-03").load().values
    assert np.array_equal(*chk), "2015-2021 and 2000-2020 files differ in 2016: members or grid do not match"
    is21 = xr.DataArray(a.time.dt.year.values == 2021, dims="time", coords={"time": a.time})
    out = a.copy()
    out["Y"] = xr.where(is21, b.Y, a.Y).transpose(*a.Y.dims).astype(a.Y.dtype)
    out["Y"].attrs = a.Y.attrs
    out.attrs = dict(a.attrs)
    out.attrs["regrid_method"] = "conservative (2000-2021; bilinear elsewhere)"
    out.attrs["history_2026-11-15"] = ("2021 spliced from archive/retired/conservative_targets_2015_2021/"
                                       f"Y_{n}_daily_conservative_2015_2021.nc by processing/build_conservative_2021.py")
    path = f"{D}/training_data/conservative_targets/Y_{n}_daily_conservative_2000_2021.nc"
    out.to_netcdf(path + ".tmp")
    os.replace(path + ".tmp", path)
    v = xr.open_dataset(path)
    for yr, src in (("2010", a), ("2021", b)):
        same = np.array_equal(v.Y.sel(time=f"{yr}-02").values, src.Y.sel(time=f"{yr}-02").values)
        print(ds, yr, "matches", "2000-2020 file" if src is a else "2015-2021 file", same)
    print("wrote", path)
