"""
Calibration option 2 (2026-10-07): ICESat-2 pass-cell targets for freezing EngressNet's mean and calibrating only its
noise parameters against observations (--train-only-noise --obs-ft-only; pixel CRPS where observed).

Same QC'd pass-cells as the evaluation notebooks (extent >= 2 km, >= 20 segments, median segment uncertainty <= 1 m), mean
reference (segment-length-weighted pass-cell mean; several passes over a cell on one day are averaged), on the 150 x 310
0.1 deg sub-domain grid, NaN where unobserved. Two folds split by winter season (Oct-May), so no calibration day is adjacent
to an evaluation day:
  fold A: calibrate on 2018/19 + 2020/21  -> score on 2019/20 + 2021/22
  fold B: calibrate on 2019/20 + 2021/22  -> score on 2018/19 + 2020/21
Output: /glade/derecho/scratch/skygale/downscaling/obs_eval_tests/T_IS2cells_mean_fold{A,B}.nc, target(time, lat, lon).
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "/glade/work/skygale/projects/SeaIceDownscaling/observations/processing")
from obs_common import UNC_MAX

D = "/glade/derecho/scratch/skygale/downscaling/obs_eval_tests"
PC = "/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells/pass_cells.parquet"
LAT = np.round(60.0 + 0.1 * np.arange(150), 1)
LON = np.round(178.1 + 0.1 * np.arange(310), 1)
FOLDS = {"A": (2018, 2020), "B": (2019, 2021)}  # calibration seasons (start year of the Oct-May winter)


def season(dates):
    return np.where(dates.month >= 8, dates.year, dates.year - 1)


def main():
    d = pd.read_parquet(PC).query("in_domain")
    d = d[(d.extent_km >= 2.0) & (d.n_seg >= 20) & (d.unc_med <= UNC_MAX)]
    d = d.groupby(["date", "j", "i"], as_index=False)["h_mean"].mean()
    d["season"] = season(pd.DatetimeIndex(d.date))
    for fold, seasons in FOLDS.items():
        s = d[d.season.isin(seasons)]
        days = pd.DatetimeIndex(sorted(s.date.unique()))
        T = np.full((len(days), LAT.size, LON.size), np.nan, np.float32)
        T[days.get_indexer(s.date), s.j.values, s.i.values] = s.h_mean.values
        ds = xr.Dataset({"target": (("time", "lat", "lon"), T, {"units": "m",
                                                                "long_name": "ICESat-2 pass-cell mean sea ice thickness (QC'd)"})},
                        coords={"time": days, "lat": LAT, "lon": LON},
                        attrs={"description": __doc__.strip(), "calibration_seasons": ", ".join(f"{y}/{(y + 1) % 100:02d}" for y in seasons)})
        out = f"{D}/T_IS2cells_mean_fold{fold}.nc"
        ds.to_netcdf(out, encoding={"target": {"zlib": True}})
        print(f"fold {fold}: seasons {seasons}, {len(days)} days, {int(np.isfinite(T).sum())} observed cells -> {out}")


if __name__ == "__main__":
    main()
