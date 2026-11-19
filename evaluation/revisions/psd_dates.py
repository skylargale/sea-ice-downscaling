#!/usr/bin/env python3
"""
Isotropic power spectra on single days (2026-11-19): 1 Jan, 1 Apr and 11 Nov 2021, MESACLIP perfect model, for the
supplementary figure domain_mean_psd_dates.png (the per-date counterpart of domain_mean_psd.png, which averages the
spectra over the 2021 test year).
Same spectrum as run_daily_eval_batch.py section 13 (isotropic_psd: ocean-mean removed, land set to 0, 2-D Hann window,
|FFT|^2 / (H W), 40 radial wavenumber bins up to the Nyquist wavenumber) and the same fields as figure data psd.csv:
truth, EngressNet ensemble mean, the deterministic U-Net (det-MSE run), bilinear, PIOMAS (nearest month, regridded by
rdeb.build_piomas_regridded). PIOMAS NaNs on ocean cells, if any, are set to the ocean mean (zero anomaly) and counted.
Writes figures/figures_v2/data_<FINAL_SET>/psd_dates.csv (long format: date, wavenumber, one column per field).
Run from the project root:  python evaluation/revisions/psd_dates.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "evaluation"))
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
import run_daily_eval_batch as rdeb  # noqa: E402
from final_set import FS, NAME as FINAL_SET  # noqa: E402

DATES = ["2021-01-01", "2021-04-01", "2021-11-11"]
N_PSD_BINS = 40


def run_dir(batch):
    m = sorted(d for d in glob.glob(f"{V6}/results/{batch}/*{FS['map_window']}*") if os.path.isdir(d))
    assert len(m) == 1, (batch, m)
    return m[0]


def isotropic_psd_one(field, ocean, dlat, dlon, mean_lat, n_bins=N_PSD_BINS):
    """run_daily_eval_batch.isotropic_psd for one (H, W) field."""
    H, W = field.shape
    dy = 111.0 * dlat
    dx = 111.0 * dlon * np.cos(np.deg2rad(mean_lat))
    window = np.outer(np.hanning(H), np.hanning(W))
    ky, kx = np.fft.fftfreq(H, d=dy), np.fft.fftfreq(W, d=dx)
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    edges = np.linspace(0, min(ky.max(), kx.max()), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, n_bins - 1)
    f = field.astype(np.float64).copy()
    f[ocean] -= f[ocean].mean()
    f[~ocean] = 0.0
    spec = (np.abs(np.fft.fft2(f * window)) ** 2 / (H * W)).ravel()
    psd = np.array([spec[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)])
    return centers, psd


def main():
    ed = os.path.join(run_dir(FS["MESA"]["eng"]), "eval_data")
    f = np.load(os.path.join(ed, "fields.npz"))
    times = pd.to_datetime(pd.read_csv(os.path.join(ed, "sample_times.csv"))["time"])
    sel = [int(np.flatnonzero(times.dt.normalize() == pd.Timestamp(d))[0]) for d in DATES]
    assert len(set(sel)) == len(DATES), sel
    dfields = np.load(os.path.join(run_dir(FS["MESA"]["det"]), "eval_data", "fields.npz"))
    assert np.allclose(dfields["Y_test_phys"][sel], f["Y_test_phys"][sel]), "det-MSE run has a different truth"
    import pickle
    geo = pickle.load(open(os.path.join(ed, "tile_geometry.pkl"), "rb"))[0]
    lat, lon = np.asarray(geo["target_lat"]), np.asarray(geo["target_lon"])
    ocean = f["mask_test"][0, 0] <= 0.5
    dlat, dlon, mean_lat = float(np.diff(lat).mean()), float(np.diff(lon).mean()), float(lat.mean())
    piomas = rdeb.build_piomas_regridded(lat, lon, times.iloc[sel])[:, 0]
    rows = []
    for j, (d, i) in enumerate(zip(DATES, sel)):
        fields = {"Truth": f["Y_test_phys"][i, 0], "Stochastic UNet Mean": f["Y_pred_phys"][i, 0],
                  "Deterministic UNet": dfields["Y_pred_det_phys"][i, 0], "Bilinear": f["Y_base_phys"][i, 0],
                  "PIOMAS (obs)": piomas[j]}
        n_nan = int(np.isnan(fields["PIOMAS (obs)"][ocean]).sum())
        if n_nan:
            p = fields["PIOMAS (obs)"].copy(); p[ocean & np.isnan(p)] = np.nanmean(p[ocean]); fields["PIOMAS (obs)"] = p
        out = {}
        for name, fld in fields.items():
            k, out[name] = isotropic_psd_one(fld, ocean, dlat, dlon, mean_lat)
        for b in range(N_PSD_BINS):
            rows.append({"date": d, "wavenumber_cycles_per_km": k[b], **{n: out[n][b] for n in fields}})
        print(d, "sample", i, times.iloc[i].date(), "| PIOMAS NaN ocean cells filled:", n_nan, flush=True)
    path = f"{V6}/figures/figures_v2/data_{FINAL_SET}/psd_dates.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
