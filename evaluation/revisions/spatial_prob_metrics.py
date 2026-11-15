"""
Revisions 2026-11, section D (+ the scoring for E): spatial and multivariate probabilistic metrics on the 2021 test year,
ocean cells, for one dataset and training window. Reads only saved eval_data/fields.npz (no GPU).

Methods (each scored as an ensemble; single-valued methods as a 1-member ensemble / point mass):
  EngressNet           results/<DS>_final_ema, preds_all_phys (K = 20)
  Deterministic U-Net  results/<DS>_detmse_ema, Y_pred_det_phys (point mass)
  Bilinear             Y_base_phys, land zeroed (point mass)
  --extra NAME=BATCH[@substr]  any batch's preds_all_phys for this window (e.g. MC-dropout, H noise modes);
                       @substr picks one run when the batch has several for the window (e.g. @_s1_)
  --seedwin B1,B2,...  "Seed/window ensemble (det.)": Y_pred_det_phys of every window of the listed det-MSE batches pooled
                       into one ensemble (3 seeds x 4 windows = 12 members); identical test set in every window, so it is the
                       same ensemble whichever --window is being scored

Metrics:
  ES whole field   fair energy score of the ocean-cell vector, beta = 1 (reported) and 0.8 (training value):
                   mean_k ||x_k - y||^b - 1/(2K(K-1)) sum_{k != l} ||x_k - x_l||^b  (point mass: ||x - y||^b). Also / sqrt(n_ocean).
  ES tiled 10x10   same score on the 10 x 10 px tiles the training loss uses (15 x 31 tiles), land zeroed, mean over tiles with
                   >= 1 ocean cell, then over samples.
  Variogram score  Scheuerer & Hamill (2015), p = 0.5, weights 1/d (km, great circle), over all pairs of a fixed random subset
                   of VS_CELLS ocean cells (seed 0, identical for every method and window); normalised by the weight sum.
  FSS              thresholds 0.15 / 0.5 / 1.0 m, neighbourhoods 1 / 5 / 11 / 21 px; forecast = ensemble exceedance fraction
                   (point mass: binary); fractions over ocean cells only (land excluded from the neighbourhood mean);
                   FSS = 1 - sum (Pf - Of)^2 / (sum Pf^2 + sum Of^2), sums over ocean cells and all samples.
  ACC              anomaly correlation (uncentred, over ocean cells, per sample then averaged) of the ensemble mean / point
                   value, anomalies w.r.t. a per-cell day-of-year truth climatology from this window's TRAINING years only
                   (all members, SIT clipped at 6 m as in training), smoothed with a circular 31-day running mean.
  member-level     SSIM, pattern r, gradient MAE, IIEE (1e-4 m threshold, cos-lat), RMSE: each member scored with the paper's
                   ocean-cell definitions (evaluation/recompute_ocean_metrics.py), then averaged over members; plus the same
                   for the ensemble mean.
  CRPS / spread    pixel CRPS (K^2 estimator, as the paper), corrected spread-skill sqrt((K+1)/K mean s^2)/RMSE, rank-histogram
                   fraction outside the ensemble (ranks 0 and K; detection limit as rdeb.detection_limited; ideal 2/(K+1)).
  PSD              radially averaged power spectra (same Hann-window formula as run_daily_eval_batch section 13) of truth,
                   ensemble mean, every member (mean and 10-90 % across members) -> <out>_psd.csv.

Usage:
  python spatial_prob_metrics.py --dataset MESA --window 2015-2020 [--extra "MC-dropout U-Net=REV_E2_MESA_mcdropout"]
         [--seedwin MESA_detmse_ema,REV_E1_MESA_detmse_seed1,REV_E1_MESA_detmse_seed2] [--out-dir results/revisions_2026-11/D_metrics]
"""
import argparse
import glob
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import uniform_filter

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "training"))
sys.path.insert(0, os.path.join(V6, "evaluation"))
import functions_engressnet as fe  # noqa: E402
import run_daily_eval_batch as rdeb  # noqa: E402
sys.path.insert(0, os.path.join(V6, "evaluation", "revisions"))
from final_set import FS, NAME as FINAL_SET  # noqa: E402
SFX = "" if FINAL_SET == "final1" else f"_{FINAL_SET}"

Y_FILES = {"MESA": "/glade/derecho/scratch/skygale/downscaling/training_data/Y_MESA_HR_daily.nc",
           "FOSI": "/glade/derecho/scratch/skygale/downscaling/training_data/Y_FOSI_HR_JRA55_daily.nc"}
FSS_THR, FSS_N = [0.15, 0.5, 1.0], [1, 5, 11, 21]
VS_CELLS, VS_P = 1000, 0.5
TILE = 10
CHUNK = 32
torch.set_num_threads(int(os.environ.get("NCPUS", "4")))


def run_dir(batch, window):
    """batch may be "BATCH@substr" to pick one run of a batch that holds several runs for the same window (e.g. seeds)."""
    batch, _, sub = batch.partition("@")
    m = [d for d in glob.glob(os.path.join(V6, "results", batch, f"*_{window}_2021_*")) if os.path.isdir(d) and sub in os.path.basename(d)]
    return m[0] if len(m) == 1 else None


def noleap_doy(times):
    cum = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    return np.array([cum[t.month - 1] + min(t.day, 28 if t.month == 2 else 31) - 1 for t in times])


# ---------------------------------------------------------------- scores on (B, K, n) tensors
def energy(X, y, beta):
    """X (B, K, n), y (B, n) -> (B,) fair energy score."""
    K = X.shape[1]
    t1 = torch.linalg.vector_norm(X - y[:, None], dim=-1).pow(beta).mean(1)
    if K == 1:
        return t1
    D = torch.cdist(X, X)
    t2 = D.pow(beta).sum((1, 2)) / (K * (K - 1))
    return t1 - 0.5 * t2


def crps_ens(X, y):
    """X (M, K), y (M,) -> (M,) pixel CRPS, K^2 estimator (as recompute_spread_skill.py)."""
    K = X.shape[1]
    w = 2 * np.arange(1, K + 1) - K - 1
    return np.abs(X - y[:, None]).mean(1) - (np.sort(X, 1) * w).sum(1) / K ** 2


# ---------------------------------------------------------------- per-field structure metrics (recompute_ocean_metrics)
def structure_metrics(pred, truth, ocean, hlat):
    """pred, truth: torch (N, 1, H, W); ocean (H, W) bool. Paper ocean-cell definitions."""
    oc = ocean.expand(pred.shape[0], -1, -1)
    err = (pred - truth)[:, 0]
    e_o = err[oc].double()
    r = []
    for i in range(pred.shape[0]):
        a, b = pred[i, 0][ocean].double(), truth[i, 0][ocean].double()
        if a.std() > 0 and b.std() > 0:
            r.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    gx_ok = (ocean[:, 1:] & ocean[:, :-1]).expand(pred.shape[0], -1, -1)
    gy_ok = (ocean[1:, :] & ocean[:-1, :]).expand(pred.shape[0], -1, -1)
    p0, t0 = pred[:, 0], truth[:, 0]
    gx = ((p0[:, :, 1:] - p0[:, :, :-1]) - (t0[:, :, 1:] - t0[:, :, :-1]))[gx_ok].abs().double()
    gy = ((p0[:, 1:, :] - p0[:, :-1, :]) - (t0[:, 1:, :] - t0[:, :-1, :]))[gy_ok].abs().double()
    ss, ws = [], []
    for i in range(0, pred.shape[0], 128):
        v = fe.ssim(pred[i:i + 128], truth[i:i + 128])
        ss.append(v.item() if torch.is_tensor(v) else float(v)); ws.append(min(128, pred.shape[0] - i))
    return {"RMSE": e_o.pow(2).mean().sqrt().item(), "Pattern Corr": float(np.mean(r)) if r else np.nan,
            "Grad MAE": torch.cat([gx, gy]).mean().item(), "SSIM": float(np.average(ss, weights=ws)),
            "IIEE": float(fe.ice_edge_error(pred, truth, threshold=1e-4, mask_bool=oc, lat=hlat))}


def isotropic_psd(fields, ocean, dlat, dlon, mean_lat, n_bins=40):
    """fields (N, H, W) numpy -> (bin centres [cycles/km], mean PSD over samples)."""
    N, H, W = fields.shape
    dy, dx = 111.0 * dlat, 111.0 * dlon * np.cos(np.deg2rad(mean_lat))
    win = np.outer(np.hanning(H), np.hanning(W))
    ky, kx = np.fft.fftfreq(H, d=dy), np.fft.fftfreq(W, d=dx)
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    edges = np.linspace(0, min(ky.max(), kx.max()), n_bins + 1)
    idx = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    out = np.zeros(n_bins)
    for i in range(N):
        f = fields[i].astype(np.float64).copy()
        f[ocean] -= f[ocean].mean()
        f[~ocean] = 0.0
        spec = (np.abs(np.fft.fft2(f * win)) ** 2 / (H * W)).ravel()
        out += np.bincount(idx, weights=spec, minlength=n_bins) / np.maximum(cnt, 1)
    return 0.5 * (edges[:-1] + edges[1:]), out / N


# ---------------------------------------------------------------- main scoring of one ensemble
def score(name, ens, truth, ocean, ctx):
    """ens: numpy (N, K, H, W) float32 (land already 0); truth (N, H, W). Returns dict of metric -> value."""
    t0 = time.time()
    N, K = ens.shape[:2]
    H, W = truth.shape[1:]
    oc_idx = np.flatnonzero(ocean.ravel())
    res = {"K": K}
    # --- energy scores
    es = {1.0: [], 0.8: []}
    est = {1.0: [], 0.8: []}
    nh, nw = round(H / TILE), round(W / TILE)
    assert nh * TILE == H and nw * TILE == W, "tiling assumes 10 px divides the field (150 x 310)"
    tile_oc = ocean.reshape(nh, TILE, nw, TILE).any((1, 3)).ravel()
    for i in range(0, N, CHUNK):
        X = torch.from_numpy(ens[i:i + CHUNK].reshape(-1, K, H * W)[:, :, oc_idx]).double()
        y = torch.from_numpy(truth[i:i + CHUNK].reshape(-1, H * W)[:, oc_idx]).double()
        Xt = torch.from_numpy(ens[i:i + CHUNK] * ocean).double()
        B = Xt.shape[0]
        Xt = Xt.reshape(B, K, nh, TILE, nw, TILE).permute(0, 2, 4, 1, 3, 5).reshape(B, nh * nw, K, TILE * TILE)
        yt = torch.from_numpy(truth[i:i + CHUNK] * ocean).double().reshape(B, nh, TILE, nw, TILE).permute(0, 1, 3, 2, 4).reshape(B, nh * nw, TILE * TILE)
        Xt, yt = Xt[:, tile_oc], yt[:, tile_oc]
        for b in (1.0, 0.8):
            es[b].append(energy(X, y, b).numpy())
            est[b].append(energy(Xt.reshape(-1, K, TILE * TILE), yt.reshape(-1, TILE * TILE), b).reshape(B, -1).mean(1).numpy())
    n_oc = len(oc_idx)
    for b in (1.0, 0.8):
        e = np.concatenate(es[b]).mean()
        res[f"ES whole field (beta={b:g})"] = e
        res[f"ES whole field / sqrt(n_ocean) (beta={b:g})"] = e / math.sqrt(n_oc) if b == 1.0 else e / n_oc ** (b / 2)
        res[f"ES tiled 10x10 (beta={b:g})"] = np.concatenate(est[b]).mean()
    # --- variogram score
    vi, (pi, pj), w = ctx["vs_idx"], ctx["vs_pairs"], ctx["vs_w"]
    vs = []
    for i in range(0, N, 8):
        Xs = torch.from_numpy(ens[i:i + 8].reshape(-1, K, H * W)[:, :, vi]).float()
        ys = torch.from_numpy(truth[i:i + 8].reshape(-1, H * W)[:, vi]).float()
        xd = (Xs[:, :, pi] - Xs[:, :, pj]).abs().pow(VS_P).mean(1)
        yd = (ys[:, pi] - ys[:, pj]).abs().pow(VS_P)
        vs.append((((yd - xd) ** 2) * w).sum(1).numpy() / w.sum().item())
    res[f"Variogram score (p={VS_P:g}, {len(vi)} cells)"] = np.concatenate(vs).mean()
    # --- FSS
    ocf = ocean.astype(np.float64)
    for thr in FSS_THR:
        P = (ens > thr).mean(1).astype(np.float64) * ocf
        O = (truth > thr).astype(np.float64) * ocf
        for n in FSS_N:
            den_oc = uniform_filter(ocf, size=n, mode="constant")
            Pf = uniform_filter(P, size=(1, n, n), mode="constant") / np.maximum(den_oc, 1e-12)
            Of = uniform_filter(O, size=(1, n, n), mode="constant") / np.maximum(den_oc, 1e-12)
            num = ((Pf - Of) ** 2)[:, ocean].sum()
            den = (Pf ** 2)[:, ocean].sum() + (Of ** 2)[:, ocean].sum()
            res[f"FSS thr={thr:g} m n={n}"] = 1 - num / den if den > 0 else np.nan
    # --- ensemble mean / point value
    mu = ens.mean(1)
    # ACC vs training-years climatology
    clim = ctx["clim"][ctx["doy"]]  # (N, H, W)
    fa, oa = (mu - clim)[:, ocean], (truth - clim)[:, ocean]
    den = np.sqrt((fa ** 2).sum(1) * (oa ** 2).sum(1))
    ok = den > 0
    res["ACC (training-years DOY climatology)"] = float(((fa * oa).sum(1)[ok] / den[ok]).mean())
    # --- CRPS, spread-skill, rank histogram (ocean)
    Eo = np.moveaxis(ens, 1, -1)[:, ocean].reshape(-1, K).astype(np.float64)
    yo = truth[:, ocean].ravel().astype(np.float64)
    res["CRPS (point mass: MAE)"] = crps_ens(Eo, yo).mean() if K > 1 else np.abs(Eo[:, 0] - yo).mean()
    rmse = np.sqrt(((Eo.mean(1) - yo) ** 2).mean())
    res["Ensemble-mean RMSE"] = rmse
    if K > 1:
        res["Spread-skill (corrected)"] = np.sqrt((K + 1) / K * Eo.var(1, ddof=1).mean()) / rmse
        er, tr = rdeb.detection_limited(Eo, yo)
        below, ties = (er < tr[:, None]).sum(1), (er == tr[:, None]).sum(1)
        ranks = below + np.random.default_rng(0).integers(0, ties + 1)
        res["Rank: fraction outside ensemble"] = float(((ranks == 0) | (ranks == K)).mean())
        res["Rank: ideal outside fraction"] = 2 / (K + 1)
    del Eo, yo
    # --- structure: ensemble mean and members
    oct_ = torch.from_numpy(ocean)
    tt = torch.from_numpy(truth[:, None])
    for k, v in structure_metrics(torch.from_numpy(mu[:, None]), tt, oct_, ctx["hlat"]).items():
        res[f"{k} (ensemble mean / point value)"] = v
    if K > 1:
        mem = [structure_metrics(torch.from_numpy(np.ascontiguousarray(ens[:, k, None])), tt, oct_, ctx["hlat"]) for k in range(K)]
        for k in mem[0]:
            res[f"{k} (member mean)"] = float(np.mean([m[k] for m in mem]))
    # --- PSD
    kc, p_mu = isotropic_psd(mu, ocean, *ctx["psd_grid"])
    ctx["psd"][f"{name} | mean"] = p_mu
    if K > 1:
        pm = np.stack([isotropic_psd(ens[:, k], ocean, *ctx["psd_grid"])[1] for k in range(K)])
        ctx["psd"][f"{name} | members mean"] = pm.mean(0)
        ctx["psd"][f"{name} | members p10"] = np.percentile(pm, 10, 0)
        ctx["psd"][f"{name} | members p90"] = np.percentile(pm, 90, 0)
    ctx["psd_k"] = kc
    print(f"  scored {name} (K={K}) in {time.time() - t0:.0f} s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["MESA", "FOSI"], required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--extra", action="append", default=[], help="NAME=BATCH (preds_all_phys of that batch, same window)")
    ap.add_argument("--seedwin", default="", help="comma-separated det-MSE batches pooled over all windows")
    ap.add_argument("--skip-core", action="store_true", help="score only --extra / --seedwin methods")
    ap.add_argument("--out-dir", default=os.path.join(V6, "results", "revisions_2026-11", f"D_metrics{SFX}"))
    ap.add_argument("--max-samples", type=int, default=0, help="smoke test: first N samples only")
    ap.add_argument("--tag", default="", help="suffix for the output file names (several jobs on one window)")
    a = ap.parse_args()
    ds, win = a.dataset, a.window
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{ds}_{win}" + ("_extra" if a.skip_core else "") + (f"_n{a.max_samples}" if a.max_samples else "")
    tag += f"_{a.tag}" if a.tag else ""
    eng_dir = run_dir(FS[ds]["eng"], win)  # FINAL_SET (final_set.py)
    det_dir = run_dir(FS[ds]["det"], win)
    f = np.load(os.path.join(eng_dir, "eval_data", "fields.npz"))
    sl = slice(0, a.max_samples) if a.max_samples else slice(None)
    truth = f["Y_test_phys"][sl, 0].astype(np.float32)
    mask = f["mask_test"][sl, 0]
    assert (mask == mask[0]).all(), "land mask varies across samples"
    ocean = mask[0] <= 0.5
    hlat = torch.from_numpy(f["hlat"][:truth.shape[1]] if f["hlat"].shape[0] != truth.shape[1] else f["hlat"])
    geo = pickle.load(open(os.path.join(eng_dir, "eval_data", "tile_geometry.pkl"), "rb"))[0]
    tlat, tlon = np.asarray(geo["target_lat"]), np.asarray(geo["target_lon"])
    times = pd.to_datetime(pd.read_csv(os.path.join(eng_dir, "eval_data", "sample_times.csv"))["time"])[sl]
    N = truth.shape[0]
    print(f"{ds} {win}: N={N}, ocean cells {ocean.sum()}", flush=True)

    # ---- context shared by all methods
    rng = np.random.default_rng(0)
    oc_flat = np.flatnonzero(ocean.ravel())
    vs_idx = np.sort(rng.choice(oc_flat, size=min(VS_CELLS, oc_flat.size), replace=False))
    iy, ix = np.unravel_index(vs_idx, ocean.shape)
    la, lo = np.deg2rad(tlat[iy]), np.deg2rad(tlon[ix])
    pi, pj = np.triu_indices(len(vs_idx), 1)
    dlat, dlon = la[pj] - la[pi], lo[pj] - lo[pi]
    hav = np.sin(dlat / 2) ** 2 + np.cos(la[pi]) * np.cos(la[pj]) * np.sin(dlon / 2) ** 2
    d_km = 2 * 6371.0 * np.arcsin(np.sqrt(hav))
    # training-years day-of-year climatology of the truth (all members), smoothed with a circular 31-day mean
    yrs = fe.parse_years(win)
    Yds = xr.open_dataset(Y_FILES[ds])
    Yv = Yds[list(Yds.data_vars)[0]]
    ylat, ylon = Yds["lat"].values, Yds["lon"].values % 360
    li = [int(np.argmin(np.abs(ylat - v))) for v in tlat]
    lj = [int(np.argmin(np.abs(ylon - (v % 360)))) for v in tlon]
    assert np.allclose(ylat[li], tlat, atol=1e-3) and np.allclose(ylon[lj], tlon % 360, atol=1e-3), "Y grid mismatch"
    tyears = np.array([t.year for t in Yds["time"].values])
    sel_t = np.flatnonzero(np.isin(tyears, yrs))
    clim_sum = np.zeros((365,) + ocean.shape); clim_n = np.zeros(365)
    doy_tr = noleap_doy(Yds["time"].values[sel_t])
    for m in range(Yv.sizes["ensemble"]):
        arr = np.clip(Yv.isel(ensemble=m, time=sel_t, channel=0).values[:, li][:, :, lj], None, 6.0)
        np.add.at(clim_sum, doy_tr, arr); np.add.at(clim_n, doy_tr, 1)
    clim = clim_sum / np.maximum(clim_n, 1)[:, None, None]
    clim = np.mean([np.roll(clim, s, axis=0) for s in range(-15, 16)], axis=0)  # circular 31-day running mean
    ctx = {"vs_idx": vs_idx, "vs_pairs": (torch.from_numpy(pi), torch.from_numpy(pj)), "vs_w": torch.from_numpy(1.0 / d_km).float(),
           "clim": clim.astype(np.float32), "doy": noleap_doy(times), "hlat": hlat, "psd": {},
           "psd_grid": (float(np.diff(tlat).mean()), float(np.diff(tlon).mean()), float(tlat.mean()))}
    print(f"  climatology from {len(sel_t)} training days x {Yv.sizes['ensemble']} members ({min(yrs)}-{max(yrs)})", flush=True)

    rows = []
    if not a.skip_core:
        ens = (f["preds_all_phys"][sl, :, 0]).astype(np.float32)
        rows.append({"method": "EngressNet", **score("EngressNet", ens, truth, ocean, ctx)})
        del ens
        rows.append({"method": "Bilinear", **score("Bilinear", (f["Y_base_phys"][sl, 0] * ocean)[:, None].astype(np.float32), truth, ocean, ctx)})
        fd = np.load(os.path.join(det_dir, "eval_data", "fields.npz"))
        assert np.allclose(fd["Y_test_phys"][sl, 0], truth, atol=1e-5), "det-MSE test set differs"
        rows.append({"method": "Deterministic U-Net", **score("Deterministic U-Net", fd["Y_pred_det_phys"][sl, 0][:, None].astype(np.float32), truth, ocean, ctx)})
        ctx["psd"]["Truth"] = isotropic_psd(truth, ocean, *ctx["psd_grid"])[1]
    for spec in a.extra:
        name, batch = spec.split("=", 1)
        d = run_dir(batch, win)
        if d is None:
            print(f"  skip {name}: no {batch} run for {win}"); continue
        fx = np.load(os.path.join(d, "eval_data", "fields.npz"))
        assert np.allclose(fx["Y_test_phys"][sl, 0], truth, atol=1e-5), f"{batch} test set differs"
        rows.append({"method": name, "batch": batch, **score(name, fx["preds_all_phys"][sl, :, 0].astype(np.float32), truth, ocean, ctx)})
    if a.seedwin:
        mem = []
        for b in a.seedwin.split(","):
            for d in sorted(glob.glob(os.path.join(V6, "results", b, "*_2021_*", "eval_data", "fields.npz"))):
                fx = np.load(d)
                assert np.allclose(fx["Y_test_phys"][sl, 0], truth, atol=1e-5), f"{d} test set differs"
                mem.append(fx["Y_pred_det_phys"][sl, 0].astype(np.float32))
                print(f"  seed/window member {len(mem)}: {os.path.relpath(d, V6)}", flush=True)
        if mem:
            rows.append({"method": "Seed/window ensemble (det.)", "batch": a.seedwin,
                         **score("Seed/window ensemble (det.)", np.stack(mem, 1), truth, ocean, ctx)})
    df = pd.DataFrame(rows).set_index("method").T
    df.to_csv(os.path.join(a.out_dir, f"{tag}.csv"))
    if ctx["psd"]:
        p = pd.DataFrame(ctx["psd"]); p.insert(0, "wavenumber_cycles_per_km", ctx["psd_k"])
        p.to_csv(os.path.join(a.out_dir, f"{tag}_psd.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(df.to_string())
    print("wrote", os.path.join(a.out_dir, f"{tag}.csv"))


if __name__ == "__main__":
    main()
