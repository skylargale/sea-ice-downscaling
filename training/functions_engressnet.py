"""
functions_engressnet.py

Shared functions for EngressNet SIT downscaling: data loading, land-sea
masking, patch extraction (or single sub-domain cropping when patches are
disabled), the stochastic UNet model, training/evaluation loops, metrics,
figure generation, and the domain SIT time series output.

Imported by train_engressnet.py and hpo_engressnet.py -- run_pipeline() is
the single entry point both scripts call, so the full pipeline only lives
in one place.
"""

import datetime
import json
import os
import pickle
import re
import warnings
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as torch_checkpoint
import xarray as xr
import xesmf as xe
import pop_tools
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt

from member_metrics import compute_member_avg_metrics, is_mesaclip_run

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=r"Latitude is outside of \[-90, 90\]")

# ==============================================================
# Defaults (all overridable via the config object passed to run_pipeline)
# ==============================================================

DEFAULT_DATA_DIR = "/glade/derecho/scratch/skygale/Downscaling_Data"
DEFAULT_X_PATH = os.path.join(DEFAULT_DATA_DIR, "X_FOSI_HR_JRA55_interp.nc")
DEFAULT_Y_PATH = os.path.join(DEFAULT_DATA_DIR, "Y_FOSI_HR_JRA55.nc")
DEFAULT_WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"
DEFAULT_RESULTS_DIR = "/glade/work/skygale/projects/SeaIceDownscaling/Version6/results"

DEFAULT_BBOX = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
DEFAULT_BBOX_REGRID = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
KIVALINA_LAT = 67.7269
KIVALINA_LON_360 = -164.5333 % 360

# Coastal-community "candidate points" for point time series (in addition
# to the domain-mean time series). Lon stored in [0, 360) to match
# llon/hlon as loaded from the .nc files.
CANDIDATE_POINTS = {
    "Kivalina": {"lat": KIVALINA_LAT, "lon": KIVALINA_LON_360},
    "Shishmaref": {"lat": 66.2567, "lon": -166.0719 % 360},
    "Kotzebue": {"lat": 66.8983, "lon": -162.5967 % 360},
    "Nome": {"lat": 64.5011, "lon": -165.4064 % 360},
    "Point Hope": {"lat": 68.3415, "lon": -166.7578 % 360},
}

DEFAULT_CONTEXT_SIZE = (16, 24)
DEFAULT_TARGET_SIZE = (8, 12)
DEFAULT_STRIDE = 4


def save_run_config(config):
    """
    Write the full run config to <output_dir>/run_config.json (machine-
    readable, everything needed to reproduce the run) and a short
    <output_dir>/description.txt (human-readable one-glance summary),
    so a results/ folder is self-describing regardless of what its
    directory name happens to be. Called at the very start of
    run_pipeline(), before any data loading, so it's written even if the
    run later crashes.
    """
    cfg_dict = dict(vars(config))
    cfg_dict["pbs_jobid"] = os.environ.get("PBS_JOBID")
    cfg_dict["pbs_jobname"] = os.environ.get("PBS_JOBNAME")
    cfg_dict["written_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(os.path.join(config.output_dir, "run_config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    if config.use_patches:
        domain_desc = f"patches (context {tuple(config.context_size)}, target {tuple(config.target_size)}, stride {config.stride})"
    else:
        sd = config.subdomain
        domain_desc = f"no-patches sub-domain (lat {sd['lat_min']} to {sd['lat_max']}, lon {sd['lon_min']} to {sd['lon_max']})"

    train_desc = ",".join(str(y) for y in config.train_years) if config.train_years else f"random {config.train_frac:.0%} split"
    test_desc = ",".join(str(y) for y in config.test_years) if config.test_years else f"random {1 - config.train_frac:.0%} split"
    months_desc = ",".join(str(m) for m in config.months) if getattr(config, "months", None) else "all months"

    lines = [
        f"X data: {config.x_path}",
        f"Y data: {config.y_path}",
        f"Train years: {train_desc}",
        f"Test years: {test_desc}",
        f"Months: {months_desc}",
        f"Domain: {domain_desc}",
        f"k (train) / k_eval: {config.k} / {config.k_eval}",
        f"num_epochs: {config.num_epochs}   batch_size: {config.batch_size}   lr: {config.lr}",
        f"PBS job: {cfg_dict['pbs_jobname']} ({cfg_dict['pbs_jobid']})",
        f"Written: {cfg_dict['written_at']}",
    ]
    with open(os.path.join(config.output_dir, "description.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def save_fig(fig, output_dir, name, dpi=200):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


# ==============================================================
# Year parsing / time selection
# ==============================================================

def parse_years(spec):
    """
    Parse a year specification string into a sorted list of ints.

    Accepts comma-separated years and/or ranges, e.g.:
        "1980-2000"                 -> 1980, 1981, ..., 2000
        "1980,1985,1990"            -> 1980, 1985, 1990
        "1980-1985,1995,2000-2002"  -> mix of both

    Returns None if spec is None (caller should fall back to a random split).
    """
    if spec is None:
        return None

    years = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", chunk)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            years.update(range(lo, hi + 1))
        elif re.fullmatch(r"\d{4}", chunk):
            years.add(int(chunk))
        else:
            raise ValueError(
                f"Could not parse year chunk {chunk!r} in spec {spec!r}. "
                "Use a 4-digit year (e.g. 1990) or a range (e.g. 1980-2000)."
            )
    return sorted(years)


def years_mask(time_coord, years):
    """
    Boolean mask, aligned with time_coord, True where the year of
    time_coord is in `years`. time_coord should be an xarray DataArray
    (e.g. X.time) so .dt.year works for both datetime64 and cftime.
    """
    time_years = time_coord.dt.year.values
    years_set = set(years)
    return np.array([int(y) in years_set for y in time_years])


def parse_months(spec):
    """
    Parse a month specification string into a sorted list of ints (1-12).

    Accepts comma-separated months and/or ranges, e.g.:
        "3-7"     -> 3, 4, 5, 6, 7   (March-July)
        "3,4,5"   -> 3, 4, 5
        "1,6-8"   -> mix of both

    Unlike parse_years, does NOT support wraparound ranges (e.g. "11-2" for
    Nov-Feb) -- a low > high chunk is swapped, same as parse_years, which
    would silently produce the wrong months for a wraparound season. Not
    needed for any season used in this project so far; revisit if one comes
    up.

    Returns None if spec is None (caller should fall back to no month filter).
    """
    if spec is None:
        return None

    months = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.fullmatch(r"(\d{1,2})\s*-\s*(\d{1,2})", chunk)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            months.update(range(lo, hi + 1))
        elif re.fullmatch(r"\d{1,2}", chunk):
            months.add(int(chunk))
        else:
            raise ValueError(
                f"Could not parse month chunk {chunk!r} in spec {spec!r}. "
                "Use a 1-2 digit month (e.g. 3) or a range (e.g. 3-7)."
            )
    if not all(1 <= m <= 12 for m in months):
        raise ValueError(f"Month spec {spec!r} contains a value outside 1-12.")
    return sorted(months)


def months_mask(time_coord, months):
    """
    Boolean mask, aligned with time_coord, True where the calendar month of
    time_coord is in `months`. time_coord should be an xarray DataArray
    (e.g. X.time) so .dt.month works for both datetime64 and cftime.
    """
    time_months = time_coord.dt.month.values
    months_set = set(months)
    return np.array([int(m) in months_set for m in time_months])


def split_train_test(X, Y, time_coord, train_years, test_years, train_frac, seed=0, months=None):
    """
    Split X, Y (numpy arrays, shape (N, T, C, H, W) / (N, T, Cy, Hh, Wh))
    into train/test sets, either by explicit year lists (train_years /
    test_years, disjoint, both required together) or -- if neither is
    given -- a random train_frac / (1 - train_frac) split like the
    original pipeline.

    If `months` is given (a list of ints 1-12, e.g. from parse_months), the
    data is first restricted to timestamps in those calendar months, before
    either the year-based or random split -- so a seasonal focus (e.g.
    March-July only) applies uniformly to both train and test samples,
    whichever split mode is used.

    Returns X_train, Y_train, X_test, Y_test as flattened (Nsamples, C, H, W)
    float tensors, plus time_train, time_test (one timestamp per sample)
    and member_train, member_test (the N/ensemble-member index per sample)
    for downstream bookkeeping (e.g. the SIT time series).
    """
    if months is not None:
        month_mask_t = months_mask(time_coord, months)
        if not month_mask_t.any():
            raise ValueError(f"No timestamps in the data match the requested months {months}.")
        X = X[:, month_mask_t]
        Y = Y[:, month_mask_t]
        time_coord = time_coord[month_mask_t]

    N, T, C, H, W = X.shape
    _, _, Cy, Hy, Wy = Y.shape
    time_vals = time_coord.values

    if train_years is not None or test_years is not None:
        if train_years is None or test_years is None:
            raise ValueError("Provide both train_years and test_years, or neither (for a random split).")

        train_mask_t = years_mask(time_coord, train_years)
        test_mask_t = years_mask(time_coord, test_years)

        if not train_mask_t.any():
            raise ValueError("No timestamps in the data match the requested train_years.")
        if not test_mask_t.any():
            raise ValueError("No timestamps in the data match the requested test_years.")
        if np.any(train_mask_t & test_mask_t):
            n_overlap = int(np.sum(train_mask_t & test_mask_t))
            raise ValueError(f"train_years and test_years overlap at {n_overlap} timestamps; they must be disjoint.")

        X_train_raw = X[:, train_mask_t]
        Y_train_raw = Y[:, train_mask_t]
        X_test_raw = X[:, test_mask_t]
        Y_test_raw = Y[:, test_mask_t]

        time_train = np.tile(time_vals[train_mask_t], N)
        time_test = np.tile(time_vals[test_mask_t], N)
        member_train = np.repeat(np.arange(N), int(train_mask_t.sum()))
        member_test = np.repeat(np.arange(N), int(test_mask_t.sum()))

        X_train = torch.tensor(X_train_raw.reshape(-1, C, H, W)).float()
        Y_train = torch.tensor(Y_train_raw.reshape(-1, Cy, Hy, Wy)).float()
        X_test = torch.tensor(X_test_raw.reshape(-1, C, H, W)).float()
        Y_test = torch.tensor(Y_test_raw.reshape(-1, Cy, Hy, Wy)).float()

    else:
        X_fields = torch.tensor(X.reshape(N * T, C, H, W)).float()
        Y_fields = torch.tensor(Y.reshape(N * T, Cy, Hy, Wy)).float()
        time_flat = np.tile(time_vals, N)
        member_flat = np.repeat(np.arange(N), T)

        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(N * T, generator=g)
        split = int(train_frac * N * T)
        train_idx, test_idx = indices[:split], indices[split:]

        X_train, Y_train = X_fields[train_idx], Y_fields[train_idx]
        X_test, Y_test = X_fields[test_idx], Y_fields[test_idx]
        time_train = time_flat[train_idx.numpy()]
        time_test = time_flat[test_idx.numpy()]
        member_train = member_flat[train_idx.numpy()]
        member_test = member_flat[test_idx.numpy()]

    return X_train, Y_train, X_test, Y_test, time_train, time_test, member_train, member_test


def select_by_years(X, Y, time_coord, years, months=None):
    """
    One-sided counterpart to split_train_test: restrict X, Y (numpy arrays,
    shape (N, T, C, H, W) / (N, T, Cy, Hy, Wy)) to timestamps matching
    `years` (required, unlike split_train_test's random-split fallback --
    there's no sensible "random half of a different dataset" mode), after
    an optional calendar-month restriction. Used for cross-dataset
    evaluation (run_pipeline's `test_x_path`/`test_y_path`), where the
    train and test fields come from two different opened datasets rather
    than two slices of the same one, so split_train_test's single-array,
    disjoint-mask design doesn't apply.

    Returns X_fields, Y_fields (flattened (Nsamples, C, H, W) float
    tensors), plus time_flat and member_flat (one timestamp / N-index per
    sample), matching split_train_test's per-side return shape.
    """
    if months is not None:
        month_mask_t = months_mask(time_coord, months)
        if not month_mask_t.any():
            raise ValueError(f"No timestamps in the data match the requested months {months}.")
        X = X[:, month_mask_t]
        Y = Y[:, month_mask_t]
        time_coord = time_coord[month_mask_t]

    N, T, C, H, W = X.shape
    _, _, Cy, Hy, Wy = Y.shape
    time_vals = time_coord.values

    mask_t = years_mask(time_coord, years)
    if not mask_t.any():
        raise ValueError(f"No timestamps in the data match the requested years {years}.")

    X_raw = X[:, mask_t]
    Y_raw = Y[:, mask_t]
    time_flat = np.tile(time_vals[mask_t], N)
    member_flat = np.repeat(np.arange(N), int(mask_t.sum()))

    X_fields = torch.tensor(X_raw.reshape(-1, C, H, W)).float()
    Y_fields = torch.tensor(Y_raw.reshape(-1, Cy, Hy, Wy)).float()

    return X_fields, Y_fields, time_flat, member_flat


def collapse_wind_vector_channel(X, channel_names):
    """
    Cross-dataset channel harmonization: FOSI's X carries full vector wind
    ('u_10', 'v_10', true east/north m/s) while MESA-HR's X carries only
    wind *speed* ('U10') -- a real predictor-schema difference between the
    two datasets' construction pipelines (see processing/), not a bug.
    A checkpoint's first conv layer is sized for whatever channel count it
    trained on, so cross-dataset eval needs both sides to present the same
    channels. If `channel_names` (X's *original* loaded channel order, i.e.
    `X_da["channel"].values`) contains both 'u_10' and 'v_10', collapse them into
    a single derived wind-speed channel (sqrt(u^2+v^2)) in their place,
    dropping wind direction, so the result matches MESA's (hi_d, aice_d,
    wind_speed) layout. No-op (returns X, channel_names unchanged) if
    'u_10'/'v_10' aren't both present -- i.e. this only ever fires on the
    FOSI side, never on MESA's.

    X: (N, T, C, H, W) numpy array, C >= len(channel_names) (any channels
    appended past len(channel_names), e.g. a coastal-fraction channel, are
    passed through untouched at the end). Returns (X_new, new_channel_names).
    """
    channel_names = list(channel_names)
    if "u_10" not in channel_names or "v_10" not in channel_names:
        return X, channel_names

    u_idx = channel_names.index("u_10")
    v_idx = channel_names.index("v_10")
    n_orig = len(channel_names)
    speed = np.sqrt(X[:, :, u_idx] ** 2 + X[:, :, v_idx] ** 2)[:, :, None]

    keep = [i for i in range(n_orig) if i not in (u_idx, v_idx)]
    insert_pos = sum(1 for i in keep if i < u_idx)
    X_kept = X[:, :, keep]
    X_extra = X[:, :, n_orig:]  # anything appended past the original schema (e.g. coastal channel)

    X_new = np.concatenate(
        [X_kept[:, :, :insert_pos], speed, X_kept[:, :, insert_pos:], X_extra], axis=2,
    )
    new_channel_names = [channel_names[i] for i in keep]
    new_channel_names.insert(insert_pos, "wind_speed")
    return X_new, new_channel_names


# ==============================================================
# Land-sea mask
# ==============================================================

def build_land_sea_mask(hlat, hlon, bbox, bbox_regrid, weighted_grids_dir):
    """
    Regrid the native POP ocean/land mask onto the high-res (Y) target grid
    and clip it to bbox. Regridding weights are cached in
    weighted_grids_dir (keyed by bbox) so repeated jobs don't recompute the
    xesmf weights every run.

    A destination cell is called "land" only if its regridded ocean fraction is below
    0.1 (i.e. >90% land) -- fixed, not a tunable knob. Originally a 0.5 majority-rule
    threshold, tightened because that let mixed coastal cells (e.g. 40% ocean / 60%
    land) get classified as land, even though they still carry real, nonzero truth ice
    signal from their ocean fraction -- hard-zeroing the model's predictions there (see
    run_pipeline's `ocean_test` masking) then disagreed with that residual truth signal
    on every such cell, inflating IIEE (an ice presence/absence metric) even though
    domain-wide accuracy was fine. Lowering the threshold reclassifies most of those
    mixed cells as ocean, where the model is allowed to predict ice and isn't forced to
    zero. Confirmed not worth making adjustable beyond this: combining it with other
    fixes (the stochastic refiner, conservative-vs-bilinear truth regridding) was swept
    at 0.5 and never beat 0.1 (see recommended_config.md).

    Returns a (1, 1, H, W) float32 torch tensor: 1 = land, 0 = ocean.
    """
    nat_ice_hr = pop_tools.get_grid("POP_tx0.1v2")
    ice_lon = nat_ice_hr.TLONG % 360
    ocean_frac_native = (nat_ice_hr.KMT > 0).astype(np.float32)  # 1 = ocean, 0 = land

    lon_min_regrid = bbox_regrid["lon_min"] % 360
    lon_max_regrid = bbox_regrid["lon_max"] % 360
    lon_min = bbox["lon_min"] % 360
    lon_max = bbox["lon_max"] % 360

    mask_ice_hr = np.any(
        (nat_ice_hr.TLAT >= bbox_regrid["lat_min"])
        & (nat_ice_hr.TLAT <= bbox_regrid["lat_max"])
        & (ice_lon >= lon_min_regrid)
        & (ice_lon <= lon_max_regrid),
        axis=1,
    )

    grid_ice_hr = xr.Dataset({
        "lat": (["nlat", "nlon"], nat_ice_hr.TLAT.isel(nlat=mask_ice_hr).values),
        "lon": (["nlat", "nlon"], ice_lon.isel(nlat=mask_ice_hr).values),
    })
    ocean_frac_src = ocean_frac_native.isel(nlat=mask_ice_hr)

    dst_hr = xr.Dataset({"lat": ("lat", hlat), "lon": ("lon", hlon % 360)})

    os.makedirs(weighted_grids_dir, exist_ok=True)
    tag = f"{bbox['lat_min']}_{bbox['lat_max']}_{bbox['lon_min']}_{bbox['lon_max']}"
    weights_path = os.path.join(weighted_grids_dir, f"ice_hr_to_0p1deg_mask_{tag}.nc")
    reuse_weights = os.path.exists(weights_path)

    regridder_ice_to_0p1deg = xe.Regridder(
        grid_ice_hr,
        dst_hr,
        method="bilinear",
        periodic=True,
        filename=weights_path,
        reuse_weights=reuse_weights,
    )

    ocean_frac_reg = regridder_ice_to_0p1deg(ocean_frac_src)
    land_mask = (ocean_frac_reg < 0.1).astype(np.float32)  # 1 = land, 0 = ocean

    land_mask = land_mask.sel(
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
        lon=slice(lon_min, lon_max),
    )
    land_mask = torch.from_numpy(land_mask.values).float()[None, None, ...]
    return land_mask

# ==============================================================
# Projection helper + domain diagnostic figure
# ==============================================================

def rounded_boundary_path(proj, lon_min, lon_max, lat_min, lat_max, n=50):
    """Quadrilateral Axes-boundary path (in projected coords) tracing the given lon/lat box.
    Under a polar projection this comes out curved along the lat edges, so panels clipped to
    it read as "map-shaped" rather than a plain rectangular Axes frame."""
    lons = np.concatenate([
        np.linspace(lon_min, lon_max, n),
        np.full(n, lon_max),
        np.linspace(lon_max, lon_min, n),
        np.full(n, lon_min),
    ])
    lats = np.concatenate([
        np.full(n, lat_min),
        np.linspace(lat_min, lat_max, n),
        np.full(n, lat_max),
        np.linspace(lat_max, lat_min, n),
    ])
    boundary_pts = proj.transform_points(ccrs.PlateCarree(), lons, lats)
    return mpath.Path(boundary_pts[:, :2])


def make_polar_proj(bbox, n=50):
    central_lon = (bbox["lon_min"] + bbox["lon_max"]) / 2
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    boundary_path = rounded_boundary_path(
        proj, bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"], n
    )
    return proj, boundary_path, central_lon


def style_polar_ax(ax, proj, boundary_path, bbox, lon_=None, lat_=None, pad_frac=0.001):
    """Coastlines, land, domain boundary, Kivalina marker, gridlines -- shared
    styling for every panel plotted on the NorthPolarStereo domain.

    If lon_/lat_ are given, the extent is zoomed to that panel's actual data
    bounds (with a small padding margin) instead of the full bbox domain --
    the boundary is then recomputed for that same zoomed box (rather than
    reusing `boundary_path`, which is sized for the full `bbox` and would
    fall outside the zoomed view, leaving the panel with matplotlib's
    default square Axes frame instead of the intended map-shaped one)."""
    lon_min = bbox["lon_min"] % 360
    lon_max = bbox["lon_max"] % 360

    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)

    if lon_ is not None and lat_ is not None:
        lo0, lo1 = float(np.min(lon_)), float(np.max(lon_))
        la0, la1 = float(np.min(lat_)), float(np.max(lat_))
        pad_lo = (lo1 - lo0) * pad_frac or 0.5
        pad_la = (la1 - la0) * pad_frac or 0.5
        ext_lo0, ext_lo1 = lo0 - pad_lo, lo1 + pad_lo
        ext_la0, ext_la1 = la0 - pad_la, la1 + pad_la
        ax.set_extent([ext_lo0, ext_lo1, ext_la0, ext_la1], crs=ccrs.PlateCarree())
        panel_boundary = rounded_boundary_path(proj, ext_lo0, ext_lo1, ext_la0, ext_la1)
    else:
        ax.set_extent([lon_min, lon_max, bbox["lat_min"], bbox["lat_max"]], crs=ccrs.PlateCarree())
        panel_boundary = boundary_path

    ax.set_boundary(panel_boundary, transform=proj)
    ax.plot(KIVALINA_LON_360, KIVALINA_LAT, marker="*", color="red", markersize=10, transform=ccrs.PlateCarree())
    ax.text(KIVALINA_LON_360 + 1, KIVALINA_LAT + 0.35, "Kivalina", color="red", fontsize=7, transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, linestyle="--", alpha=0.4)


def plot_domain_diagnostic(output_dir, bbox, llat, llon, hlat, hlon, land_mask, X, Y, proj, boundary_path, central_lon):
    lon_min = bbox["lon_min"] % 360
    lon_max = bbox["lon_max"] % 360

    fig = plt.figure(figsize=(15, 8))
    flat_proj = ccrs.PlateCarree(central_longitude=central_lon)

    ax1 = fig.add_subplot(2, 3, 1, projection=flat_proj)
    cf1 = ax1.pcolormesh(hlon, hlat, land_mask[0, 0], cmap="Greys", vmin=0, vmax=1, shading="auto", transform=ccrs.PlateCarree())
    ax1.coastlines(resolution="50m")
    ax1.set_extent([lon_min, lon_max, bbox["lat_min"], bbox["lat_max"]], crs=ccrs.PlateCarree())
    fig.colorbar(cf1, ax=ax1, shrink=0.7)
    ax1.set_title("Mask")

    ax2 = fig.add_subplot(2, 3, 2, projection=flat_proj)
    cf2 = ax2.pcolormesh(llon, llat, X[0, 0, 0], cmap="Blues", vmin=0, vmax=6, shading="auto", transform=ccrs.PlateCarree())
    ax2.coastlines(resolution="50m")
    ax2.set_extent([lon_min, lon_max, bbox["lat_min"], bbox["lat_max"]], crs=ccrs.PlateCarree())
    fig.colorbar(cf2, ax=ax2, shrink=0.7)
    ax2.set_title("X")

    ax3 = fig.add_subplot(2, 3, 3, projection=flat_proj)
    cf3 = ax3.pcolormesh(hlon, hlat, Y[0, 0, 0], cmap="Blues", vmin=0, vmax=6, shading="auto", transform=ccrs.PlateCarree())
    ax3.coastlines(resolution="50m")
    ax3.set_extent([lon_min, lon_max, bbox["lat_min"], bbox["lat_max"]], crs=ccrs.PlateCarree())
    fig.colorbar(cf3, ax=ax3, shrink=0.7)
    ax3.set_title("Y")

    proj_panels = [
        (hlon, hlat, land_mask[0, 0], "Mask", "Greys", 0, 1),
        (llon, llat, X[0, 0, 0], "X", "Blues", 0, 6),
        (hlon, hlat, Y[0, 0, 0], "Y", "Blues", 0, 6),
    ]
    for i, (lon_i, lat_i, field, title, cmap, vmin, vmax) in enumerate(proj_panels):
        ax = fig.add_subplot(2, 3, 4 + i, projection=proj)
        cf = ax.pcolormesh(lon_i, lat_i, field, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        style_polar_ax(ax, proj, boundary_path, bbox)
        ax.set_title(title, fontsize=12)
        plt.colorbar(cf, ax=ax, shrink=0.7, label="Units")

    plt.tight_layout()
    save_fig(fig, output_dir, "domain_mask_overview.png")


# ==============================================================
# Patch extraction (sliding-window) and single sub-domain cropping
# ==============================================================

def extract_patches(X, Y, land_mask, context_size, target_size, stride, llon=None, llat=None, hlon=None, hlat=None):
    """
    llon/llat/hlon/hlat are optional 1D coordinate arrays for the low-res (X)
    and high-res (Y) grids. When supplied, the function additionally returns
    tile_ids (which tile each output patch came from) and tile_geometry (the
    lon/lat slice for each tile's context and target windows), so a later
    patch index can be mapped back to real map coordinates -- used by the
    ensemble/error figures to plot patches on a cartopy projection.
    """
    N, C, H, W = X.shape
    _, C_out, H_hi, W_hi = Y.shape
    context_h, context_w = context_size
    target_h, target_w = target_size

    if context_h > H or context_w > W:
        raise ValueError(f"Context size {context_size} exceeds LR grid size ({H}, {W})")
    if target_h > context_h or target_w > context_w:
        raise ValueError(f"Target size {target_size} must be <= context size {context_size}")

    scale_y = H_hi / H
    scale_x = W_hi / W
    pad_h = (context_h - target_h) // 2
    pad_w = (context_w - target_w) // 2

    tiles = []
    for i in range(0, H - context_h + 1, stride):
        for j in range(0, W - context_w + 1, stride):
            y0 = round((i + pad_h) * scale_y)
            x0 = round((j + pad_w) * scale_x)
            y1 = y0 + int(round(target_h * scale_y))
            x1 = x0 + int(round(target_w * scale_x))
            mask_patch = land_mask[0, :, y0:y1, x0:x1]  # (C_mask, target_h, target_w)
            tiles.append((i, j, y0, x0, y1, x1, mask_patch))

    X_patches, Y_patches, mask_patches, tile_ids = [], [], [], []
    for n_ in range(N):
        for t_idx, (i, j, y0, x0, y1, x1, mask_patch) in enumerate(tiles):
            X_patches.append(X[n_, :, i:i + context_h, j:j + context_w])
            Y_patches.append(Y[n_, :, y0:y1, x0:x1])
            mask_patches.append(mask_patch)
            tile_ids.append(t_idx)

    X_patches = torch.stack(X_patches)
    Y_patches = torch.stack(Y_patches)
    mask_patches = torch.stack(mask_patches)
    tile_ids = torch.tensor(tile_ids, dtype=torch.long)

    tile_geometry = None
    if llon is not None and llat is not None and hlon is not None and hlat is not None:
        tile_geometry = []
        for (i, j, y0, x0, y1, x1, _mask_patch) in tiles:
            tile_geometry.append({
                "context_lon": llon[j:j + context_w],
                "context_lat": llat[i:i + context_h],
                "target_lon": hlon[x0:x1],
                "target_lat": hlat[y0:y1],
            })

    return X_patches, Y_patches, mask_patches, tile_ids, tile_geometry


def validate_subdomain(subdomain, bbox):
    if subdomain["lat_min"] >= subdomain["lat_max"]:
        raise ValueError("subdomain lat_min must be < lat_max")

    lon_lo = subdomain["lon_min"] % 360
    lon_hi = subdomain["lon_max"] % 360
    if lon_lo >= lon_hi:
        raise ValueError("subdomain lon_min must be < lon_max (after %360 normalization)")

    bbox_lo = bbox["lon_min"] % 360
    bbox_hi = bbox["lon_max"] % 360

    failed = []
    if subdomain["lat_min"] < bbox["lat_min"]:
        failed.append("lat_min")
    if subdomain["lat_max"] > bbox["lat_max"]:
        failed.append("lat_max")
    if lon_lo < bbox_lo:
        failed.append("lon_min")
    if lon_hi > bbox_hi:
        failed.append("lon_max")

    if failed:
        raise ValueError(
            f"Requested sub-domain {subdomain} is not fully inside the ML domain {bbox} "
            f"(failed: {failed}). Choose bounds within "
            f"lat [{bbox['lat_min']}, {bbox['lat_max']}], lon [{bbox['lon_min']}, {bbox['lon_max']}]."
        )


def crop_indices(coord_1d, lo, hi):
    """Index slice [start, stop) into a 1D coordinate array covering [lo, hi]."""
    coord = np.asarray(coord_1d)
    inside = np.where((coord >= lo) & (coord <= hi))[0]
    if inside.size == 0:
        raise ValueError(f"No grid points found in range [{lo}, {hi}] for this coordinate.")
    return int(inside.min()), int(inside.max()) + 1


def extract_full_domain(X, Y, land_mask, llon, llat, hlon, hlat, subdomain, divisor=8):
    """
    No-patch alternative to extract_patches(): crop X, Y and land_mask to
    `subdomain` and treat every sample's full cropped field as one
    training/eval example (instead of tiling). Returns the same signature
    as extract_patches so the rest of the pipeline (model, training loop,
    evaluation, figures) doesn't need to know whether patches were used.

    divisor: the low-res crop must have both dims divisible by this (default
        8, matching the UNet's 3 stride-2 pooling layers).
    """
    lat_min, lat_max = subdomain["lat_min"], subdomain["lat_max"]
    lon_min = subdomain["lon_min"] % 360
    lon_max = subdomain["lon_max"] % 360

    li0, li1 = crop_indices(llat, lat_min, lat_max)
    lj0, lj1 = crop_indices(np.asarray(llon) % 360, lon_min, lon_max)
    hi0, hi1 = crop_indices(hlat, lat_min, lat_max)
    hj0, hj1 = crop_indices(np.asarray(hlon) % 360, lon_min, lon_max)

    lr_h, lr_w = li1 - li0, lj1 - lj0
    # The UNet encoder has `divisor`-implied stride-2 max-pools (3 by default),
    # so the low-res crop needs H and W each divisible by `divisor` (and at
    # least `divisor`) or the bottleneck collapses to a zero-sized tensor deep
    # inside forward().
    if lr_h < divisor or lr_w < divisor or lr_h % divisor != 0 or lr_w % divisor != 0:
        raise ValueError(
            f"Sub-domain {subdomain} crops to a low-res grid of shape "
            f"({lr_h}, {lr_w}), but the UNet's pooling layers require both "
            f"dimensions to be multiples of {divisor} (and >= {divisor}). "
            "Widen the sub-domain, or nudge lat/lon bounds so the low-res "
            f"crop lands on a multiple of {divisor} in both directions."
        )
    # Additionally, the bottleneck (H/divisor, W/divisor) must have MORE THAN
    # ONE total spatial element, or InstanceNorm2d inside enc4/bottleneck
    # raises "Expected more than 1 spatial element when training" -- this
    # bites exactly the (divisor, divisor) case (bottleneck collapses to a
    # single 1x1 pixel), which the check above alone doesn't catch.
    if (lr_h // divisor) * (lr_w // divisor) <= 1:
        raise ValueError(
            f"Sub-domain {subdomain} crops to a low-res grid of shape "
            f"({lr_h}, {lr_w}), whose bottleneck ({lr_h // divisor}, {lr_w // divisor}) has "
            "only one spatial element -- InstanceNorm2d can't compute a variance over a "
            f"single point and will raise during training. Widen at least one "
            f"dimension to {2 * divisor}+ so the bottleneck has more than one pixel."
        )

    X_crop = X[:, :, li0:li1, lj0:lj1]
    Y_crop = Y[:, :, hi0:hi1, hj0:hj1]
    mask_crop = land_mask[0, :, hi0:hi1, hj0:hj1]  # (C_mask, h, w)

    N = X.shape[0]
    mask_patches = mask_crop.unsqueeze(0).repeat(N, 1, 1, 1)
    tile_ids = torch.zeros(N, dtype=torch.long)

    tile_geometry = [{
        "context_lon": llon[lj0:lj1],
        "context_lat": llat[li0:li1],
        "target_lon": hlon[hj0:hj1],
        "target_lat": hlat[hi0:hi1],
    }]

    return X_crop, Y_crop, mask_patches, tile_ids, tile_geometry


# ==============================================================
# UNet
# ==============================================================

class GaussianNoiseStage(nn.Module):
    """
    One decoder stage's independent stochastic-ensemble noise source
    (2026-09-16 redesign, replacing the old shared/resized global latent z).

    Draws fresh white noise at this stage's own resolution -- never resized
    from another stage, so all four decoder stages (d3/d2/d1/final) get
    genuinely independent stochastic content instead of one coarse field
    reprojected four ways. Correlates it with a shared, translation-invariant
    Gaussian kernel -- deliberately not LocallyConnected2d's per-location
    independent weights (removed permanently, see UNet's docstring): this
    project confirmed that mechanism is the source of a patchy per-pixel
    artifact for sea ice thickness specifically, a physically continuous
    field unlike the sharp-fronted atmospheric variables EnScale (Schillinger
    et al., 2025) was designed for. A shared kernel can't reproduce that
    artifact by construction -- every pixel gets the identical kernel, so
    neighboring outputs can't discontinuously diverge the way independently-
    learned per-location weights allowed.

    kernel_size is fixed at construction (5 by default, chosen once for all
    four stages) and not learnable -- a discrete tap count isn't a
    differentiable quantity, and this project's EnScale-inspired precedent
    (their k=3..6 pyramid levels) turned out to describe grid *resolution*
    doubling, not a growing local-aggregation window, so there's no basis
    for varying the footprint by stage either. (Exposed as a constructor
    argument -- and, at the UNet level, a `--noise-kernel-size` CLI flag --
    purely as a 2026-09-17 sensitivity-test knob bracketing the chosen
    default of 5 with 3 and 9; not intended to become a permanent
    per-deployment tuning parameter.) What *is* learnable, each a single
    scalar shared across this whole stage (not per-channel, when
    channels=1 -- see below -- these are identical anyway): the kernel's
    width (log_width, spatial correlation length within the fixed
    footprint) and the output's overall amplitude (log_sigma). Both
    softplus-parameterized to stay positive. Trained jointly with the rest
    of the network via the ordinary (coastal-weighted) energy loss -- no
    post-hoc calibration step, since one won't be available when deploying
    against real observations with no matching high-res truth to calibrate
    against, and no separate calibration loss term either (by explicit
    choice).

    Single output channel by default. This model has one target variable
    (sea ice thickness); the old stochastic_refine design's
    noise_mix_channels=4 most likely mirrored the EnScale paper's own
    per-*variable* channel count for their multi-variable downscaling
    target ("Gaussian noise is concatenated to the four variables in each
    pixel"), not a generic "more noise channels is richer" principle -- for
    a single-variable model that convention has nothing to map onto beyond
    1. (Also exposed as a constructor argument / `--noise-channels` CLI
    flag purely as a 2026-09-17 sensitivity-test knob checking that
    reasoning empirically against channels=2; not intended to stay a
    permanent tuning parameter.)
    """

    def __init__(self, channels=1, kernel_size=5, init_width=1.5, init_sigma=1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.log_width = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(init_width)))))
        self.log_sigma = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(init_sigma)))))
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        self.register_buffer("coords", coords)

    def _kernel(self, device, dtype):
        width = F.softplus(self.log_width).to(device=device, dtype=dtype)
        coords = self.coords.to(device=device, dtype=dtype)
        g = torch.exp(-(coords ** 2) / (2 * width ** 2 + 1e-6))
        g = g / g.sum()
        kernel2d = g[:, None] * g[None, :]
        return kernel2d.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, batch_size, height, width, device, dtype, deterministic=False):
        if deterministic:
            return torch.zeros(batch_size, self.channels, height, width, device=device, dtype=dtype)
        eps = torch.randn(batch_size, self.channels, height, width, device=device, dtype=dtype)
        kernel = self._kernel(device, dtype)
        blurred = F.conv2d(eps, kernel.expand(self.channels, 1, -1, -1), padding=self.kernel_size // 2, groups=self.channels)
        # Undo exactly the variance the blur itself introduced (Var(output) =
        # Var(input) * Sum(kernel_weight^2) for unit-variance input), so
        # log_sigma alone is the final, literal amplitude -- independent of
        # whatever correlation length log_width settles on.
        renorm_factor = torch.sqrt((kernel ** 2).sum())
        renormed = blurred / renorm_factor
        sigma = F.softplus(self.log_sigma)
        return renormed * sigma


class WindowedSelfAttention2d(nn.Module):
    """
    Non-overlapping windowed multi-head self-attention over a 2D feature
    map (Swin-style W-MSA, not shifted), applied once at the very end of the
    decoder -- a "sliding window" attention mechanism placed at the decoder
    end rather than the encoder, per the user's explicit framing (2026-08-06
    conversation): full quadratic self-attention over the whole up-sampled
    field is too expensive at these domain sizes (e.g. ~150x310 for the
    medium no-patch domain), and encoder-side attention operates several
    layers upstream of where the model's coastal smoothing is actually
    introduced (the bilinear decoder). Restricting attention to small local
    windows keeps cost linear in H*W (O(H*W*window_size^2) instead of
    O((H*W)^2)) while still letting each output pixel incorporate its local
    neighborhood right before out_conv.

    Pads H, W up to a multiple of window_size (reflect padding) before
    windowing and crops back after, so this works for any domain crop size,
    not just ones divisible by window_size -- no new constraint on top of
    the existing divisor-8/16 sub-domain sizing rule.

    max_window_batch chunks the flattened (B*nH*nW, ws*ws, C) call to
    nn.MultiheadAttention instead of running it in one shot, AND each chunk
    runs under torch.utils.checkpoint so its intermediate activations aren't
    retained for backward (recomputed on demand instead). Both pieces turned
    out necessary the hard way (2026-08-06): at the medium no-patch domain
    (~150x310 up-sampled) with window_size=8 and a real training batch
    (batch_size=16, K=20), B*nH*nW reaches ~237k. An un-chunked call tries
    to materialize a (237k*num_heads, 64, 64) attention-score tensor and
    OOMs outright (tried to allocate 14.47 GiB). Chunking alone (bounding
    each individual attention call's peak to ~256 MiB) still OOM'd on the
    resubmitted jobs -- turns out standard autograd retains *every* chunk's
    saved activations simultaneously across the whole sequential loop, so
    the *total* memory across ~58 chunks accumulated to more than the GPU
    had left after the rest of the model's own activations (the medium-
    domain baseline already runs close to the 32GB ceiling at K=20/batch=16
    without any attention module at all). Per-chunk checkpointing discards
    each chunk's activations immediately after producing its output and
    recomputes them during backward, so only one chunk's activations exist
    in memory at a time regardless of how many chunks there are -- the fix
    that actually worked, confirmed against the same OOMing jobs.

    The standalone smoke test that validated this module before use only
    used a tiny synthetic batch (B=2) and never exercised a batch size close
    to a real training step, which is why none of this was caught until the
    actual comparison-batch jobs ran on the GPU.
    """

    def __init__(self, channels, window_size=8, num_heads=4, max_window_batch=4096):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        self.window_size = window_size
        self.max_window_batch = max_window_batch
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def _run_chunk(self, x_chunk):
        attn_chunk, _ = self.attn(x_chunk, x_chunk, x_chunk)
        return self.norm(attn_chunk + x_chunk)  # residual, then normalize

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_p = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
        Hp, Wp = H + pad_h, W + pad_w

        # (B, C, Hp, Wp) -> non-overlapping ws x ws windows -> (B*nH*nW, ws*ws, C)
        x_w = x_p.unfold(2, ws, ws).unfold(3, ws, ws)  # (B, C, nH, nW, ws, ws)
        nH, nW = x_w.shape[2], x_w.shape[3]
        x_w = x_w.permute(0, 2, 3, 4, 5, 1).reshape(B * nH * nW, ws * ws, C)

        chunks = []
        for i in range(0, x_w.shape[0], self.max_window_batch):
            x_chunk = x_w[i:i + self.max_window_batch]
            if self.training and x_chunk.requires_grad:
                chunks.append(torch_checkpoint.checkpoint(self._run_chunk, x_chunk, use_reentrant=False))
            else:
                chunks.append(self._run_chunk(x_chunk))
        attn_out = torch.cat(chunks, dim=0)

        attn_out = attn_out.reshape(B, nH, nW, ws, ws, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        return attn_out[:, :, :H, :W]


class UNet(nn.Module):

    def __init__(self, in_channels, mask_channels=1,
                 attention_end=False, attn_window_size=8, attn_num_heads=4,
                 noise_channels=1, noise_kernel_size=5, late_mask_fusion=False):
        """
        Note on stochastic ensemble generation (redesigned 2026-09-16): each
        of the four decoder stages (d3, d2, d1, final) has its own
        independent GaussianNoiseStage -- a fresh noise draw at that stage's
        own resolution, spatially correlated by a shared (translation-
        invariant) Gaussian kernel, renormalized, and scaled by a learnable
        per-stage amplitude -- see GaussianNoiseStage's own docstring for the
        full mechanism and why it replaced the previous two designs:

        1. The original single global latent z: one noise field drawn once
           at the bottleneck's resolution, then merely bilinearly resized
           and 1x1-conv-projected into every stage. Resizing a coarse field
           cannot manufacture genuinely new high-frequency content at finer
           stages, and reusing the identical field at every stage let its
           effects compound/reinforce rather than average down the way
           independent per-stage noise does -- a plausible contributor to
           this architecture's measured over-dispersion (Spread/Error ~1.87)
           once the second mechanism below was removed.
        2. stochastic_refine/enscale_net (an EnScale-style (Schillinger et
           al., 2025) per-location LocallyConnected2d noise-mixing layer,
           plus the full EnScaleStage decoder replacement) -- removed
           permanently (2026-09-16). Confirmed source of a patchy,
           unrealistic per-pixel texture in every individual ensemble member
           (see recommended_config.md's 2026-09-11 finding and the
           MESA_no_stochastic_refine_avg/MESA_z_inject_final_avg comparison
           batches). It also measurably helped calibration, but that
           tradeoff wasn't worth reintroducing the artifact for.

        attention_end: applies WindowedSelfAttention2d (non-overlapping local
        self-attention, "sliding window" but at the decoder's end rather than
        the encoder) to the final 32-channel decoder feature map, after the
        mask+noise fusion below (see forward()) -- so if ever enabled, it
        operates on mask-aware features. Default False = unchanged
        architecture. attn_window_size/attn_num_heads size that attention
        block; channels (32) must be divisible by attn_num_heads.

        The high-res land mask and deep mask-processing capacity (formerly
        the optional deep_mask_head toggle, now unconditional) are folded
        into the *same* fusion step as the final stage's noise, before that
        fusion happens -- not concatenated afterward as a separate late step
        the way earlier versions of this model did. That means the mask is
        visible during the noise-mixing computation itself, not just to
        cleanup layers downstream of an already-finalized mix -- see
        forward()'s Output section.

        noise_channels, noise_kernel_size: passed straight through to every
        GaussianNoiseStage (see its docstring). 2026-09-17 sensitivity-test
        knobs only, both default to the settled values (1, 5) -- not
        intended as permanent per-run tuning parameters.

        late_mask_fusion: 2026-09-17 sensitivity-test toggle, default False
        (= the current architecture described above). When True, reverts
        *only* the mask's timing at the final stage back to how every
        earlier version of this model did it -- concatenated after a
        noise-only fusion (and after attention_end, if enabled) rather than
        before -- while keeping the new per-stage noise cascade everywhere,
        including at the final stage, unchanged. Exists to isolate whether
        moving the mask earlier (vs. the noise-cascade redesign itself) is
        what drives any change in coastal/IIEE behavior; not intended to
        stay in the codebase once that question is answered.

        Note on the target this network predicts: forward() returns the
        decoder output directly, with no internal bilinear-base residual
        add. That's deliberate -- run_pipeline() already re-expresses the
        target as a standardized physical increment (Y_phys - baseline_phys,
        normalized by the increment's own train-set mean/std, matching
        Brajard et al. 2026 (EGUsphere)'s Eq. 4 / NVIDIA CorrDiff's residual
        construction) before this model ever sees it, and the baseline is
        added back externally, in physical space, at eval time. An internal
        add of a bilinear-upsampled *encoder input* here (in X's own
        normalized space) would mix two differently-scaled quantities into
        an uninterpretable, badly-conditioned target -- see run_pipeline's
        baseline/increment construction for the actual residual arithmetic.
        """

        super().__init__()
        self.attention_end = attention_end
        self.mask_channels = mask_channels
        self.late_mask_fusion = late_mask_fusion

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)

        self.bottleneck = conv_block(512, 512)

        # Per-stage independent noise sources (see GaussianNoiseStage) plus a
        # 1x1-conv projection into that stage's decoder width -- replaces the
        # old shared/resized global z + z_proj_d3/d2/d1/final.
        self.noise_d3 = GaussianNoiseStage(channels=noise_channels, kernel_size=noise_kernel_size)
        self.noise_proj_d3 = nn.Conv2d(noise_channels, 256, 1)
        self.noise_d2 = GaussianNoiseStage(channels=noise_channels, kernel_size=noise_kernel_size)
        self.noise_proj_d2 = nn.Conv2d(noise_channels, 128, 1)
        self.noise_d1 = GaussianNoiseStage(channels=noise_channels, kernel_size=noise_kernel_size)
        self.noise_proj_d1 = nn.Conv2d(noise_channels, 64, 1)
        self.noise_final = GaussianNoiseStage(channels=noise_channels, kernel_size=noise_kernel_size)
        self.noise_proj_final = nn.Conv2d(noise_channels, 64, 1)

        # Decoder
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(512, 256, 3, padding=1))
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(256, 128, 3, padding=1))
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(128, 64, 3, padding=1))
        self.dec1 = conv_block(128, 64)

        # Concatenation adapters
        self.concat_d3 = nn.Conv2d(256 + 256, 256, 1)
        self.concat_d2 = nn.Conv2d(128 + 128, 128, 1)
        self.concat_d1 = nn.Conv2d(64 + 64, 64, 1)

        # Upsample-then-conv instead of a ConvTranspose2d(kernel_size=stride=4): with
        # kernel_size == stride, every 4x4 output block is generated independently from a
        # single input pixel through the same shared kernel, so any asymmetry the kernel
        # learns repeats identically in every block -- a visible tiling/checkerboard
        # artifact once the domain is wide enough to show several repeats.
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        if self.late_mask_fusion:
            # Ablation-only (see docstring): reconstructs every earlier version's mask
            # timing -- a noise-only fusion down to 32 channels first (2 layers, no
            # activation after the very last conv, matching the old final_conv exactly),
            # then (after attention_end, if any) the mask concatenates in and gets its
            # own 2-layer block (matching the old, now-permanently-folded-in
            # deep_mask_head exactly) before out_conv.
            self.final_conv = nn.Sequential(
                nn.Conv2d(64 + 64, 32, 3, padding=1),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
            )
            self.mask_head = nn.Sequential(
                nn.Conv2d(32 + mask_channels, 32, 3, padding=1),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            # Final fusion: upsampled decoder features + this stage's projected noise +
            # the high-res land mask, concatenated together *before* any of the four conv
            # layers run -- so the mixing computation itself is mask-aware, not just
            # cleanup layers downstream of an already-finalized mix (see class docstring).
            # Depth/activation pattern (4 conv+IN+ReLU-ish layers, no activation before
            # the very last conv) matches what the old final_conv (2 layers) +
            # deep_mask_head (2 more, now unconditional) produced together, just with
            # mask folded in at the start instead of after the first 2 layers.
            self.final_conv = nn.Sequential(
                nn.Conv2d(64 + 64 + mask_channels, 32, 3, padding=1),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
            )
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1)

        if self.attention_end:
            # Applied after the mask+noise fusion above -- see class docstring.
            self.attn_end = WindowedSelfAttention2d(32, window_size=attn_window_size, num_heads=attn_num_heads)

    def forward(self, x, up_size, mask=None, deterministic=False):

        B, C, H, W = x.shape

        if self.mask_channels > 0:
            if mask is None:
                raise ValueError(
                    "mask is required when mask_channels > 0: the land mask is "
                    "concatenated into the final noise/feature fusion, not added "
                    "as a separate later step."
                )
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d3 = self.up3(b)
        n3 = self.noise_proj_d3(self.noise_d3(B, *d3.shape[-2:], d3.device, d3.dtype, deterministic))
        d3 = self.concat_d3(torch.cat([d3, n3], dim=1))
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        n2 = self.noise_proj_d2(self.noise_d2(B, *d2.shape[-2:], d2.device, d2.dtype, deterministic))
        d2 = self.concat_d2(torch.cat([d2, n2], dim=1))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        n1 = self.noise_proj_d1(self.noise_d1(B, *d1.shape[-2:], d1.device, d1.dtype, deterministic))
        d1 = self.concat_d1(torch.cat([d1, n1], dim=1))
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        up = self.final_upsample(d1)
        up = F.interpolate(up, size=up_size, mode="bilinear", align_corners=False)
        nf = self.noise_proj_final(self.noise_final(B, *up.shape[-2:], up.device, up.dtype, deterministic))

        if self.late_mask_fusion:
            # Ablation-only path (see docstring): mask stays out of the fusion until
            # after attention_end, matching every earlier version of this model.
            out = self.final_conv(torch.cat([up, nf], dim=1))
            if self.attention_end:
                out = self.attn_end(out)
            if self.mask_channels > 0:
                out = torch.cat([out, mask.to(out.dtype)], dim=1)
                out = self.mask_head(out)
        else:
            fuse_parts = [up, nf]
            if self.mask_channels > 0:
                fuse_parts.append(mask.to(up.dtype))
            out = self.final_conv(torch.cat(fuse_parts, dim=1))

            if self.attention_end:
                # Local windowed self-attention on the mask+noise-fused 32-channel
                # decoder output -- see class docstring.
                out = self.attn_end(out)

        out = self.out_conv(out)

        # `out` already *is* the predicted standardized increment: the baseline was
        # subtracted from the target in physical space before normalization
        # (run_pipeline), so no internal residual add belongs here -- see the class
        # docstring's note on what this network predicts.
        return out

# ==============================================================
# Losses
# ==============================================================

def energy_loss(preds, y, weight=None, beta=1.0):
    """
    Energy-score loss used for stochastic prediction, optionally per-pixel weighted.

    Args:
        preds (torch.Tensor): Ensemble predictions of shape [B, K, C, H, W].
        y (torch.Tensor): Ground-truth targets of shape [B, C, H, W].
        weight: optional non-negative weight map, either (H, W) (shared across the
            batch) or (B, H, W) (per-sample, e.g. when patches carry different land
            masks). None = uniform, identical to the original unweighted loss. Assumes
            a single-channel target (C == 1, true for SIT-only Y here).
        beta (float, optional): Power parameter of the energy score. Default is 1.

    Returns:
        torch.Tensor: Scalar energy-score loss.
    """
    B, K_, C, H, W = preds.shape

    flat_preds = preds.reshape(B, K_, -1)
    flat_y = y.reshape(B, 1, -1)

    if weight is not None:
        assert C == 1, "per-pixel `weight` assumes a single-channel target (Y is SIT-only)."
        # sqrt(weight) rescaling turns a weighted L2 distance into a plain L2 distance of
        # the rescaled vectors: ||sqrt(w)*(a-b)||_2 == sqrt(sum(w*(a-b)**2)), so the
        # existing (efficient) vector_norm/cdist calls below stay exact and unchanged.
        w = torch.as_tensor(weight, dtype=preds.dtype, device=preds.device)
        sqrt_w = torch.sqrt(w).reshape(*w.shape[:-2], H * W)
        if sqrt_w.dim() == 1:
            flat_preds = flat_preds * sqrt_w
            flat_y = flat_y * sqrt_w
        else:  # (B, H*W) per-sample -> broadcast over the K/ensemble axis
            flat_preds = flat_preds * sqrt_w.unsqueeze(1)
            flat_y = flat_y * sqrt_w.unsqueeze(1)

    EPS = 0.0 if float(beta).is_integer() else 1e-5

    s1 = (torch.linalg.vector_norm(flat_preds - flat_y, ord=2, dim=2) + EPS).pow(beta).mean()
    s2 = (torch.cdist(flat_preds, flat_preds, p=2) + EPS).pow(beta).mean() * K_ / (K_ - 1)

    return s1 - 0.5 * s2


def coastal_band_mask(land_mask, coastal_width=5):
    """
    Boolean (..., H, W) mask: True for ocean cells within `coastal_width` pixels of land,
    False for land cells and for open-ocean cells farther from land. Dilation is done via
    max-pool on the binary land mask -- a dependency-free stand-in for
    scipy.ndimage.binary_dilation. `land_mask` can carry leading batch dims (e.g. (N, H, W)
    for per-sample masks that vary across patches).
    """
    land = torch.as_tensor(land_mask, dtype=torch.float32)
    lead_shape = land.shape[:-2]
    H, W = land.shape[-2:]
    land_dilated = F.max_pool2d(
        land.reshape(-1, 1, H, W), kernel_size=2 * coastal_width + 1, stride=1, padding=coastal_width
    )
    land_dilated = land_dilated.reshape(*lead_shape, H, W)
    return (land_dilated > 0.5) & (land <= 0.5)


def build_coastal_weight_map(land_mask, coastal_width=5, coastal_boost=2.0):
    """
    Per-pixel loss weight: 1 everywhere by default, boosted to `coastal_boost` over ocean
    cells within `coastal_width` pixels of land (see `coastal_band_mask`). Supports the
    same leading batch dims as `coastal_band_mask`.

    Land is deliberately NOT zeroed out here. An earlier version weighted land at 0 (on
    the reasoning that it's ~constant and just dilutes the gradient), but land is over
    half of this domain, and the model's residual architecture (base + correction, with
    no other supervision over land) has nothing else forcing its land predictions toward
    the correct ~0 -- excluding land from the loss let those predictions drift to
    physically nonsensical values (~0.4m mean, vs. ~0.02m when land stays in the loss),
    which swamped every domain-wide metric even though ocean-only accuracy genuinely
    improved. Keeping land at the baseline weight of 1 preserves that ocean/coastal gain
    without breaking the rest of the domain.
    """
    land = torch.as_tensor(land_mask, dtype=torch.float32)
    band = coastal_band_mask(land, coastal_width=coastal_width)
    return torch.where(band, torch.full_like(land, coastal_boost), torch.ones_like(land))


# ==============================================================
# Train / evaluate
# ==============================================================

def train_model(model, optimizer, X_train, Y_train, mask_train, device, K, num_epochs, batch_size,
                 coastal_width=5, coastal_boost=2.0, beta=1.0, verbose=True):
    loss_array = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        idx = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), batch_size):
            optimizer.zero_grad()

            batch_idx = idx[i:i + batch_size]
            X_batch = X_train[batch_idx].to(device)
            Y_batch = Y_train[batch_idx].to(device)
            mask_batch = mask_train[batch_idx].to(device)

            B = X_batch.shape[0]
            X_rep = X_batch.repeat_interleave(K, dim=0)
            mask_rep = mask_batch.repeat_interleave(K, dim=0)

            preds = model(X_rep, up_size=Y_batch.shape[-2:], mask=mask_rep)
            preds = preds.reshape(B, K, preds.shape[1], preds.shape[2], preds.shape[3])

            # Per-sample loss weight: land at baseline weight 1, coastal ocean band
            # up-weighted -- built fresh per batch since mask_batch can vary across
            # samples (sliding-window patches each cover a different piece of coastline).
            # Land is now also hard-masked to exactly 0 in the model's forward() itself
            # (see UNet.forward), so its weight here mostly just keeps its near-zero loss
            # contribution from being silently dropped.
            coastal_weight = build_coastal_weight_map(mask_batch[:, 0], coastal_width, coastal_boost)
            loss = energy_loss(preds, Y_batch, weight=coastal_weight, beta=beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * B
            del X_batch, Y_batch, mask_batch, X_rep, mask_rep, preds, loss

        epoch_loss /= X_train.size(0)
        loss_array.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.6f}")

    return loss_array


def evaluate_model(model, X_test, Y_test, mask_test, device, K_eval, eval_batch_size):
    model.eval()
    preds_all, preds_mean, preds_std, preds_det = [], [], [], []

    with torch.inference_mode():
        for i in range(0, X_test.shape[0], eval_batch_size):
            X_batch = X_test[i:i + eval_batch_size].to(device)
            Y_batch = Y_test[i:i + eval_batch_size].to(device)
            mask_batch = mask_test[i:i + eval_batch_size].to(device)

            # z=None (default) makes forward() sample its own fresh global latent
            # noise, so each ensemble member gets independent stochastic detail
            # without duplicating that sampling logic here. deterministic=True
            # (below) forces z to exact zero instead.
            ensemble_preds = []
            for k in range(K_eval):
                pred = model(X_batch, Y_batch.shape[-2:], mask=mask_batch)
                ensemble_preds.append(pred)

            preds = torch.stack(ensemble_preds, dim=0).permute(1, 0, 2, 3, 4)
            pred_mean = preds.mean(dim=1)
            pred_std = preds.std(dim=1)

            pred_det = model(X_batch, Y_batch.shape[-2:], mask=mask_batch, deterministic=True)

            preds_all.append(preds.cpu())
            preds_mean.append(pred_mean.cpu())
            preds_std.append(pred_std.cpu())
            preds_det.append(pred_det.cpu())

    return (
        torch.cat(preds_all, dim=0),
        torch.cat(preds_mean, dim=0),
        torch.cat(preds_std, dim=0),
        torch.cat(preds_det, dim=0),
    )


def plot_loss_curve(output_dir, loss_array, y_train_shape):
    epochs = np.arange(1, len(loss_array) + 1)
    norm_factor = np.sqrt(np.prod(y_train_shape[1:]))
    loss_array_scaled = np.array(loss_array) / norm_factor

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(epochs, loss_array_scaled, linewidth=2, color='blue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (per-pixel scale)")
    ax.set_xticks(np.arange(0, len(loss_array) + 1, 5))
    plt.tight_layout()
    save_fig(fig, output_dir, "loss_curve.png")


# ==============================================================
# Metrics
# ==============================================================

def mae(pred, truth):
    return torch.mean(torch.abs(pred - truth)).item()


def rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth) ** 2)).item()


def bias(pred, truth):
    return torch.mean(pred - truth).item()


def grad_mae(pred, truth):
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = truth[:, :, :, 1:] - truth[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = truth[:, :, 1:, :] - truth[:, :, :-1, :]
    grad_error = torch.mean(torch.abs(dx_p - dx_t)) + torch.mean(torch.abs(dy_p - dy_t))
    return grad_error.item()


def pattern_corr(pred, truth):
    """
    Per-sample Pearson correlation between predicted and true spatial
    fields, averaged across samples. Captures spatial structure independent
    of magnitude bias.
    """
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = truth.reshape(B, -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p ** 2).sum(dim=1) * (t ** 2).sum(dim=1)) + 1e-8
    return (num / den).mean().item()


def spread_skill_ratio(pred_mean, pred_std, truth):
    error = torch.abs(pred_mean - truth)
    return (pred_std.mean() / error.mean()).item()


def _gaussian_window(window_size, sigma, device_):
    coords = torch.arange(window_size, dtype=torch.float32, device=device_) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(pred, truth, window_size=11, sigma=1.5, data_range=None):
    """
    Structural similarity index (Wang et al., 2004). Complements Grad MAE
    with a standard windowed structural/perceptual metric, widely reported
    alongside PSNR in the super-resolution / downscaling literature.
    """
    device_ = pred.device
    C = pred.shape[1]
    if data_range is None:
        data_range = (truth.max() - truth.min()).clamp(min=1e-6)

    window = _gaussian_window(window_size, sigma, device_)
    window = window.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=C)
    mu_t = F.conv2d(truth, window, padding=pad, groups=C)
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_p_sq
    sigma_t_sq = F.conv2d(truth * truth, window, padding=pad, groups=C) - mu_t_sq
    sigma_pt = F.conv2d(pred * truth, window, padding=pad, groups=C) - mu_pt

    k1, k2 = 0.01, 0.03
    C1, C2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean().item()


def ice_edge_error(pred, truth, threshold=0.0, mask_bool=None, lat=None):
    """
    Integrated Ice Edge Error (IIEE; Goessling et al., 2016), adapted for a
    thickness field: fraction of ocean-domain AREA where the predicted and
    true binary ice masks (SIT > threshold) disagree (0 = perfect edge
    placement).

    lat (H,), optional: latitudes of pred/truth's H axis (our target grid is
    regular lat/lon, per build_land_sea_mask's dst_hr). When given, cells are
    weighted by cos(lat) before summing -- the same physical-area-weighting
    convention used everywhere else in evaluation/run_daily_eval_batch.py
    (coslat_weighted_pattern_corr, the domain-mean bias/trend/anomaly-corr
    helpers) -- rather than a plain per-pixel count, which silently assumes
    every cell has equal physical area. Without lat, falls back to the old
    equal-area pixel-count fraction.

    mask_bool (N, H, W), optional: restrict to these cells (typically "not
    land"). IIEE is conceptually an ocean-only diagnostic in the literature.
    Without this, land gets included too, and de-normalizing Y_test_phys
    (`* Y_std + Y_mean`) doesn't round-trip land's normalized ~0 back to an
    exact float32 0.0 -- that leftover ~1e-6-scale noise is still ">
    threshold", so it reads as "ice truth present" over virtually all of
    land, while the model's hard-zeroed land predictions read as "ice
    absent". That mismatched practically every land cell and inflated IIEE
    by roughly the land fraction of the domain, even though nothing was
    physically wrong with either the truth or the prediction.
    """
    ice_pred = pred[:, 0] > threshold
    ice_truth = truth[:, 0] > threshold
    disagree = (ice_pred != ice_truth).float()  # (N, H, W)

    if lat is not None:
        lat_t = torch.as_tensor(lat, dtype=disagree.dtype, device=disagree.device)
        weight = torch.cos(torch.deg2rad(lat_t))[None, :, None].expand_as(disagree)
    else:
        weight = torch.ones_like(disagree)

    if mask_bool is not None:
        disagree = disagree[mask_bool]
        weight = weight[mask_bool]
    return (disagree * weight).sum().div(weight.sum()).item()


def masked_mae(pred, truth, mask_bool):
    """MAE over channel 0 (SIT), restricted to cells where mask_bool (N, H, W) is True."""
    sel = torch.abs(pred[:, 0] - truth[:, 0])[mask_bool]
    return sel.mean().item() if sel.numel() > 0 else float("nan")


def masked_rmse(pred, truth, mask_bool):
    """RMSE over channel 0 (SIT), restricted to cells where mask_bool (N, H, W) is True."""
    sel = ((pred[:, 0] - truth[:, 0]) ** 2)[mask_bool]
    return torch.sqrt(sel.mean()).item() if sel.numel() > 0 else float("nan")


def compute_metrics_table(Y_base_phys, Y_pred_det_phys, Y_pred_phys, Y_spread_phys, Y_test_phys,
                           mask_test=None, coastal_width=5, preds_all_phys=None, per_member=False, hlat=None):
    """
    Per-method metrics table (Bilinear / Deterministic UNet / Stochastic UNet Mean).

    If `mask_test` (N, C_mask, H, W) is given, also reports "Coastal MAE"/"Coastal RMSE" --
    the same MAE/RMSE but restricted to ocean cells within `coastal_width` pixels of land
    (per-sample, via `coastal_band_mask`, so this works whether every test sample shares
    one land mask [--no-patches] or each has its own [--patches, sliding-window tiles]).

    `per_member=True` (requires `preds_all_phys`, (N, K, C, H, W)): the "Stochastic UNet
    Mean" row's MAE/RMSE/Bias/Grad MAE/Pattern Corr/SSIM/IIEE/Coastal MAE/RMSE are computed
    as the mean, over the K individual saved ensemble members, of metric(member_k, truth) --
    instead of metric(ensemble_mean, truth) -- via `member_metrics.compute_member_avg_metrics`.
    Intended for MESACLIP runs, whose saved "truth" is itself a single ensemble-member
    realization rather than a deterministic physical target, so comparing it against the
    smoother ensemble mean understates real error (see member_metrics.py's module
    docstring). `run_pipeline` sets this automatically via `is_mesaclip_run(config.output_dir)`.
    Spread/Error is unaffected either way -- it's already single-truth-aware by design.
    """
    coastal_band = coastal_band_mask(mask_test[:, 0], coastal_width=coastal_width) if mask_test is not None else None
    ocean_bool = (mask_test[:, 0] <= 0.5) if mask_test is not None else None

    member_avg = None
    if per_member:
        if preds_all_phys is None:
            raise ValueError("per_member=True requires preds_all_phys")
        member_avg = compute_member_avg_metrics(preds_all_phys, Y_test_phys, mask_test, coastal_width, hlat=hlat)

    rows = []
    for label, pred in [
        ("Bilinear", Y_base_phys), ("Deterministic UNet", Y_pred_det_phys), ("Stochastic UNet Mean", Y_pred_phys),
    ]:
        if member_avg is not None and label == "Stochastic UNet Mean":
            row = dict(member_avg)
        else:
            row = {
                "MAE": mae(pred, Y_test_phys),
                "RMSE": rmse(pred, Y_test_phys),
                "Bias": bias(pred, Y_test_phys),
                "Grad MAE": grad_mae(pred, Y_test_phys),
                "Pattern Corr": pattern_corr(pred, Y_test_phys),
                "SSIM": ssim(pred, Y_test_phys),
                "IIEE": ice_edge_error(pred, Y_test_phys, mask_bool=ocean_bool, lat=hlat),
                "Coastal MAE": masked_mae(pred, Y_test_phys, coastal_band) if coastal_band is not None else np.nan,
                "Coastal RMSE": masked_rmse(pred, Y_test_phys, coastal_band) if coastal_band is not None else np.nan,
            }
        row["Method"] = label
        row["Spread/Error"] = spread_skill_ratio(Y_pred_phys, Y_spread_phys, Y_test_phys) if label == "Stochastic UNet Mean" else np.nan
        rows.append(row)

    cols = ["Method", "MAE", "RMSE", "Bias", "Grad MAE", "Pattern Corr", "SSIM", "IIEE",
            "Coastal MAE", "Coastal RMSE", "Spread/Error"]
    return pd.DataFrame(rows)[cols].round(4)


# ==============================================================
# SIT domain time series (no-patch mode)
# ==============================================================

def compute_domain_timeseries(Y_pred_phys, Y_pred_det_phys, Y_base_phys, Y_test_phys, land_mask, sample_times, output_dir):
    """
    Domain-mean (ocean-only) SIT time series for the no-patch case: one
    value per test sample, for the bilinear baseline, deterministic UNet,
    stochastic UNet ensemble mean, and truth. Ocean cells are weighted by
    (1 - land_mask) so land pixels don't pull down the domain-mean
    thickness. Written to CSV and plotted for comparison against
    observations at the chosen domain.

    land_mask: (1, 1, H, W) tensor on the same grid as the *_phys tensors.
    sample_times: array of length Nsamples with one timestamp per sample.
    """
    ocean_weight = (1.0 - land_mask[0, 0]).clamp(0, 1)  # (H, W)
    weight_sum = ocean_weight.sum().clamp(min=1e-6)

    def domain_mean(field_phys):
        w = ocean_weight[None, None, :, :]
        return ((field_phys * w).sum(dim=(1, 2, 3)) / weight_sum).numpy()

    try:
        time_index = pd.to_datetime(sample_times)
    except Exception:
        # Fallback for calendar types pandas can't parse directly (e.g. some
        # cftime calendars) -- keep the raw string representation instead.
        time_index = [str(t) for t in sample_times]

    df = pd.DataFrame({
        "time": time_index,
        "bilinear": domain_mean(Y_base_phys),
        "deterministic_unet": domain_mean(Y_pred_det_phys),
        "stochastic_unet_mean": domain_mean(Y_pred_phys),
        "truth": domain_mean(Y_test_phys),
    }).sort_values("time").reset_index(drop=True)

    csv_path = os.path.join(output_dir, "sit_timeseries.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved SIT time series: {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["time"], df["truth"], label="Truth", color="black", linewidth=2)
    ax.plot(df["time"], df["stochastic_unet_mean"], label="Stochastic UNet Mean", color="tab:blue")
    ax.plot(df["time"], df["deterministic_unet"], label="Deterministic UNet", color="tab:orange")
    ax.plot(df["time"], df["bilinear"], label="Bilinear", color="tab:green", linestyle="--")
    ax.set_ylabel("Domain-mean SIT (m)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    save_fig(fig, output_dir, "sit_timeseries.png")

    return df


# ==============================================================
# Candidate-point (coastal community) time series
# ==============================================================

def nearest_grid_index(lat_grid, lon_grid, point_lat, point_lon_360, land_mask_hw=None):
    """
    Nearest-neighbor (iy, ix) index into a rectilinear (lat, lon) grid for a
    single point. lon_grid and point_lon_360 must both be in [0, 360)
    convention (matches hlon/llon as loaded from the .nc files).

    Uses a simple equirectangular approximation (great for picking the
    nearest of a handful of nearby grid cells at these latitudes; not meant
    for precise geodesic distance work). Returns (iy, ix, approx_dist_km).

    If `land_mask_hw` ((H, W); 1 = land, 0 = ocean) is given, land cells are
    excluded from the search so the nearest OCEAN cell is returned instead.
    Without this, a coastal point can snap to a land cell, which the data
    pipeline fills with a constant 0 (see process_scalar's `.fillna(0)` in
    build_X_Y_from_HR.ipynb) -- silently producing a flat "truth" time
    series with no error, just a `nearest_cell_is_land` flag nobody read.
    """
    lat_grid = np.asarray(lat_grid)
    lon_grid = np.asarray(lon_grid) % 360

    dlat_deg = lat_grid[:, None] - point_lat  # (H, 1)
    dlon_deg = (lon_grid[None, :] - point_lon_360 + 180) % 360 - 180  # (1, W), wrapped
    dlon_deg = dlon_deg * np.cos(np.deg2rad(point_lat))

    dist2 = dlat_deg ** 2 + dlon_deg ** 2  # (H, W), broadcasts fine

    if land_mask_hw is not None:
        land_mask_hw = np.asarray(land_mask_hw)
        ocean = land_mask_hw <= 0.5
        if not ocean.any():
            raise ValueError("land_mask_hw has no ocean cells -- can't find a nearest ocean grid point.")
        dist2 = np.where(ocean, dist2, np.inf)

    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
    approx_dist_km = float(np.sqrt(dist2[iy, ix])) * 111.0  # ~111 km / degree
    return int(iy), int(ix), approx_dist_km


def extract_candidate_point_timeseries(fields_phys, lat_grid, lon_grid, land_mask_hw, sample_times, points=None, channel=0, flat_std_tol=1e-6):
    """
    Nearest-grid-cell time series at named candidate points (e.g. Kivalina,
    Shishmaref, Kotzebue, Nome) for one or more physical-space fields, all
    assumed to share the same (lat_grid, lon_grid) target grid -- i.e.
    intended for the no-patch, single-sub-domain case (see the note in
    save_evaluation_data about why patch mode is skipped).

    The nearest-cell search excludes both (a) cells flagged land by
    `land_mask_hw` and (b) cells where the "truth" field is ~constant over
    time (std < flat_std_tol). (b) matters because bilinear regridding onto
    the high-res target grid bleeds the land-side fill value (0) into
    cells immediately adjacent to the coast -- these get flagged as ocean
    by `land_mask_hw` (their regridded ocean fraction is > 0.5) but still
    come out numerically flat, which a land-only exclusion doesn't catch.

    Args:
        fields_phys: dict of {method_name: (Nsamples, C, H, W) array/tensor},
            e.g. {"truth": Y_test_phys, "stochastic_unet_mean": Y_pred_phys, ...}.
            Must include a "truth" key so the flat-cell check has something
            to test variance against.
        lat_grid, lon_grid: 1D coord arrays matching the H, W dims above.
        land_mask_hw: (H, W) array, 1 = land, 0 = ocean, or None to skip the
            land/flat-cell exclusion entirely (nearest cell wins regardless).
        sample_times: length-Nsamples array with one timestamp per sample.
        points: dict of {name: {"lat":.., "lon":..}} (lon in [0, 360)).
            Defaults to CANDIDATE_POINTS.
        channel: channel index into each field's C axis (0 = SIT).
        flat_std_tol: truth cells with temporal std below this (in the same
            physical units as the field, e.g. meters of SIT) are treated as
            invalid, same as land.

    Returns:
        (df, locations) where df is a long-format DataFrame with columns
        [point, method, time, value, grid_iy, grid_ix, dist_km,
        nearest_cell_is_land], and locations is {point: (iy, ix, dist_km)}.
    """
    if points is None:
        points = CANDIDATE_POINTS

    try:
        time_index = pd.to_datetime(sample_times)
    except Exception:
        time_index = [str(t) for t in sample_times]

    exclude_hw = None
    if land_mask_hw is not None:
        exclude_hw = np.asarray(land_mask_hw) > 0.5
        truth_field = fields_phys.get("truth")
        if truth_field is not None:
            truth_np = truth_field.numpy() if hasattr(truth_field, "numpy") else np.asarray(truth_field)
            flat_hw = truth_np[:, channel].std(axis=0) < flat_std_tol
            exclude_hw = exclude_hw | flat_hw

    locations = {}
    rows = []
    for point_name, pt in points.items():
        iy, ix, dist_km = nearest_grid_index(lat_grid, lon_grid, pt["lat"], pt["lon"], land_mask_hw=exclude_hw)
        locations[point_name] = (iy, ix, dist_km)
        is_land = bool(land_mask_hw[iy, ix] > 0.5) if land_mask_hw is not None else None

        for method_name, field in fields_phys.items():
            field_np = field.numpy() if hasattr(field, "numpy") else np.asarray(field)
            values = field_np[:, channel, iy, ix]
            for t, v in zip(time_index, values):
                rows.append({
                    "point": point_name, "method": method_name, "time": t, "value": float(v),
                    "grid_iy": iy, "grid_ix": ix, "dist_km": dist_km, "nearest_cell_is_land": is_land,
                })

    df = pd.DataFrame(rows).sort_values(["point", "method", "time"]).reset_index(drop=True)
    return df, locations


# ==============================================================
# Save evaluation data (for later, customizable notebook plotting)
# ==============================================================

def save_evaluation_data(output_dir, X_test_sit_phys, Y_base_phys, Y_pred_det_phys, preds_all_phys,
                          Y_pred_phys, Y_test_phys, mask_test, test_tile_ids, tile_geometry,
                          time_test, land_mask, hlat, hlon, llat, llon, bbox, use_patches,
                          candidate_points=None):
    """
    Dump everything needed to rebuild/customize the time-series, ensemble,
    and error figures later in a notebook, without re-running the model:
    the physical-space prediction/truth tensors, per-sample tile geometry
    (lon/lat of each tile's context+target windows), the full-domain land
    mask and coordinate grids, and per-sample timestamps. Also computes and
    saves candidate-point (coastal community) time series when running in
    no-patch mode.

    Writes to <output_dir>/eval_data/:
        fields.npz                      - all gridded tensors, as float32 numpy arrays
        tile_geometry.pkl               - list of dicts of lon/lat arrays, one per tile
        meta.json                       - bbox, use_patches, candidate point definitions
        sample_times.csv                - one row per test sample: sample_idx, time, tile_id
        candidate_point_timeseries.csv  - long-format point time series (no-patch mode only)

    Returns the eval_data directory path.
    """
    eval_dir = os.path.join(output_dir, "eval_data")
    os.makedirs(eval_dir, exist_ok=True)

    def to_np(t):
        return t.numpy() if hasattr(t, "numpy") else np.asarray(t)

    np.savez_compressed(
        os.path.join(eval_dir, "fields.npz"),
        X_test_sit_phys=to_np(X_test_sit_phys).astype(np.float32),
        Y_base_phys=to_np(Y_base_phys).astype(np.float32),
        Y_pred_det_phys=to_np(Y_pred_det_phys).astype(np.float32),
        preds_all_phys=to_np(preds_all_phys).astype(np.float32),
        Y_pred_phys=to_np(Y_pred_phys).astype(np.float32),
        Y_test_phys=to_np(Y_test_phys).astype(np.float32),
        mask_test=to_np(mask_test).astype(np.float32),
        test_tile_ids=to_np(test_tile_ids).astype(np.int64),
        land_mask=to_np(land_mask).astype(np.float32),
        hlat=np.asarray(hlat), hlon=np.asarray(hlon),
        llat=np.asarray(llat), llon=np.asarray(llon),
    )

    with open(os.path.join(eval_dir, "tile_geometry.pkl"), "wb") as f:
        pickle.dump(tile_geometry, f)

    try:
        time_index = pd.to_datetime(time_test)
    except Exception:
        time_index = [str(t) for t in time_test]
    pd.DataFrame({
        "sample_idx": np.arange(len(time_test)),
        "time": time_index,
        "tile_id": to_np(test_tile_ids),
    }).to_csv(os.path.join(eval_dir, "sample_times.csv"), index=False)

    points = candidate_points or CANDIDATE_POINTS
    with open(os.path.join(eval_dir, "meta.json"), "w") as f:
        json.dump({
            "bbox": bbox,
            "use_patches": use_patches,
            "n_samples": int(to_np(Y_test_phys).shape[0]),
            "candidate_points": points,
        }, f, indent=2)

    if not use_patches:
        # A single coherent sub-domain grid makes point extraction
        # meaningful (same reasoning as compute_domain_timeseries); with
        # tiled patches, a given point may not fall inside any individual
        # test tile, and tiles can overlap, so we skip it there.
        fields_phys = {
            "truth": Y_test_phys, "bilinear": Y_base_phys,
            "deterministic_unet": Y_pred_det_phys, "stochastic_unet_mean": Y_pred_phys,
        }
        geo = tile_geometry[0]
        point_df, locations = extract_candidate_point_timeseries(
            fields_phys, geo["target_lat"], geo["target_lon"], to_np(mask_test)[0, 0],
            time_test, points=points,
        )
        point_df.to_csv(os.path.join(eval_dir, "candidate_point_timeseries.csv"), index=False)
        print(f"Saved candidate-point time series for: {list(locations.keys())}")
        for name, (iy, ix, dist_km) in locations.items():
            print(f"  {name}: nearest grid cell ({iy}, {ix}), ~{dist_km:.1f} km away")
    else:
        print("use_patches=True: skipping candidate-point time series "
              "(no single coherent target grid to search across all test tiles).")

    print(f"Saved evaluation data for notebook plotting to: {eval_dir}")
    return eval_dir


# ==============================================================
# Ensemble / error figures
# ==============================================================

def plot_ensemble_figure(output_dir, X_test_sit_phys, Y_base_phys, Y_pred_det_phys, preds_all_phys,
                          Y_pred_phys, Y_test_phys, test_tile_ids, tile_geometry, proj, boundary_path,
                          bbox, idxs, mem_idx=4, member_avg_error=False):
    """
    member_avg_error=True adds a 7th "Mean Member Error" panel -- the
    average, over the K saved ensemble members, of |member_k - truth| --
    on its own colorbar (different units/scale than the six SIT-field
    panels). Intended for MESACLIP runs, where the "One Member"/"Ensemble
    Mean" panels alone don't show how far a typical individual member
    actually sits from the single-realization truth -- see
    member_metrics.py's module docstring for why that comparison matters.
    """
    num_samples = len(idxs)
    mem_idx = min(mem_idx, preds_all_phys.shape[1] - 1)
    panel_titles = ["Low-Res Input", "Bilinear", "Deterministic", "One Member", "Ensemble Mean", "High-Res Truth"]
    n_field_cols = len(panel_titles)
    n_cols = n_field_cols + (1 if member_avg_error else 0)
    if member_avg_error:
        panel_titles = panel_titles + ["Mean Member Error"]

    fig, axs = plt.subplots(
        num_samples, n_cols, figsize=(3 * n_cols, 3.3 * num_samples), constrained_layout=True, dpi=300,
        subplot_kw={"projection": proj},
    )
    if num_samples == 1:
        axs = axs[None, :]

    im_err = None
    for row, idx in enumerate(idxs):
        geo = tile_geometry[test_tile_ids[idx].item()]
        ctx_lon, ctx_lat = geo["context_lon"], geo["context_lat"]
        tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]

        lowres = X_test_sit_phys[idx, 0]
        bilinear = Y_base_phys[idx, 0]
        deterministic = Y_pred_det_phys[idx, 0]
        ens_member = preds_all_phys[idx, mem_idx, 0]
        ens_mean = Y_pred_phys[idx, 0]
        truth = Y_test_phys[idx, 0]

        fields = [lowres, bilinear, deterministic, ens_member, ens_mean, truth]
        lons = [ctx_lon, tgt_lon, tgt_lon, tgt_lon, tgt_lon, tgt_lon]
        lats = [ctx_lat, tgt_lat, tgt_lat, tgt_lat, tgt_lat, tgt_lat]

        for col, (field, lon_, lat_) in enumerate(zip(fields, lons, lats)):
            ax = axs[row, col]
            im = ax.pcolormesh(lon_, lat_, field, transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=3, shading="auto")
            style_polar_ax(ax, proj, boundary_path, bbox, lon_, lat_)
            if row == 0:
                ax.set_title(panel_titles[col], fontsize=14)

        if member_avg_error:
            member_err = (preds_all_phys[idx, :, 0] - truth).abs().mean(dim=0)
            ax_err = axs[row, n_field_cols]
            im_err = ax_err.pcolormesh(
                tgt_lon, tgt_lat, member_err, transform=ccrs.PlateCarree(), cmap="viridis", vmin=0, vmax=2, shading="auto",
            )
            style_polar_ax(ax_err, proj, boundary_path, bbox, tgt_lon, tgt_lat)
            if row == 0:
                ax_err.set_title(panel_titles[n_field_cols], fontsize=14)

        axs[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=14)

    cbar = fig.colorbar(im, ax=axs[:, :n_field_cols], aspect=30, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Sea ice thickness (m)", fontsize=16)
    if member_avg_error and im_err is not None:
        cbar2 = fig.colorbar(im_err, ax=axs[:, n_field_cols], aspect=30, shrink=0.8, pad=0.02)
        cbar2.ax.tick_params(labelsize=12)
        cbar2.set_label("Mean |member - truth| (m)", fontsize=14)

    save_fig(fig, output_dir, "ensemble_figure.png", dpi=300)


def plot_error_figure(output_dir, Y_base_phys, Y_pred_det_phys, Y_pred_phys, Y_test_phys,
                       test_tile_ids, tile_geometry, proj, boundary_path, bbox, idxs,
                       preds_all_phys=None, member_avg_error=False):
    """
    member_avg_error=True (requires preds_all_phys, (N, K, C, H, W)) adds a
    4th "Mean Member Error" panel -- the average, over the K saved ensemble
    members, of |member_k - truth| -- on the same color scale as the other
    three error panels (no extra colorbar needed). Intended for MESACLIP
    runs, where the "Ensemble Mean" error panel alone understates real
    error the same way the metrics table's "Stochastic UNet Mean" row did
    before member_metrics.py -- see that module's docstring.
    """
    num_samples = len(idxs)
    panel_titles_err = ["Bilinear", "Deterministic", "Ensemble Mean"]
    if member_avg_error:
        panel_titles_err = panel_titles_err + ["Mean Member Error"]
    n_cols = len(panel_titles_err)

    fig, axs = plt.subplots(
        num_samples, n_cols, figsize=(3 * n_cols, 2.7 * num_samples), constrained_layout=True, dpi=300,
        subplot_kw={"projection": proj},
    )
    if num_samples == 1:
        axs = axs[None, :]

    for row, idx in enumerate(idxs):
        geo = tile_geometry[test_tile_ids[idx].item()]
        tgt_lon, tgt_lat = geo["target_lon"], geo["target_lat"]

        bilinear = Y_base_phys[idx, 0].numpy()
        det = Y_pred_det_phys[idx, 0].numpy()
        ens = Y_pred_phys[idx, 0].numpy()
        truth = Y_test_phys[idx, 0].numpy()

        bilinear_ae = abs(bilinear - truth)
        det_ae = abs(det - truth)
        ens_ae = abs(ens - truth)
        fields = [bilinear_ae, det_ae, ens_ae]
        if member_avg_error:
            member_ae = np.abs(preds_all_phys[idx, :, 0].numpy() - truth).mean(axis=0)
            fields.append(member_ae)

        for col, field in enumerate(fields):
            ax = axs[row, col]
            im = ax.pcolormesh(tgt_lon, tgt_lat, field, transform=ccrs.PlateCarree(), cmap="viridis", vmin=0, vmax=2, shading="auto")
            style_polar_ax(ax, proj, boundary_path, bbox, tgt_lon, tgt_lat)
            if row == 0:
                ax.set_title(panel_titles_err[col], fontsize=14)

        axs[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=14)

    cbar = fig.colorbar(im, ax=axs, aspect=20, shrink=0.9, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("|Y - Truth| Absolute Error (m)", fontsize=14)

    save_fig(fig, output_dir, "error_figure.png", dpi=300)


# ==============================================================
# Full pipeline
# ==============================================================

def run_pipeline(config):
    """
    End-to-end EngressNet pipeline: load data, build the land-sea mask,
    split into train/test (by year or randomly), extract patches (or crop
    to a single sub-domain), train, evaluate, compute metrics, and
    (optionally) write figures + a domain SIT time series.

    `config` is any object with attribute access (argparse.Namespace is
    fine) providing: x_path, y_path, output_dir, weighted_grids_dir, bbox,
    bbox_regrid, use_patches, subdomain, context_size, target_size, stride,
    train_years, test_years, train_frac, months, k, num_epochs, batch_size, lr,
    k_eval, eval_batch_size, make_figures, seed. Optionally
    test_x_path/test_y_path: when both are set, the model trains on x_path/
    y_path (restricted to train_years) but evaluates on a *different*
    dataset (test_x_path/test_y_path, restricted to test_years) -- e.g.
    train on FOSI, test on MESA-HR. Requires train_years and test_years both
    set, and the two datasets to share a grid.

    Returns a dict with the key results (metrics_df, loss_array, etc.) so
    callers like hpo_engressnet.py can pull out a metric without re-reading
    files from disk.
    """
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)
    save_run_config(config)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Output directory:", config.output_dir)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    print("Loading data...")

    X_ds = xr.open_dataset(config.x_path)
    Y_ds = xr.open_dataset(config.y_path)
    X_da, Y_da = X_ds.X, Y_ds.Y

    if "time" not in X_da.dims:
        raise ValueError(f"Expected a 'time' dimension on X, found dims {X_da.dims}.")
    if X_da.dims.index("time") != 1:
        raise ValueError(
            f"run_pipeline() assumes dims (N, time, C, H, W) but X has dims {X_da.dims}. "
            "Update the reshape logic in run_pipeline() if your dimension order differs."
        )

    X_time = X_da["time"]

    llat, llon = X_da.lat.values, X_da.lon.values
    hlat, hlon = Y_da.lat.values, Y_da.lon.values

    X = X_da.values
    Y = Y_da.values

    # Opt-in schema harmonization (pretrain-on-MESA/fine-tune-on-FOSI transfer learning): collapse
    # FOSI's vector wind (u_10, v_10) into a single wind-speed channel so this run's in_channels
    # matches a MESA-trained checkpoint's, and --init-checkpoint's strict state_dict load doesn't
    # hit a first-conv-layer channel-count mismatch. No-op if X_da has no channel coord or doesn't
    # carry both u_10/v_10 (e.g. it's already MESA). Applied here (before any split) so it's
    # already in effect on both sides of the cross-dataset-eval channel-harmonization block below.
    if getattr(config, "collapse_wind_vector", False) and "channel" in X_da.coords:
        X, _ = collapse_wind_vector_channel(X, list(X_da["channel"].values))

    # Clip SIT max thickness (channel 0; remove spurious CESM1.3 artifacts)
    X[:, :, 0, :, :] = np.clip(X[:, :, 0, :, :], None, 6.0)
    Y = np.clip(Y, None, 6.0)

    # ------------------------------------------------------------
    # Land-sea mask
    # ------------------------------------------------------------
    print("Building land-sea mask...")
    land_mask = build_land_sea_mask(hlat, hlon, config.bbox, config.bbox_regrid, config.weighted_grids_dir)
    print(X.shape, Y.shape, land_mask.shape)

    proj, boundary_path, central_lon = make_polar_proj(config.bbox)

    if config.make_figures:
        print("Rendering domain diagnostic figure...")
        plot_domain_diagnostic(config.output_dir, config.bbox, llat, llon, hlat, hlon, land_mask, X, Y, proj, boundary_path, central_lon)

    # ------------------------------------------------------------
    # Sub-domain validation (no-patch mode)
    # ------------------------------------------------------------
    if not config.use_patches:
        if config.subdomain is None:
            raise ValueError("config.subdomain is required when config.use_patches is False.")
        validate_subdomain(config.subdomain, config.bbox)

    # ------------------------------------------------------------
    # Train/test split (by year, or random fallback) -- or, if
    # test_x_path/test_y_path are set, a cross-dataset split: train fields
    # come from X/Y (above) restricted to train_years, test fields come from
    # a *separately opened* dataset restricted to test_years (e.g. train on
    # FOSI, evaluate the resulting checkpoint on MESA-HR). This requires an
    # explicit train_years/test_years pair (no random-split fallback -- a
    # random split of two different datasets isn't a meaningful "test set")
    # and a matching target (Y) grid, since land_mask/subdomain cropping
    # below are computed once from the primary (train) dataset's grid and
    # reused for both sides.
    # ------------------------------------------------------------
    print("Splitting...")
    test_x_path = getattr(config, "test_x_path", None)
    test_y_path = getattr(config, "test_y_path", None)
    if test_x_path or test_y_path:
        if not (test_x_path and test_y_path):
            raise ValueError("test_x_path and test_y_path must be given together (or neither).")
        if config.train_years is None or config.test_years is None:
            raise ValueError(
                "Cross-dataset evaluation (test_x_path/test_y_path) requires both "
                "config.train_years and config.test_years -- there's no random-split "
                "fallback across two different datasets."
            )
        print(f"Cross-dataset eval: training on {config.x_path}, testing on {test_x_path}")

        X_test_ds = xr.open_dataset(test_x_path)
        Y_test_ds = xr.open_dataset(test_y_path)
        X_test_da, Y_test_da = X_test_ds.X, Y_test_ds.Y

        if "time" not in X_test_da.dims or X_test_da.dims.index("time") != 1:
            raise ValueError(
                f"test_x_path dataset must have dims (N, time, C, H, W); found {X_test_da.dims}."
            )

        llat2, llon2 = X_test_da.lat.values, X_test_da.lon.values
        hlat2, hlon2 = Y_test_da.lat.values, Y_test_da.lon.values
        if llat2.shape != llat.shape or llon2.shape != llon.shape or not (
            np.allclose(llat2, llat) and np.allclose(llon2, llon)
        ):
            raise ValueError(
                "test_x_path's low-res grid doesn't match x_path's -- cross-dataset eval "
                "requires both datasets to share a grid (land_mask/subdomain cropping are "
                f"computed once from the training dataset). Shapes: train {llat.shape}, test {llat2.shape}."
            )
        if hlat2.shape != hlat.shape or hlon2.shape != hlon.shape or not (
            np.allclose(hlat2, hlat) and np.allclose(hlon2, hlon)
        ):
            raise ValueError(
                "test_y_path's high-res grid doesn't match y_path's -- cross-dataset eval "
                "requires both datasets to share a grid. Shapes: "
                f"train {hlat.shape}, test {hlat2.shape}."
            )

        X2 = X_test_da.values
        Y2 = Y_test_da.values
        X2[:, :, 0, :, :] = np.clip(X2[:, :, 0, :, :], None, 6.0)
        Y2 = np.clip(Y2, None, 6.0)

        if getattr(config, "collapse_wind_vector", False) and "channel" in X_test_da.coords:
            # X (train) was already collapsed above, before this cross-dataset block, using
            # X_da's original channel coord -- collapse X2 (test) the same way here, using
            # X_test_da's own original coord, so both sides land on the same 3-channel schema
            # a checkpoint pretrained with --collapse-wind-vector expects. Skips the
            # mismatch-triggered harmonization below entirely: X_da/X_test_da's channel
            # coords are now stale relative to the already-collapsed X/X2 arrays, so
            # re-deriving from them would desync names from actual channel count.
            X2, _ = collapse_wind_vector_channel(X2, list(X_test_da["channel"].values))
        elif "channel" in X_da.coords and "channel" in X_test_da.coords:
            # Predictor-schema harmonization: FOSI's X carries full vector wind
            # (u_10, v_10) while MESA-HR's carries only wind speed (U10) -- a real
            # difference between the two datasets' predictor-construction pipelines,
            # discovered via a mismatched-tensor-shape crash the first time this
            # cross-dataset path was smoke-tested. Collapse whichever side has the
            # vector-wind pair down to a single derived speed channel so both sides
            # present the same channel count/order (see collapse_wind_vector_channel's
            # docstring) -- confirmed by the user as the preferred fix over dropping
            # wind entirely. Uses the *original* dataset channel coordinate, not
            # X.shape.
            train_channel_names = list(X_da["channel"].values)
            test_channel_names = list(X_test_da["channel"].values)
            if len(train_channel_names) != len(test_channel_names):
                print(f"Predictor channel mismatch: train={train_channel_names}, test={test_channel_names} "
                      "-- attempting to reconcile by collapsing vector wind (u_10, v_10) to wind speed.")
                X, train_channel_names = collapse_wind_vector_channel(X, train_channel_names)
                X2, test_channel_names = collapse_wind_vector_channel(X2, test_channel_names)
                print(f"After harmonization: train channels={train_channel_names}, test channels={test_channel_names}.")
        if X.shape[2] != X2.shape[2]:
            raise ValueError(
                f"Train X has {X.shape[2]} channels, test X has {X2.shape[2]} -- cross-dataset eval "
                "requires matching predictor channels, and the vector-wind-to-speed harmonization "
                "(collapse_wind_vector_channel) wasn't enough to reconcile them. Extend it or align "
                "the two datasets' predictor schemas manually."
            )

        X_train_fields, Y_train_fields, time_train, member_train = select_by_years(
            X, Y, X_time, config.train_years, months=getattr(config, "months", None),
        )
        X_test_fields, Y_test_fields, time_test, member_test = select_by_years(
            X2, Y2, X_test_da["time"], config.test_years, months=getattr(config, "months", None),
        )
    else:
        (X_train_fields, Y_train_fields, X_test_fields, Y_test_fields,
         time_train, time_test, member_train, member_test) = split_train_test(
            X, Y, X_time, config.train_years, config.test_years, config.train_frac, seed=config.seed,
            months=getattr(config, "months", None),
        )

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------
    X_mean = X_train_fields.mean(dim=(0, 2, 3), keepdim=True)
    X_std = X_train_fields.std(dim=(0, 2, 3), keepdim=True)
    Y_mean = Y_train_fields.mean(dim=(0, 2, 3), keepdim=True)
    Y_std = Y_train_fields.std(dim=(0, 2, 3), keepdim=True)

    X_train = (X_train_fields - X_mean) / (X_std + 1e-6)
    X_test = (X_test_fields - X_mean) / (X_std + 1e-6)
    Y_train = (Y_train_fields - Y_mean) / (Y_std + 1e-6)
    Y_test = (Y_test_fields - Y_mean) / (Y_std + 1e-6)

    # ------------------------------------------------------------
    # Patches, or single full sub-domain
    # ------------------------------------------------------------
    if config.use_patches:
        print("Extracting patches...")
        X_train, Y_train, mask_train, _, _ = extract_patches(
            X_train, Y_train, land_mask, config.context_size, config.target_size, config.stride,
        )
        X_test, Y_test, mask_test, test_tile_ids, tile_geometry = extract_patches(
            X_test, Y_test, land_mask, config.context_size, config.target_size, config.stride,
            llon=llon, llat=llat, hlon=hlon, hlat=hlat,
        )
    else:
        print("Cropping to sub-domain (no patches)...")
        X_train, Y_train, mask_train, _, _ = extract_full_domain(
            X_train, Y_train, land_mask, llon, llat, hlon, hlat, config.subdomain,
        )
        X_test, Y_test, mask_test, test_tile_ids, tile_geometry = extract_full_domain(
            X_test, Y_test, land_mask, llon, llat, hlon, hlat, config.subdomain,
        )

    print("X_train:   ", X_train.shape)
    print("Y_train:   ", Y_train.shape)
    print("mask_train:", mask_train.shape)
    print("X_test:    ", X_test.shape)
    print("Y_test:    ", Y_test.shape)
    print("mask_test: ", mask_test.shape)

    # ------------------------------------------------------------
    # Re-express the training target as a standardized *physical increment*
    # (Y_phys - baseline_phys, normalized by the increment's own train-set mean/std)
    # instead of the standardized raw value -- matches Brajard et al. 2026 (EGUsphere,
    # Arctic SIT super-resolution via conditional diffusion)'s Eq. 4 / NVIDIA CorrDiff's
    # residual construction. See UNet.forward()'s docstring note for the matching model
    # side of this (no internal bilinear-base residual add -- `out` already *is* the
    # predicted standardized increment). Computed here (post-crop) rather than pre-crop
    # so the bilinear baseline is built from the same cropped tile geometry as
    # Y_train/Y_test. X_test_sit_phys/Y_base_phys (the "Bilinear" baseline used
    # throughout the rest of this function) are exactly this same baseline, just named
    # for their other role -- computed once here, not duplicated later.
    # ------------------------------------------------------------
    print("Converting target to a standardized physical increment...")
    sit_idx = 0
    # De-normalize with the exact same (std + 1e-6) denominator used to normalize above
    # (X_train/X_test lines 2170-2171) -- using bare X_std here would not invert that
    # division exactly, leaving a small nonzero constant offset for pixels whose true
    # physical value is 0.0 (e.g. open water), which then reads as "ice present" under
    # ice_edge_error's strict `> 0.0` threshold. See matching Y_std/inc_std fixes below.
    X_train_sit_phys = X_train[:, sit_idx:sit_idx + 1] * (X_std[:, sit_idx:sit_idx + 1] + 1e-6) + X_mean[:, sit_idx:sit_idx + 1]
    X_test_sit_phys = X_test[:, sit_idx:sit_idx + 1] * (X_std[:, sit_idx:sit_idx + 1] + 1e-6) + X_mean[:, sit_idx:sit_idx + 1]
    baseline_train_phys = F.interpolate(X_train_sit_phys, size=Y_train.shape[-2:], mode="bilinear", align_corners=False)
    Y_base_phys = F.interpolate(X_test_sit_phys, size=Y_test.shape[-2:], mode="bilinear", align_corners=False).cpu()
    baseline_test_phys = Y_base_phys.to(Y_test.device)

    Y_train_phys = Y_train * (Y_std + 1e-6) + Y_mean
    Y_test_phys_raw = Y_test * (Y_std + 1e-6) + Y_mean

    inc_train_phys = Y_train_phys - baseline_train_phys
    inc_test_phys = Y_test_phys_raw - baseline_test_phys

    inc_mean = inc_train_phys.mean(dim=(0, 2, 3), keepdim=True)
    inc_std = inc_train_phys.std(dim=(0, 2, 3), keepdim=True)

    Y_train = (inc_train_phys - inc_mean) / (inc_std + 1e-6)
    Y_test = (inc_test_phys - inc_mean) / (inc_std + 1e-6)
    print(f"  Increment stats (train, physical units): mean={float(inc_mean.flatten()[0]):.4f}, "
          f"std={float(inc_std.flatten()[0]):.4f}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = UNet(
        in_channels=X_train.shape[1], mask_channels=mask_train.shape[1],
        attention_end=getattr(config, "attention_end", False),
        attn_window_size=getattr(config, "attn_window_size", 8),
        attn_num_heads=getattr(config, "attn_num_heads", 4),
        noise_channels=getattr(config, "noise_channels", 1),
        noise_kernel_size=getattr(config, "noise_kernel_size", 5),
        late_mask_fusion=getattr(config, "late_mask_fusion", False),
    )
    model = nn.DataParallel(model).to(device)

    init_checkpoint = getattr(config, "init_checkpoint", None)
    if init_checkpoint:
        print(f"Loading pretrained weights from {init_checkpoint} ...")
        state_dict = torch.load(init_checkpoint, map_location=device)
        model.module.load_state_dict(state_dict)  # strict: architecture (incl. in_channels) must match exactly
        print("Pretrained weights loaded.")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    print("Model setup complete.")

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    print("Starting training...")
    loss_array = train_model(
        model, optimizer, X_train, Y_train, mask_train, device, config.k, config.num_epochs, config.batch_size,
        coastal_width=config.coastal_width, coastal_boost=config.coastal_boost, beta=config.beta,
    )

    torch.save(model.module.state_dict(), os.path.join(config.output_dir, "model_state_dict.pt"))
    np.save(os.path.join(config.output_dir, "loss_array.npy"), np.array(loss_array))
    print("Saved model checkpoint and loss history.")

    if config.make_figures:
        plot_loss_curve(config.output_dir, loss_array, Y_train.shape)

    # ------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------
    print("Running evaluation...")
    preds_all, Y_pred, Y_spread, Y_pred_det = evaluate_model(model, X_test, Y_test, mask_test, device, config.k_eval, config.eval_batch_size)

    # De-normalize using the *increment's* own stats (not Y_mean/Y_std, which describe the
    # raw value's distribution, not the increment's), then add back the physical bilinear
    # baseline computed earlier -- mirrors Brajard et al.'s Eq. 4 construction inverted for
    # inference. baseline_test_phys already matches Y_test's (N,C,H,W) shape/crop;
    # preds_all needs it broadcast over the extra K-member axis.
    # Multiply by (inc_std + 1e-6), matching the (inc_std + 1e-6) divisor used to
    # normalize the increment above (lines 2233-2234) -- bare inc_std here would leave
    # the same kind of nonzero-floor artifact as the X_std/Y_std fix above, this time on
    # the reconstructed full field rather than the raw channel.
    Y_test_phys = Y_test * (inc_std + 1e-6) + inc_mean + baseline_test_phys
    Y_pred_phys = (Y_pred * (inc_std + 1e-6) + inc_mean + baseline_test_phys).clamp(min=0.0)
    Y_spread_phys = Y_spread * (inc_std + 1e-6)
    Y_pred_det_phys = (Y_pred_det * (inc_std + 1e-6) + inc_mean + baseline_test_phys).clamp(min=0.0)
    preds_all_phys = (preds_all * (inc_std + 1e-6) + inc_mean + baseline_test_phys.unsqueeze(1)).clamp(min=0.0)

    # Hard-zero land in the model's own physical-space predictions (not the bilinear
    # baseline or truth). This can't be done inside UNet.forward(): the model operates in
    # normalized (z-scored increment) space there, where 0 isn't physical zero SIT, it's
    # whatever the training-set increment mean maps to after this de-normalization+baseline
    # add. Doing it here guarantees literal zero ice over land in every metric and plot,
    # instead of relying on the coastal loss weight to merely encourage `out` to cancel
    # non-zero bleed near the coast.
    ocean_test = (1.0 - mask_test.to(Y_pred_phys.dtype)).clamp(0.0, 1.0)
    Y_pred_phys = Y_pred_phys * ocean_test
    Y_pred_det_phys = Y_pred_det_phys * ocean_test
    preds_all_phys = preds_all_phys * ocean_test.unsqueeze(1)

    # X_test_sit_phys/Y_base_phys (the "Bilinear" baseline used throughout the rest of this
    # function) were already computed above, as part of building the increment target --
    # nothing further to do here.

    print("Inference complete.")

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    # Cross-dataset eval (test_y_path set): the per-member-averaged-metrics
    # correction should key off which dataset supplies the *test* truth
    # (Y_test_phys), not the training data or the output-dir naming
    # convention -- a "<train>_train_<test>_test"-style cross-run's folder
    # name doesn't map cleanly onto is_mesaclip_run's MESA_-prefix heuristic
    # either way, and using it would get the direction backwards half the
    # time. Same-dataset runs are unaffected -- unchanged output-dir check.
    test_y_path = getattr(config, "test_y_path", None)
    if test_y_path:
        mesaclip_run = "MESA" in os.path.basename(test_y_path).upper()
    else:
        mesaclip_run = is_mesaclip_run(config.output_dir)
    # `hlat` here is still the full pre-crop grid Y_da.lat came from -- Y_test_phys etc. are
    # cropped (patches or sub-domain, see extract_patches/extract_full_domain), so IIEE's
    # cos(lat) weighting needs the matching cropped lat, which only tile_geometry carries.
    # With patches, different tiles (and therefore different N samples) can sit at different
    # latitudes, so a single (H,) lat array isn't valid across the whole batch -- same
    # limitation already noted elsewhere for domain-mean metrics (patches mode skips those
    # entirely). Only apply area weighting in the single-tile (no-patches) case.
    metrics_hlat = tile_geometry[0]["target_lat"] if len(tile_geometry) == 1 else None
    metrics_df = compute_metrics_table(
        Y_base_phys, Y_pred_det_phys, Y_pred_phys, Y_spread_phys, Y_test_phys,
        mask_test=mask_test, coastal_width=config.coastal_width,
        preds_all_phys=preds_all_phys if mesaclip_run else None, per_member=mesaclip_run,
        hlat=metrics_hlat,
    )
    if mesaclip_run:
        detected_via = "test_y_path" if test_y_path else "output dir starts with 'MESA_'"
        print(f"MESACLIP run detected (via {detected_via}): 'Stochastic UNet Mean' "
              "metrics are per-member-then-averaged against the single-realization truth, not "
              "ensemble-mean-vs-truth -- see member_metrics.py.")
    metrics_df.to_csv(os.path.join(config.output_dir, "metrics.csv"), index=False)
    print(metrics_df)

    result = {
        "output_dir": config.output_dir,
        "loss_array": loss_array,
        "metrics_df": metrics_df,
    }

    # ------------------------------------------------------------
    # SIT time series (no-patch mode only -- a single coherent domain
    # makes a domain-mean time series meaningful; with tiled patches each
    # test sample only covers a small, arbitrarily-placed window)
    # ------------------------------------------------------------
    if not config.use_patches:
        print("Computing domain SIT time series...")
        timeseries_df = compute_domain_timeseries(
            Y_pred_phys, Y_pred_det_phys, Y_base_phys, Y_test_phys, mask_test[0:1], time_test, config.output_dir,
        )
        result["timeseries_df"] = timeseries_df

    # ------------------------------------------------------------
    # Save raw evaluation data for later, customizable notebook plotting
    # (time series, ensemble figure, error figure, candidate points).
    # Independent of config.make_figures, since the point is to have the
    # data available even when the quick-look PNGs weren't rendered.
    # ------------------------------------------------------------
    if getattr(config, "save_eval_data", True):
        print("Saving evaluation data for notebook plotting...")
        eval_dir = save_evaluation_data(
            config.output_dir, X_test_sit_phys, Y_base_phys, Y_pred_det_phys, preds_all_phys,
            Y_pred_phys, Y_test_phys, mask_test, test_tile_ids, tile_geometry,
            time_test, land_mask, hlat, hlon, llat, llon, config.bbox, config.use_patches,
        )
        result["eval_data_dir"] = eval_dir

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------
    if config.make_figures:
        n_available = Y_test_phys.shape[0]
        num_samples = min(3, n_available)
        fig_idxs = np.random.choice(n_available, num_samples, replace=False)

        print("Rendering ensemble figure...")
        plot_ensemble_figure(
            config.output_dir, X_test_sit_phys, Y_base_phys, Y_pred_det_phys, preds_all_phys,
            Y_pred_phys, Y_test_phys, test_tile_ids, tile_geometry, proj, boundary_path, config.bbox, fig_idxs,
            member_avg_error=mesaclip_run,
        )

        print("Rendering error figure...")
        plot_error_figure(
            config.output_dir, Y_base_phys, Y_pred_det_phys, Y_pred_phys, Y_test_phys,
            test_tile_ids, tile_geometry, proj, boundary_path, config.bbox, fig_idxs,
            preds_all_phys=preds_all_phys, member_avg_error=mesaclip_run,
        )

    print("All done. Outputs written to:", config.output_dir)
    return result
