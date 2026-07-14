"""Per-file regridding + region selection.

The two worker functions here (``process_ice_file``, ``process_atm_file``)
are the direct replacements for the original notebook's ``process_file`` /
``process_file_hi``. The key structural change: **every regridder, mask,
and bbox the worker needs is passed in as an explicit argument**, rather
than read off module-level globals.

Why this matters: the original relied on ``ProcessPoolExecutor`` workers
inheriting globals like ``regridder_coarse_ice`` (which didn't even exist --
that was the crash) via fork-time memory inheritance. Even after fixing the
naming, that pattern is fragile: it silently depends on the multiprocessing
start method being "fork" (the Linux default, but not guaranteed in every
HPC/MPI-aware environment, and not the default on macOS/Windows). Passing
everything explicitly works under "fork" *and* "spawn", and is also what
lets these functions be unit-tested without constructing the full pipeline.

xESMF Regridder objects ARE picklable (they pickle their weight file path
and re-load on the other side), so passing one as a plain function argument
to ProcessPoolExecutor.map is safe.

A note on regridding method choice for sea ice variables: bilinear
interpolation is fine for `hi` (a continuous thickness field) but can
introduce small negative values or >1 artifacts for `aice` (concentration,
bounded [0,1]) at the ice edge / coastline, and doesn't strictly conserve
total ice area. If area conservation matters for your downstream metrics,
build the `aice` regridder with method="conservative" instead of
"bilinear" (requires cell corner coordinates, which pop_tools grids
provide as ULAT/ULONG -- not wired up here since the original script used
bilinear throughout, but flagging it since it's a common source of subtle
bias in regridded sea ice concentration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
import xesmf as xe

from .config import RegionBBox
from .grids import area_average_to_grid


def select_region(da: xr.DataArray, bbox: RegionBBox) -> xr.DataArray:
    """Subset a 2D-lat/lon-coordinate (or rectilinear lat/lon) DataArray to
    a region bbox, correctly handling the 0/360 longitude convention.

    This replaces the original notebook's
    ``da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))``
    after a bare ``lon % 360``, which silently produces an empty or wrong
    selection for any bbox that crosses the 0/360 seam. ``PipelineConfig``
    already rejects seam-crossing regions at config time (see config.py),
    so this function asserts that invariant rather than re-implementing
    wraparound selection logic that should never actually be needed --
    if you add a seam-crossing region in the future, extend this function
    (split into two slices and concat) rather than relaxing the config
    check.
    """
    if bbox.crosses_seam():
        raise ValueError(
            f"Region '{bbox.name}' crosses the 0/360 seam; select_region() "
            f"does not implement wraparound selection. Split the bbox into "
            f"two non-crossing pieces and concat the results instead."
        )
    lon_min, lon_max = bbox.as_0_360()

    if "lon" in da.dims and da["lon"].ndim == 1:
        # Rectilinear destination: simple slice works, lon coord is 1D.
        return da.sel(lat=slice(bbox.lat_min, bbox.lat_max), lon=slice(lon_min, lon_max))

    # Curvilinear/2D lat-lon destination (e.g. EASE-Grid 2.0 on y/x dims):
    # build a boolean mask instead of slicing, since `.sel` with `slice`
    # doesn't apply to non-dimension 2D coordinates.
    lat2d, lon2d = da["lat"], da["lon"] % 360
    mask = (
        (lat2d >= bbox.lat_min) & (lat2d <= bbox.lat_max) &
        (lon2d >= lon_min) & (lon2d <= lon_max)
    )
    if not bool(mask.any()):
        raise ValueError(
            f"Region '{bbox.name}' selected zero cells on this destination grid. "
            f"Check that the grid actually covers this bbox."
        )
    # Trim to the bounding box of the mask so we still get a compact 2D
    # array rather than a sparse, mostly-NaN full-grid array.
    y_idx, x_idx = np.where(mask.values)
    y_sl = slice(y_idx.min(), y_idx.max() + 1)
    x_sl = slice(x_idx.min(), x_idx.max() + 1)
    dims = da["lat"].dims  # e.g. ("y", "x")
    da_sub = da.isel({dims[0]: y_sl, dims[1]: x_sl})
    mask_sub = mask.isel({dims[0]: y_sl, dims[1]: x_sl})
    return da_sub.where(mask_sub)


@dataclass(frozen=True)
class IceFileJob:
    """Everything needed to process one ice-component history file.

    ``area_average_dst_grid``: if set, ``regridder`` is expected to be the
    HR-to-HR regridder (``ice_hr_to_hr``), and after regridding the result
    is block-averaged (cos-lat weighted, see ``grids.area_average_to_grid``)
    down onto this destination grid -- the "native area-mean pipeline".
    If ``None`` (default), ``regridder`` is applied as the sole regridding
    step (the "interpolated xESMF pipeline", e.g. ``ice_hr_to_lr``).
    """

    filepath: str
    varname: str
    lat_mask: np.ndarray  # from grids.IceSourceGrid, matches the file's native nlat
    regridder: xe.Regridder
    bbox: RegionBBox
    area_average_dst_grid: xr.Dataset | None = None


def process_ice_file(job: IceFileJob) -> xr.DataArray:
    """Open one ice-component file, subset to >=40N, regrid, optionally
    block-average down to a coarser destination (native area-mean
    pipeline), subset to the community bbox, and fill NaN with 0.

    Equivalent to the ``ice_vars`` branch of the original ``process_file``,
    but with the regridder and mask passed explicitly (see module
    docstring) and using the boolean ``lat_mask`` directly instead of the
    undefined ``jmin_high_ice`` contiguous-slice variable.

    Order matters here: area-averaging (if requested) happens BEFORE the
    bbox subset, not after. Subsetting to the bbox first and then binning
    would leave incomplete destination bins at the bbox edges (a coarse
    bin partially clipped by the bbox boundary gets averaged over fewer
    HR cells than its interior neighbors, silently biasing edge cells).
    Averaging over the full regridded HR domain first means every LR bin
    is computed from its complete footprint, and the bbox subset afterward
    just selects which of those complete bins to keep. fillna(0) happens
    last, after averaging, so genuinely-missing input cells are excluded
    from the weighted mean (via skipna) rather than counted as zero.
    """
    with xr.open_dataset(job.filepath) as ds:
        da = ds[job.varname].rename({"nj": "nlat", "ni": "nlon"})
        da = da.isel(nlat=job.lat_mask)
        da_reg = job.regridder(da)
        if job.area_average_dst_grid is not None:
            da_reg = area_average_to_grid(da_reg, job.area_average_dst_grid)
        da_reg = select_region(da_reg, job.bbox)
        da_reg = da_reg.fillna(0).astype(np.float32)
        da_reg = da_reg.load()  # materialize before the file closes
    return da_reg


@dataclass(frozen=True)
class AtmFileJob:
    """Counterpart of IceFileJob for atmosphere-component files. See
    IceFileJob's docstring for what ``area_average_dst_grid`` does.
    """

    filepath: str
    varname: str
    ncol_mask: np.ndarray
    regridder: xe.Regridder
    bbox: RegionBBox
    area_average_dst_grid: xr.Dataset | None = None


def process_atm_file(job: AtmFileJob) -> xr.DataArray:
    """Open one atm-component file, subset to >=40N, regrid, optionally
    block-average down to a coarser destination, subset to the community
    bbox, and fill NaN with 0.

    Equivalent to the ``atm_vars`` branch of the original ``process_file``.
    See ``process_ice_file`` for why area-averaging happens before, not
    after, the bbox subset.
    """
    with xr.open_dataset(job.filepath) as ds:
        da = ds[job.varname]
        if "lev" in da.dims:
            da = da.isel(lev=-1, drop=True)
        da = da.isel(ncol=job.ncol_mask)
        da_reg = job.regridder(da)
        if job.area_average_dst_grid is not None:
            da_reg = area_average_to_grid(da_reg, job.area_average_dst_grid)
        da_reg = select_region(da_reg, job.bbox)
        da_reg = da_reg.fillna(0).astype(np.float32)
        da_reg = da_reg.load()
    return da_reg
