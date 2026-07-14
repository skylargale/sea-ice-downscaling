"""Native and destination grid construction, and xESMF regridder factory.

* Source-grid row masks (``mask_high_ice`` etc.) are kept as **boolean
  masks** end-to-end, rather than being converted to a ``slice(jmin, None)``
  that assumes the >=40N rows are a single contiguous block. For a
  tripole/displaced-pole grid like POP's gx1v7/tx0.1v2, that assumption is
  not guaranteed -- a boolean mask is correct regardless.
* The *exact same* mask used to build a destination-facing grid descriptor
  (``build_ice_source_grid``) is reused when subsetting the actual data
  array in ``process_file`` (regrid.py), so the regridder's stored source
  shape always matches what gets fed to it at apply time. In the original
  script these two used different code paths (the static grid object vs.
  the opened dataset) which only worked if both `.isel` calls happened to
  produce identical row sets -- fragile and easy to silently break.
* Regridders are returned in a plain dict keyed by name and the names match
  what process_file actually calls -- the original defined
  ``grid["regrid"]["ice_hr_to_lr"]`` etc. but the worker functions referred
  to nonexistent variables like ``regridder_coarse_ice``. That mismatch is
  why the original would crash with NameError.
* Adds an EASE-Grid 2.0 Northern Hemisphere 25 km destination
  (``ease2_n25km``), built from the official NSIDC EASE2_N25km grid
  definition (a Lambert azimuthal equal-area projection), so `aice`/`hi`
  can be regridded onto it directly with xESMF, the same way you already
  regrid onto the 1deg/0p1deg rectilinear destinations. This requires
  ``pyproj`` (already a dependency of most xesmf/cartopy stacks).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pop_tools
import xarray as xr
import xesmf as xe


def area_average_to_grid(da: xr.DataArray, dst_grid: xr.Dataset) -> xr.DataArray:
    """
    Area-weighted (cos-latitude) block-average of an HR rectilinear array
    down onto a coarser rectilinear destination grid (e.g. 0p1deg -> 1deg).

    This is the "native area-mean pipeline" coarsening step: it runs AFTER
    ``da`` has already been regridded with xESMF onto a 1D-lat/lon
    rectilinear destination (the ``ice_hr_to_hr`` / ``atm_hr_to_hr``
    regridders), and bins that fine grid down onto ``dst_grid`` (built with
    ``lr_dest_grid``). It is NOT meant to run on the native curvilinear
    POP/SCRIP source coordinates -- those are still on irregular
    nlat/nlon/ncol dims and have no well-defined ``groupby_bins`` axis.

    ``dst_grid["lat"]`` / ``dst_grid["lon"]`` (as built by
    ``build_rectilinear_dest_grid``) are CELL CENTERS, evenly spaced --
    the same convention the interpolated xESMF pipeline regrids directly
    onto. Bin EDGES are derived here by offsetting each center by half the
    (uniform) grid step, so every nominal destination cell gets its own
    bin -- using the centers themselves as edges (as an earlier version of
    this function did) silently drops the last lat row and the last lon
    column, since N centers only define N-1 inter-center bins.

    Bins on BOTH lat and lon (an earlier version only binned lat, which
    silently averaged together every longitude at a given latitude band --
    fine for a zonal mean, wrong for a regional predictor field). Computes
    a true weighted mean (sum of weighted values / sum of weights per
    bin), not just `(da * weight).mean()`, which only rescales values by
    cos(lat) without normalizing back out.

    Requires ``da`` to have 1D ``lat``/``lon`` dimension coordinates (true
    for the rectilinear 1deg/0p1deg destinations) and a uniform step size
    matching ``dst_grid``'s. Will raise for a curvilinear destination (e.g.
    EASE-Grid 2.0) where lat/lon are 2D -- area-averaging onto a projected
    equal-area grid should instead sum cells within each destination
    cell's footprint, which isn't what this function does; don't call it
    in that case.
    """
    if da["lat"].ndim != 1 or da["lon"].ndim != 1:
        raise ValueError(
            "area_average_to_grid expects a 1D-rectilinear lat/lon source "
            "(e.g. the output of the *_hr_to_hr regridder onto '0p1deg'). "
            "Got lat.ndim={} lon.ndim={} -- this looks like a curvilinear "
            "destination (e.g. ease2_n25km), which this function does not "
            "support.".format(da["lat"].ndim, da["lon"].ndim)
        )

    def _centers_to_edges(centers: np.ndarray, *, label: str) -> np.ndarray:
        if centers.size < 2:
            raise ValueError(f"dst_grid['{label}'] needs >=2 points to infer a step size.")
        step = centers[1] - centers[0]
        if not np.allclose(np.diff(centers), step, rtol=1e-6):
            raise ValueError(
                f"dst_grid['{label}'] is not uniformly spaced; "
                f"area_average_to_grid assumes a regular rectilinear grid."
            )
        return np.concatenate([centers - step / 2.0, [centers[-1] + step / 2.0]])

    lat_centers = dst_grid["lat"].values
    lon_centers = dst_grid["lon"].values
    lat_edges = _centers_to_edges(lat_centers, label="lat")
    lon_edges = _centers_to_edges(lon_centers, label="lon")

    weights = np.cos(np.deg2rad(da["lat"])).broadcast_like(da)
    num = da * weights

    def _sum_lat_lon(arr: xr.DataArray) -> xr.DataArray:
        return (
            arr.groupby_bins("lat", lat_edges, labels=lat_centers)
            .sum(skipna=True, min_count=1)
            .groupby_bins("lon", lon_edges, labels=lon_centers)
            .sum(skipna=True, min_count=1)
        )

    num_binned = _sum_lat_lon(num)
    den_binned = _sum_lat_lon(weights)
    result = num_binned / den_binned
    result = result.rename({"lat_bins": "lat", "lon_bins": "lon"})
    return result


# ---------------------------------------------------------------------------
# EASE-Grid 2.0 Northern Hemisphere 25 km definition
# ---------------------------------------------------------------------------
# Official NSIDC EASE2_N25km parameters (Brodzik et al. 2012, 2014):
#   Projection: Lambert Azimuthal Equal-Area, centered on the north pole
#   Ellipsoid: WGS84
#   Grid: 720 x 720 cells, 25 km nominal resolution
#   map_x range: [-9000000, 9000000] m (cell centers), same for map_y
EASE2_N25KM_PROJ4 = "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
EASE2_N25KM_NCOLS = 720
EASE2_N25KM_NROWS = 720
EASE2_N25KM_RES_M = 25_025.26  # exact NSIDC cell size in meters
EASE2_N25KM_EXTENT_M = 9_000_000.0  # map_x/map_y extent from center to edge


def build_ease2_n25km_grid() -> xr.Dataset:
    """Build the EASE-Grid 2.0 NH 25km destination grid as 2D lat/lon.

    Returns a dataset with 2D ``lat``/``lon`` coordinates on dims
    ``("y", "x")`` -- the curvilinear form xESMF expects for a projected
    destination grid (this mirrors how you already build the POP ice grids
    as 2D ``nlat``/``nlon`` coordinate datasets).

    Requires `pyproj`. Cross-checks cell count/extent against the values
    NSIDC publishes for EASE2_N25km; if NSIDC ever revises the grid
    definition, this function will need updating accordingly -- it is not
    fetched dynamically.
    """
    try:
        import pyproj
    except ImportError as e:
        raise ImportError(
            "build_ease2_n25km_grid requires pyproj. Install with "
            "`conda install -c conda-forge pyproj` or `pip install pyproj`."
        ) from e

    half_res = EASE2_N25KM_RES_M / 2.0
    x = np.linspace(
        -EASE2_N25KM_EXTENT_M + half_res,
        EASE2_N25KM_EXTENT_M - half_res,
        EASE2_N25KM_NCOLS,
    )
    y = np.linspace(
        EASE2_N25KM_EXTENT_M - half_res,
        -EASE2_N25KM_EXTENT_M + half_res,
        EASE2_N25KM_NROWS,
    )
    xx, yy = np.meshgrid(x, y)

    transformer = pyproj.Transformer.from_crs(
        EASE2_N25KM_PROJ4, "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(xx, yy)

    ds = xr.Dataset(
        {
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon % 360),  # keep convention consistent with other dst grids
        },
        coords={
            "map_x": ("x", x),
            "map_y": ("y", y),
        },
        attrs={
            "grid_name": "EASE2_N25km",
            "projection": EASE2_N25KM_PROJ4,
            "description": "NSIDC EASE-Grid 2.0 Northern Hemisphere, 25 km nominal resolution",
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Native source grids
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceSourceGrid:
    """A native POP ice grid descriptor plus the mask used to build it.

    Keeping the mask alongside the grid descriptor (instead of discarding it
    once the regridder is built, as the original code did) is what lets
    ``regrid.process_ice_file`` apply the *identical* row selection to each
    opened data file before calling the regridder.
    """

    grid: xr.Dataset  # 2D lat/lon on dims (nlat, nlon), >=40N rows only
    lat_mask: np.ndarray  # boolean, shape (n_native_nlat,) -- True for kept rows


def build_ice_source_grid(pop_grid_name: str, lat_cutoff: float = 40.0) -> IceSourceGrid:
    """Build a >=lat_cutoff source-grid descriptor for a POP grid.

    Equivalent to the original notebook's `grid["ice"]["lr"|"hr"]`
    construction, but returns the boolean mask alongside the grid so the
    same mask can be reapplied to data files later (see regrid.py).
    """
    pop_grid = pop_tools.get_grid(pop_grid_name)
    lat_mask = np.any(pop_grid.TLAT.values >= lat_cutoff, axis=1)
    grid = xr.Dataset(
        {
            "lat": (("nlat", "nlon"), pop_grid.TLAT.values[lat_mask, :]),
            "lon": (("nlat", "nlon"), pop_grid.TLONG.values[lat_mask, :]),
        }
    )
    return IceSourceGrid(grid=grid, lat_mask=lat_mask)


@dataclass(frozen=True)
class AtmSourceGrid:
    grid: xr.Dataset  # 1D lat/lon on dim ncol, >=40N columns only
    ncol_mask: np.ndarray  # boolean, shape (n_native_ncol,)
    locstream: bool = True


def build_atm_source_grid(scrip_path: str, lat_cutoff: float = 40.0) -> AtmSourceGrid:
    """Build a >=lat_cutoff source-grid descriptor for a SCRIP atm grid."""
    scrip = xr.open_dataset(scrip_path)
    ncol_mask = scrip.grid_center_lat.values >= lat_cutoff
    grid = xr.Dataset(
        {
            "lat": ("ncol", scrip.grid_center_lat.values[ncol_mask]),
            "lon": ("ncol", scrip.grid_center_lon.values[ncol_mask]),
        }
    )
    scrip.close()
    return AtmSourceGrid(grid=grid, ncol_mask=ncol_mask)


# ---------------------------------------------------------------------------
# Destination grids
# ---------------------------------------------------------------------------
def build_rectilinear_dest_grid(
    lat_min: float, lat_max: float, lat_step: float, lon_step: float
) -> xr.Dataset:
    """Build a regular lat/lon destination grid, e.g. the original
    notebook's "1deg" (lat_step=1.0) / "0p1deg" (lat_step=0.1) grids.
    """
    return xr.Dataset(
        {
            "lat": ("lat", np.arange(lat_min, lat_max + lat_step / 2, lat_step)),
            "lon": ("lon", np.arange(0, 360, lon_step)),
        }
    )


DEST_GRID_BUILDERS = {
    "1deg": lambda: build_rectilinear_dest_grid(40, 90, 1.0, 1.0),
    "0p1deg": lambda: build_rectilinear_dest_grid(40, 90, 0.1, 0.1),
    "ease2_n25km": build_ease2_n25km_grid,
}


def build_dest_grid(name: str) -> xr.Dataset:
    try:
        return DEST_GRID_BUILDERS[name]()
    except KeyError as e:
        raise KeyError(
            f"Unknown destination grid '{name}'. Available: {list(DEST_GRID_BUILDERS)}"
        ) from e


# ---------------------------------------------------------------------------
# Regridder construction
# ---------------------------------------------------------------------------
def make_regridder(
    src_grid: xr.Dataset,
    dst_grid: xr.Dataset,
    *,
    method: Literal["bilinear", "conservative", "nearest_s2d", "patch"] = "bilinear",
    periodic: bool,
    weights_dir: str,
    weight_filename: str,
    reuse_weights: bool = True,
    locstream: bool = False,
) -> xe.Regridder:
    """Build (or load cached) an xESMF regridder, with weights written under
    a single managed directory instead of scattered into the working dir.

    ``reuse_weights=True`` by default -- the original script always
    rebuilt weights from scratch (``reuse_weights=False``) on every run,
    which is correct-but-slow for grids that never change between runs.
    Set to False explicitly the first time you change a grid definition,
    or just delete the cached weight file.
    """
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    weight_path = str(Path(weights_dir) / weight_filename)
    return xe.Regridder(
        src_grid,
        dst_grid,
        method=method,
        periodic=periodic,
        filename=weight_path,
        reuse_weights=reuse_weights and Path(weight_path).exists(),
        locstream_in=locstream
    )


@dataclass(frozen=True)
class GridBundle:
    """Everything regrid.py needs: source grids (with masks) + regridders.

    This is the single object that should be built once in the main process
    and passed explicitly into worker functions (see regrid.py) rather than
    relying on module-level globals that may or may not exist in a forked /
    spawned child process.

    ``dst_lr``/``dst_hr`` are the destination-grid Datasets themselves (not
    just the regridders built onto them) -- the native area-mean pipeline
    needs ``dst_lr`` directly, since ``area_average_to_grid`` bins onto its
    lat/lon bin edges rather than going through an xESMF regridder.
    """

    ice_hr: IceSourceGrid
    ice_lr: IceSourceGrid
    atm_hr: AtmSourceGrid
    atm_lr: AtmSourceGrid

    regridders: dict[str, xe.Regridder]
    dst_lr: xr.Dataset
    dst_hr: xr.Dataset


def build_grid_bundle(config) -> GridBundle:
    """
    Build native source grids and all regridders needed for one config.
    """
    gp = config.grid_paths

    # ----------------------------------------------------------
    # Source grids
    # ----------------------------------------------------------
    ice_hr = build_ice_source_grid(gp.pop_grid_hr)
    ice_lr = build_ice_source_grid(gp.pop_grid_lr)
    atm_hr = build_atm_source_grid(gp.atm_scrip_hr)
    atm_lr = build_atm_source_grid(gp.atm_scrip_lr)

    # ----------------------------------------------------------
    # Destination grids
    # ----------------------------------------------------------
    dst_lr = build_dest_grid(config.lr_dest_grid)   # coarse predictor grid (1deg)
    dst_hr = build_dest_grid(config.hr_dest_grid)   # common high-res grid (0.1deg)

    # `periodic=True` is only meaningful for a destination with a genuine
    # wraparound longitude dimension (the rectilinear 1deg/0p1deg grids,
    # which span the full 0-360 range). The EASE2 destination is a finite
    # Lambert-azimuthal projection with no periodic axis -- passing
    # periodic=True there is semantically wrong even if xESMF doesn't
    # immediately error on it.
    lr_periodic = config.lr_dest_grid in ("1deg", "0p1deg")
    hr_periodic = config.hr_dest_grid in ("1deg", "0p1deg")

    regridders: dict[str, xe.Regridder] = {}

    # ----------------------------------------------------------
    # 1. HR ice (tx0.1) -> LR common grid (1deg)
    # Used for interpolated (coarsened) hi, aice predictors
    # ----------------------------------------------------------
    regridders["ice_hr_to_lr"] = make_regridder(
        ice_hr.grid,
        dst_lr,
        method="bilinear",
        periodic=lr_periodic,
        weights_dir=config.weights_dir,
        weight_filename=f"ice_hr_to_{config.lr_dest_grid}.nc",
    )

    # ----------------------------------------------------------
    # 2. HR ice (tx0.1) -> HR common grid (0.1deg)
    # Used for subsequent area-average (coarsened) hi, aice predictors
    # Used for target SIT
    # ----------------------------------------------------------
    regridders["ice_hr_to_hr"] = make_regridder(
        ice_hr.grid,
        dst_hr,
        method="bilinear",
        periodic=hr_periodic,
        weights_dir=config.weights_dir,
        weight_filename=f"ice_hr_to_{config.hr_dest_grid}.nc",
    )

    # ----------------------------------------------------------
    # 3. HR atmosphere (ne120) -> LR common grid (1deg)
    # Used for interpolated (coarsened) U, V predictors
    # ----------------------------------------------------------
    regridders["atm_hr_to_lr"] = make_regridder(
        atm_hr.grid,
        dst_lr,
        method="nearest_s2d",
        periodic=False,
        locstream=True,
        weights_dir=config.weights_dir,
        weight_filename=f"atm_hr_to_{config.lr_dest_grid}.nc",
    )

    # ----------------------------------------------------------
    # 4. HR atmosphere (ne120) -> HR common grid (0.1deg)
    # Used for subsequent area-average (coarsened) U, V predictors
    # ----------------------------------------------------------
    regridders["atm_hr_to_hr"] = make_regridder(
        atm_hr.grid,
        dst_hr,
        method="nearest_s2d",
        periodic=False,
        locstream=True,
        weights_dir=config.weights_dir,
        weight_filename=f"atm_hr_to_{config.hr_dest_grid}.nc",
    )

    print("Regridders built (reused from cache where the weight file already existed):")
    for key, r in regridders.items():
        print(f"  {key}: {r.filename}")

    return GridBundle(
        ice_hr=ice_hr, ice_lr=ice_lr, atm_hr=atm_hr, atm_lr=atm_lr,
        regridders=regridders, dst_lr=dst_lr, dst_hr=dst_hr,
    )
