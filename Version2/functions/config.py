"""Central configuration for the MESACLIP perfect-model downscaling pipeline.

Everything that was a bare module-level global in the original notebook
(``low_vars``, ``comps``, ``bbox``, file glob roots, etc.) lives here as a
single dataclass-backed config object. Import ``DEFAULT_CONFIG`` for the
Cambridge Bay perfect-model setup, or build your own ``PipelineConfig`` for a
different community / variable set.

Nothing in this module touches disk or opens any dataset -- it's pure
configuration, which makes it safe to import inside multiprocessing workers
without re-triggering expensive I/O or regridder construction.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Component lookup: which CESM component writes each variable's history files
# ---------------------------------------------------------------------------
VAR_COMPONENT = {
    "hi": "ice",
    "aice": "ice",
    "U": "atm",
    "V": "atm",
}

ICE_VARS = ("hi", "aice")
ATM_VARS = ("U", "V")


@dataclass(frozen=True)
class RegionBBox:
    """Lon/lat bounding box for a coastal community subset.

    Longitudes are stored as given (can be negative); use ``as_0_360`` to
    get the wrapped form used when selecting against a 0-360 lon coordinate.
    Use ``crosses_seam`` to check whether a naive ``slice(lon_min, lon_max)``
    selection on a 0-360 coordinate would be safe.
    """

    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def as_0_360(self) -> tuple[float, float]:
        return self.lon_min % 360, self.lon_max % 360

    def crosses_seam(self) -> bool:
        """True if the box straddles the 0/360 (equivalently +/-180) seam.

        A box defined with lon_min < lon_max in -180/180 terms can still
        invert once wrapped to 0-360 (e.g. lon_min=170, lon_max=-170 wraps
        to 170, 190 -- fine -- but lon_min=-10, lon_max=10 wraps to 350, 10
        -- NOT fine, since 350 > 10 and a plain slice would select nothing
        or the complement of what you want).
        """
        lo, hi = self.as_0_360()
        return lo > hi


# Known coastal-community regions. Add more here rather than commenting /
# uncommenting bboxes in the analysis script.
REGIONS = {
    "cambridge_bay": RegionBBox("cambridge_bay", -130, -80, 60, 80),
    # Kivalina, AK sits close to the 180 meridian. -190 % 360 = 170 and
    # -140 % 360 = 220, so this particular box does NOT cross the seam
    # (170 < 220) -- but always check with `crosses_seam()` rather than
    # assuming, especially if you nudge these bounds later.
    "kivalina": RegionBBox("kivalina", -190, -140, 60, 80),
}


@dataclass(frozen=True)
class GridPaths:
    """Filesystem paths to native-grid descriptor files."""

    atm_scrip_lr: str = (
        "/glade/p/cesmdata/cseg/inputdata/share/scripgrids/ne30np4_091226_pentagons.nc"
    )
    atm_scrip_hr: str = (
        "/glade/p/cesmdata/cseg/inputdata/share/scripgrids/ne120np4_pentagons_100310.nc"
    )
    pop_grid_lr: str = "POP_gx1v7"
    pop_grid_hr: str = "POP_tx0.1v2"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for one perfect-model experiment run."""

    # ---- ensemble member roots ----
    low_res_glob: str = "/glade/campaign/collections/gdex/data/d651030/BHIST/*"
    high_res_glob: str = "/glade/campaign/collections/gdex/data/d651007/b.e13.*"

    # ---- variables ----
    low_vars: tuple[str, ...] = ("hi", "aice", "U", "V")
    target_var: str = "hi"
    start_year: int = 1920

    # ---- predictor construction mode ----
    # False (default): the "interpolated xESMF pipeline" -- X's ice/atm
    #   channels are regridded directly from the HR native grid to
    #   `lr_dest_grid` via the `*_hr_to_lr` xESMF regridders (bilinear /
    #   nearest_s2d).
    # True: the "native area-mean pipeline" -- X's channels are first
    #   regridded HR-native -> `hr_dest_grid` (the `*_hr_to_hr` regridders,
    #   same ones Y is built from), then block-averaged (cos-lat weighted,
    #   binned on lat AND lon) down to `lr_dest_grid` resolution using
    #   `grids.area_average_to_grid`. This only supports a rectilinear
    #   `lr_dest_grid` ("1deg" / "0p1deg") -- it will raise if `lr_dest_grid`
    #   is "ease2_n25km", since area-averaging onto a curvilinear projected
    #   grid isn't implemented (see area_average_to_grid's docstring).
    use_area_average_lr: bool = False

    # ---- region ----
    region: RegionBBox = field(default_factory=lambda: REGIONS["cambridge_bay"])

    # ---- destination grids ----
    # "1deg" / "0p1deg" are the legacy rectilinear destinations.
    # `lr_dest_grid` is where the coarsened predictor channels (X) land --
    # set it to "ease2_n25km" to regrid ice/atm predictors onto the NSIDC
    # EASE-Grid 2.0 Northern Hemisphere 25 km grid (see grids.py) via the
    # interpolated xESMF pipeline. NOTE: "ease2_n25km" is INCOMPATIBLE with
    # `use_area_average_lr=True` (see above) -- use one or the other.
    # `hr_dest_grid` is the common high-res grid both X (pre-coarsening)
    # and Y land on; it stays rectilinear in all of today's use cases.
    lr_dest_grid: str = "0p1deg"
    hr_dest_grid: str = "0p1deg"

    # ---- grid file paths ----
    grid_paths: GridPaths = field(default_factory=GridPaths)

    # ---- output ----
    output_dir: str = "/glade/work/skygale/_projects/SeaIceDownscaling/data"
    created_by: str = "Sky Gale"

    # ---- regridding weight cache ----
    weights_dir: str = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

    # ---- parallelism ----
    max_workers_io: int = 64
    max_workers_hr: int = 2

    def __post_init__(self) -> None:
        for v in self.low_vars:
            if v not in VAR_COMPONENT:
                raise ValueError(
                    f"Unknown variable '{v}': add it to VAR_COMPONENT in config.py "
                    f"so the pipeline knows which CESM component (atm/ice/...) writes it."
                )
        if self.target_var not in VAR_COMPONENT:
            raise ValueError(f"Unknown target_var '{self.target_var}'.")
        if self.use_area_average_lr and self.lr_dest_grid not in ("1deg", "0p1deg"):
            raise ValueError(
                f"use_area_average_lr=True requires a rectilinear lr_dest_grid "
                f"('1deg' or '0p1deg'), got '{self.lr_dest_grid}'. The native "
                f"area-mean pipeline bins onto a 1D lat/lon grid and does not "
                f"support a curvilinear destination like 'ease2_n25km'. Either "
                f"set use_area_average_lr=False to use the interpolated xESMF "
                f"pipeline onto ease2_n25km instead, or switch lr_dest_grid to "
                f"a rectilinear option."
            )
        if self.region.crosses_seam():
            raise ValueError(
                f"Region '{self.region.name}' crosses the 0/360 longitude seam "
                f"({self.region.lon_min}, {self.region.lon_max} -> "
                f"{self.region.as_0_360()}). A plain lon slice will not select "
                f"the right cells for this box -- see select_region() in regrid.py "
                f"for the wraparound-aware path, and use that explicitly."
            )


DEFAULT_CONFIG = PipelineConfig(lr_dest_grid="1deg", hr_dest_grid="0p1deg")
