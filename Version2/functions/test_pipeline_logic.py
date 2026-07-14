"""Unit tests for logic that doesn't require GLADE filesystem access,
pop_tools, or xesmf -- i.e. everything that can be checked in CI without
an HPC environment.

Run with: pytest tests/test_pipeline_logic.py -v
"""

import math

import numpy as np
import pytest

from sea_ice_downscaling.config import PipelineConfig, RegionBBox, REGIONS
from sea_ice_downscaling.file_discovery import parse_file_start_year
from sea_ice_downscaling.channels import (
    LEGACY_POP_CHANNELS,
    NEW_PIPELINE_CHANNELS,
    apply_channel_processing,
    select_channels,
    find_channel_index,
)


# ---------------------------------------------------------------------------
# file_discovery: year parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path,expected_year",
    [
        ("/glade/.../b.e13.BHIST.cice.h.hi.192001-200512.nc", 1920),
        ("/glade/.../b.e13.BHIST.cam.h0.U.1920-01.nc", 1920),
        ("/glade/.../some.member.id.with.dots.cice.h.aice.200601-210012.nc", 2006),
        ("/glade/.../weird-name_v2.ice.h.hi.205001-209912.nc", 2050),
    ],
)
def test_parse_file_start_year(path, expected_year):
    assert parse_file_start_year(path) == expected_year


def test_parse_file_start_year_raises_on_unparseable_name():
    with pytest.raises(ValueError):
        parse_file_start_year("/glade/no_date_here.nc")


# ---------------------------------------------------------------------------
# config: longitude seam detection
# ---------------------------------------------------------------------------
def test_cambridge_bay_does_not_cross_seam():
    assert REGIONS["cambridge_bay"].crosses_seam() is False


def test_kivalina_does_not_cross_seam():
    assert REGIONS["kivalina"].crosses_seam() is False


def test_prime_meridian_straddling_box_crosses_seam():
    box = RegionBBox("test", -10, 10, 60, 80)
    assert box.crosses_seam() is True


def test_pipeline_config_rejects_seam_crossing_region():
    bad_region = RegionBBox("bad", -10, 10, 60, 80)
    with pytest.raises(ValueError, match="crosses the 0/360"):
        PipelineConfig(region=bad_region)


def test_pipeline_config_rejects_unknown_variable():
    with pytest.raises(ValueError, match="Unknown variable"):
        PipelineConfig(low_vars=("hi", "aice", "not_a_real_var"))


def test_default_config_constructs_cleanly():
    cfg = PipelineConfig()
    assert cfg.region.name == "cambridge_bay"


# ---------------------------------------------------------------------------
# EASE-Grid 2.0 projection geometry sanity check (no pyproj dependency)
# ---------------------------------------------------------------------------
def test_ease2_n25km_extent_reaches_near_equator():
    """The published EASE2_N25km half-extent (9,000,000 m) should put the
    grid edge near the equator under a polar Lambert azimuthal equal-area
    projection -- this is a coarse spherical-approximation cross-check of
    the geometry used in grids.build_ease2_n25km_grid, independent of the
    actual pyproj/WGS84 implementation.
    """
    R = 6371228.0  # mean earth radius, m
    rho_edge = 9_000_000.0
    half_angle = math.asin(rho_edge / (2 * R))
    lat_edge = 90 - 2 * math.degrees(half_angle)
    assert -5 < lat_edge < 5  # near the equator, not e.g. mid-latitudes


# ---------------------------------------------------------------------------
# curvilinear region masking (the logic behind regrid.select_region's
# 2D-lat/lon branch, used for EASE-Grid 2.0 destinations)
# ---------------------------------------------------------------------------
def test_curvilinear_region_mask_selects_expected_area():
    ny, nx = 50, 50
    y_idx, x_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lat2d = 50 + 35 * (y_idx / ny)
    lon2d = 200 + 100 * (x_idx / nx)

    lat_min, lat_max = 60, 80
    lon_min, lon_max = 230, 280

    mask = (
        (lat2d >= lat_min) & (lat2d <= lat_max) &
        (lon2d >= lon_min) & (lon2d <= lon_max)
    )
    assert mask.sum() > 0

    yy, xx = np.where(mask)
    sub_lat = lat2d[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    sub_lon = lon2d[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    assert sub_lat.min() >= lat_min - 1.0  # loose bound, grid resolution dependent
    assert sub_lon.max() <= lon_max


# ---------------------------------------------------------------------------
# channels: name-based unit conversion/clipping/subsetting
# ---------------------------------------------------------------------------
def test_legacy_channel_processing_matches_old_positional_behavior():
    np.random.seed(0)
    X_old = np.random.randn(1, 2, 5, 4, 4).astype(np.float32) * 10
    channel_order = ["hi", "Tsfc", "SST", "uvel", "vvel"]

    X_proc = apply_channel_processing(X_old, channel_order, LEGACY_POP_CHANNELS)

    assert np.allclose(X_proc[:, :, 3, :, :], X_old[:, :, 3, :, :] / 100.0)
    assert np.allclose(X_proc[:, :, 4, :, :], X_old[:, :, 4, :, :] / 100.0)
    assert X_proc[:, :, 0, :, :].max() <= 6.0
    assert np.allclose(X_proc[:, :, 1, :, :], X_old[:, :, 1, :, :])


def test_select_channels_matches_old_positional_subsetting():
    np.random.seed(0)
    X_old = np.random.randn(1, 2, 5, 4, 4).astype(np.float32) * 10
    channel_order = ["hi", "Tsfc", "SST", "uvel", "vvel"]
    X_proc = apply_channel_processing(X_old, channel_order, LEGACY_POP_CHANNELS)

    X_sub, new_order = select_channels(X_proc, channel_order, ["hi", "uvel", "vvel"])
    assert new_order == ["hi", "uvel", "vvel"]
    assert np.allclose(X_sub[:, :, 0], X_proc[:, :, 0])
    assert np.allclose(X_sub[:, :, 1], X_proc[:, :, 3])
    assert np.allclose(X_sub[:, :, 2], X_proc[:, :, 4])


def test_new_pipeline_channels_do_not_apply_legacy_cm_per_s_conversion():
    """The bug this guards against: U/V from this package's atm-component
    pipeline are already m/s, NOT cm/s like the old POP ocean velocities.
    Applying the legacy /100 conversion to them would silently shrink wind
    data 100x with no error.
    """
    np.random.seed(0)
    X_new = np.random.randn(1, 2, 4, 4, 4).astype(np.float32)
    channel_order = ["hi", "aice", "U", "V"]
    X_proc = apply_channel_processing(X_new, channel_order, NEW_PIPELINE_CHANNELS)
    assert np.allclose(X_proc[:, :, 2, :, :], X_new[:, :, 2, :, :])
    assert np.allclose(X_proc[:, :, 3, :, :], X_new[:, :, 3, :, :])


def test_apply_channel_processing_raises_on_unrecognized_channel():
    X = np.random.randn(1, 2, 3, 4, 4).astype(np.float32)
    with pytest.raises(KeyError):
        apply_channel_processing(X, ["hi", "aice", "mystery_var"], NEW_PIPELINE_CHANNELS)


def test_select_channels_raises_on_missing_requested_channel():
    X = np.random.randn(1, 2, 3, 4, 4).astype(np.float32)
    with pytest.raises(KeyError):
        select_channels(X, ["hi", "aice", "U"], ["hi", "V"])


def test_find_channel_index():
    assert find_channel_index(["hi", "uvel", "vvel"], "hi") == 0
    assert find_channel_index(["hi", "uvel", "vvel"], "vvel") == 2
    with pytest.raises(KeyError):
        find_channel_index(["hi", "uvel"], "not_present")
