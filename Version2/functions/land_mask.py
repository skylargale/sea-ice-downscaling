"""Land-sea mask construction for masking out land in loss/eval, functionized
from the notebook's land-mask cell.

Reuses ``grids.build_ice_source_grid`` (for the >=40N row mask) and
``regrid.select_region`` (for the seam-safe bbox subset) instead of
duplicating that logic a second time with a second, possibly-divergent
implementation -- the original notebook had its own one-off version of
both the row-masking and the lon%360 bbox selection, separate from the
data-prep script's versions. Keeping one implementation means a future fix
to seam handling or row masking only needs to happen once.
"""

from __future__ import annotations

import numpy as np
import pop_tools
import torch
import xarray as xr
import xesmf as xe

from .config import RegionBBox
from .grids import build_ice_source_grid
from .regrid import select_region


def build_land_mask(
    pop_grid_name: str,
    dst_grid: xr.Dataset,
    region: RegionBBox,
    weights_dir: str,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Build a binary land mask on the destination grid, subset to a
    region bbox.

    Replicates the notebook's approach: take the native POP grid's KMT
    (>0 = ocean, 0 = land) mask, regrid it the same way the data itself
    gets regridded, then threshold the regridded fractional mask back to
    binary (since bilinear regridding of a 0/1 field produces fractional
    values at coastlines).

    Returns (land_mask_tensor, dst_lat, dst_lon) where land_mask_tensor
    has shape [1, 1, H, W] (matching the notebook's
    ``torch.from_numpy(...).float()[None, None, ...]``) and dst_lat/lon
    are the 1D or 2D coordinate arrays of the region-subset destination
    grid, useful for plotting.
    """
    src = build_ice_source_grid(pop_grid_name)
    native_grid = pop_tools.get_grid(pop_grid_name)
    mask_native = (native_grid.KMT > 0).astype(np.float32)
    mask_sel = mask_native.isel(nlat=src.lat_mask)

    import os
    weight_path = os.path.join(weights_dir, f"{pop_grid_name}_to_landmask_dst.nc")
    os.makedirs(weights_dir, exist_ok=True)
    regridder = xe.Regridder(
        src.grid, dst_grid,
        method="bilinear", periodic=True,
        filename=weight_path,
        reuse_weights=os.path.exists(weight_path),
    )

    mask_reg = regridder(mask_sel)
    land_mask_da = (mask_reg < threshold).astype(np.float32)
    land_mask_da = select_region(land_mask_da, region)

    dst_lat = land_mask_da["lat"].values
    dst_lon = land_mask_da["lon"].values
    land_mask = torch.from_numpy(land_mask_da.values).float()[None, None, ...]

    return land_mask, dst_lat, dst_lon
