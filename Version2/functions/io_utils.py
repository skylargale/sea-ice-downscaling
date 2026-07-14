"""Writing X/Y datasets to disk in a CREDIT-friendly way.

NCAR MILES's CREDIT platform consumes processed training data as chunked
Zarr stores, with separately-stored per-variable normalization/scaling
statistics rather than baked-in normalization
(see https://github.com/NCAR/miles-credit). This module:

1. Writes Zarr by default (NetCDF still available via `fmt="netcdf"`) with
   explicit chunking, instead of the original's single unchunked
   `to_netcdf` call on the full multi-decade, multi-ensemble array, which
   will use a lot of memory and be slow to write/read back for anything
   beyond toy domains.
2. Computes and saves per-channel mean/std as a small sidecar
   (`*_scaling.nc`), the same role CREDIT's separately-published
   "Scaling/transform values for normalizing the data" file plays for ERA5.

This module does NOT try to match CREDIT's exact internal config schema or
Dataset class -- I haven't verified those interfaces against current
CREDIT source/docs and don't want to hand you copied-but-wrong field names.
Before wiring this into an actual CREDIT training config, check
https://miles-credit.readthedocs.io/en/latest/ for the current expected
Zarr variable/dim naming and config keys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def default_chunks(ds: xr.Dataset) -> dict[str, int]:
    """A reasonable default chunking: one chunk per (ensemble, channel),
    full spatial extent, and a moderate time chunk. Tune for your actual
    array sizes -- this is a starting point, not a tuned answer.
    """
    chunks = {"ensemble": 1, "channel": 1}
    if "time" in ds.dims:
        chunks["time"] = min(120, ds.sizes["time"])  # ~10 years of monthly data
    for spatial_dim in ds.dims:
        if spatial_dim not in chunks:
            chunks[spatial_dim] = ds.sizes[spatial_dim]  # no chunking across space
    return chunks


def save_dataset(
    ds: xr.Dataset,
    output_dir: str,
    name: str,
    *,
    fmt: str = "zarr",
    chunks: dict[str, int] | None = None,
) -> str:
    """Save a dataset, chunked, as Zarr (default) or NetCDF.

    Returns the path written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chunks = chunks or default_chunks(ds)
    ds_chunked = ds.chunk(chunks)

    if fmt == "zarr":
        path = str(Path(output_dir) / f"{name}.zarr")
        ds_chunked.to_zarr(path, mode="w", consolidated=True)
    elif fmt == "netcdf":
        path = str(Path(output_dir) / f"{name}.nc")
        encoding = {
            var: {"zlib": True, "complevel": 4, "chunksizes": tuple(
                chunks.get(d, ds_chunked.sizes[d]) for d in ds_chunked[var].dims
            )}
            for var in ds_chunked.data_vars
        }
        ds_chunked.to_netcdf(path, encoding=encoding, engine="h5netcdf")
    else:
        raise ValueError(f"Unknown fmt '{fmt}'; expected 'zarr' or 'netcdf'.")

    print(f"Saved to: {path}")
    return path


def compute_and_save_scaling(
    ds: xr.Dataset,
    var_name: str,
    output_dir: str,
    name: str,
) -> str:
    """Compute per-channel mean/std (over ensemble, time, and space) and
    save as a small sidecar NetCDF -- the role CREDIT's separately
    distributed ERA5 scaling file plays. Compute this from TRAINING data
    only if you have a train/val/test split; this function doesn't know
    about splits, so call it on the appropriately-subset dataset.
    """
    da = ds[var_name]
    reduce_dims = [d for d in da.dims if d != "channel"]
    mean = da.mean(dim=reduce_dims, skipna=True)
    std = da.std(dim=reduce_dims, skipna=True)
    scaling = xr.Dataset({"mean": mean, "std": std})
    scaling.attrs["description"] = (
        f"Per-channel mean/std for '{var_name}', computed over dims {reduce_dims}. "
        f"Apply as (x - mean) / std before feeding to the model."
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(output_dir) / f"{name}_scaling.nc")
    scaling.to_netcdf(path)
    print(f"Saved scaling stats to: {path}")
    return path
