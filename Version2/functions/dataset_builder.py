from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

from .config import PipelineConfig, VAR_COMPONENT
from .grids import GridBundle
from .regrid import select_region


# ── worker state: sent once per process, not once per job ─────────────────

_regridders: dict = {}

def _init_workers(regridders: dict) -> None:
    global _regridders
    _regridders = regridders


def _process_file_both(args):
    """Open one file, return (interp_da, coarsen_da) in a single file read."""
    component, filepath, varname, mask, rg_to_lr_key, rg_to_hr_key, bbox = args
    rg_to_lr = _regridders[rg_to_lr_key]
    rg_to_hr = _regridders[rg_to_hr_key]

    with xr.open_dataset(filepath) as ds:
        da = ds[varname]
        if component == "ice":
            da = da.rename({"nj": "nlat", "ni": "nlon"}).isel(nlat=mask)
        else:
            if "lev" in da.dims:
                da = da.isel(lev=-1, drop=True)
            da = da.isel(ncol=mask)
        da.load()

    # Pipeline 1: bilinear directly to LR
    da_interp = select_region(rg_to_lr(da), bbox).fillna(0).astype(np.float32)

    # Pipeline 2: bilinear to HR rectilinear, then block average to LR
    da_area = rg_to_hr(da)
    da_area = da_area.coarsen(lat=10, lon=10, boundary="trim").mean()
    da_area = select_region(da_area, bbox).fillna(0).astype(np.float32)

    return da_interp, da_area


def _process_file_target(args):
    """Y pipeline: no coarsening, just regrid to HR destination."""
    component, filepath, varname, mask, rg_key, bbox = args
    rg = _regridders[rg_key]
    with xr.open_dataset(filepath) as ds:
        da = ds[varname]
        if component == "ice":
            da = da.rename({"nj": "nlat", "ni": "nlon"}).isel(nlat=mask)
        da.load()
    return select_region(rg(da), bbox).fillna(0).astype(np.float32)


# ── time alignment (inner join on coordinate values) ──────────────────────

def _align_on_time(arrays):
    exclude = {d for a in arrays for d in a.coords if d != "time"}
    aligned = list(xr.align(*arrays, join="inner", exclude=exclude))
    if aligned[0].sizes["time"] == 0:
        raise ValueError(
            f"Time alignment produced empty intersection. "
            f"Input sizes: {[a.sizes['time'] for a in arrays]}"
        )
    return aligned


def _assemble(
    raw_parts: dict[tuple[int, int], list],
    n_members: int,
    n_vars: int,
    low_vars: list[str],
    pipeline_desc: str,
    config: PipelineConfig,
) -> xr.Dataset:
    member_das = []
    for i in range(n_members):
        channels = [
            xr.concat([da for _, da in sorted(raw_parts[(i, j)])], dim="time")
            for j in range(n_vars)
        ]
        channels = _align_on_time(channels)
        da = xr.concat(channels, dim="channel")
        da.name = "X"
        member_das.append(da.expand_dims({"ensemble": [i]}))

    member_das = _align_on_time(member_das)
    X = xr.concat(member_das, dim="ensemble")
    if "channel" not in X.coords:
        X = X.assign_coords(channel=("channel", low_vars))
    X.attrs.update(
        description=(
            f"Low-res CESM predictors, {pipeline_desc}, "
            f"dest grid '{config.lr_dest_grid}', region '{config.region.name}'."
        ),
        source="POP/pop_tools + xESMF",
        created_by=config.created_by,
    )
    X = X.transpose("ensemble", "time", "channel", ...)
    return xr.Dataset({"X": X}, attrs=X.attrs)


# ── dataset builders ───────────────────────────────────────────────────────

def build_both_predictor_datasets(
    coarsen_files: list[dict[str, list[str]]],
    grids: GridBundle,
    config: PipelineConfig,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Returns (X_interp_ds, X_area_ds) from a single file pass."""
    low_vars = list(config.low_vars)

    all_jobs, job_keys = [], []
    for i, member in enumerate(coarsen_files):
        for j, v in enumerate(low_vars):
            comp = VAR_COMPONENT[v]
            mask = grids.ice_hr.lat_mask if comp == "ice" else grids.atm_hr.ncol_mask
            for k, f in enumerate(member[v]):
                all_jobs.append((
                    comp, f, v, mask,
                    f"{comp}_hr_to_lr",
                    f"{comp}_hr_to_hr",
                    config.region,
                ))
                job_keys.append((i, j, k))

    print(f"Processing {len(all_jobs)} files "
          f"({len(coarsen_files)} members × {len(low_vars)} vars, both pipelines)...")

    with ProcessPoolExecutor(
        max_workers=config.max_workers_io,
        initializer=_init_workers,
        initargs=(grids.regridders,),
    ) as exe:
        results = list(exe.map(_process_file_both, all_jobs))

    interp_parts: dict[tuple[int, int], list] = defaultdict(list)
    area_parts:   dict[tuple[int, int], list] = defaultdict(list)
    for (i, j, k), (da_interp, da_area) in zip(job_keys, results):
        interp_parts[(i, j)].append((k, da_interp))
        area_parts[(i, j)].append((k, da_area))

    n_members, n_vars = len(coarsen_files), len(low_vars)
    X_interp_ds = _assemble(
        interp_parts, n_members, n_vars, low_vars,
        "bilinear directly to LR", config,
    )
    X_area_ds = _assemble(
        area_parts, n_members, n_vars, low_vars,
        f"bilinear to HR rectilinear + gric cell mean",
        config,
    )
    return X_interp_ds, X_area_ds


def build_target_dataset(
    high_res_files: list[dict[str, list[str]]],
    grids: GridBundle,
    config: PipelineConfig,
) -> xr.Dataset:
    all_jobs, job_keys = [], []
    for i, member in enumerate(high_res_files):
        for k, f in enumerate(member[config.target_var]):
            all_jobs.append((
                "ice", f, config.target_var,
                grids.ice_hr.lat_mask, "ice_hr_to_hr",
                config.region,
            ))
            job_keys.append((i, k))

    print(f"Processing {len(all_jobs)} target files ({len(high_res_files)} members)...")

    with ProcessPoolExecutor(
        max_workers=config.max_workers_hr,
        initializer=_init_workers,
        initargs=(grids.regridders,),
    ) as exe:
        results = list(exe.map(_process_file_target, all_jobs))

    parts: dict[int, list] = defaultdict(list)
    for (i, k), da in zip(job_keys, results):
        parts[i].append((k, da))

    member_das = []
    for i in range(len(high_res_files)):
        da = xr.concat([da for _, da in sorted(parts[i])], dim="time")
        da.name = "Y"
        member_das.append(da.expand_dims({"ensemble": [i], "channel": [0]}))

    member_das = _align_on_time(member_das)
    Y = xr.concat(member_das, dim="ensemble")
    if "channel" not in Y.coords:
        Y = Y.assign_coords(channel=("channel", [config.target_var]))
    Y.attrs.update(
        description=(
            f"HR CESM target, dest grid '{config.hr_dest_grid}', "
            f"region '{config.region.name}'."
        ),
        source="POP/pop_tools + xESMF",
        created_by=config.created_by,
    )
    Y = Y.transpose("ensemble", "time", "channel", ...)
    return xr.Dataset({"Y": Y}, attrs=Y.attrs)
