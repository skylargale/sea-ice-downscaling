"""Top-level entry point for one perfect-model-experiment data build.

Run as a script:

    python -m sea_ice_downscaling.build_dataset --region cambridge_bay --dest-grid ease2_n25km

or import and call ``run_pipeline(config)`` directly from a notebook /
CREDIT preprocessing config, which is the intended integration point for
moving this off of notebook-only execution.
"""

from __future__ import annotations

import argparse

import torch

from .config import DEFAULT_CONFIG, REGIONS, PipelineConfig, VAR_COMPONENT
from .dataset_builder import build_predictor_dataset, build_target_dataset
from .file_discovery import collect_files, discover_member_dirs, summarize_collection
from .grids import build_grid_bundle
from .io_utils import compute_and_save_scaling, save_dataset


def run_pipeline(config: PipelineConfig, *, save_fmt: str = "zarr") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    low_res_dirs = discover_member_dirs(config.low_res_glob)
    high_res_dirs = discover_member_dirs(config.high_res_glob)

    low_res_files = collect_files(low_res_dirs, config.low_vars, VAR_COMPONENT, config.start_year)
    high_res_files = collect_files(high_res_dirs, [config.target_var], VAR_COMPONENT, config.start_year)
    coarsen_files = collect_files(high_res_dirs, config.low_vars, VAR_COMPONENT, config.start_year)

    summarize_collection("Low-res ", low_res_files)
    summarize_collection("High-res", high_res_files)
    summarize_collection("Coarsen ", coarsen_files)

    print("Building native grids and regridders...")
    grids = build_grid_bundle(config)
    print("Grid setup done.")

    print("Building X (low-res predictors, coarsened from high-res)...")
    X_ds = build_predictor_dataset(coarsen_files, grids, config)
    x_path = save_dataset(X_ds, config.output_dir, "X_perfmodexp_cons", fmt=save_fmt)
    compute_and_save_scaling(X_ds, "X", config.output_dir, "X_perfmodexp_cons")

    print("Building Y (high-res target)...")
    Y_ds = build_target_dataset(high_res_files, grids, config)
    y_path = save_dataset(Y_ds, config.output_dir, "Y_perfmodexp", fmt=save_fmt)
    compute_and_save_scaling(Y_ds, "Y", config.output_dir, "Y_perfmodexp")

    print("Done.")
    print(f"  X: {x_path}")
    print(f"  Y: {y_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", choices=list(REGIONS), default="cambridge_bay")
    p.add_argument(
        "--dest-grid", default="1deg",
        help="Destination grid for the coarsened ice predictor channels "
             "(e.g. '1deg' or 'ease2_n25km').",
    )
    p.add_argument("--start-year", type=int, default=1920)
    p.add_argument("--save-fmt", choices=["zarr", "netcdf"], default="zarr")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = PipelineConfig(
        region=REGIONS[args.region],
        dest_grid=args.dest_grid,
        start_year=args.start_year,
    )
    run_pipeline(config, save_fmt=args.save_fmt)


if __name__ == "__main__":
    main()
