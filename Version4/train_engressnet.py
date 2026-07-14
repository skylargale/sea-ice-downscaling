"""
train_engressnet.py

CLI entry point for EngressNet SIT downscaling training. Parses arguments
(training/test years, patches vs. single sub-domain, sub-domain bounds,
model/training hyperparameters, ...) and hands off to run_pipeline() in
functions_engressnet.py, which does the actual data loading, training,
evaluation, and figure generation.

Examples:
    # Patch-based, as before, with an explicit train/test year split
    python train_engressnet.py --train-years 1980-2005 --test-years 2006-2014 --patches

    # No patches: train directly on one lat/lon sub-domain (must be inside
    # the ML domain: lat 60-80, lon -190 to -140), and also get a domain
    # SIT time series for the test period
    python train_engressnet.py --train-years 1980-2005 --test-years 2006-2014 \\
        --no-patches --lat-min 65 --lat-max 72 --lon-min -170 --lon-max -155

    # No year filter -> falls back to the original random 80/20 split
    python train_engressnet.py --patches
"""

import argparse
import os
import time
import torch
import functions_engressnet as fe
torch.manual_seed(0)


def parse_args():
    p = argparse.ArgumentParser(description="Train EngressNet for SIT downscaling.")

    # Paths
    p.add_argument("--data-dir", default=fe.DEFAULT_DATA_DIR)
    p.add_argument("--x-path", default=None, help="Defaults to <data-dir>/X_FOSI_HR_JRA55_interp.nc")
    p.add_argument("--y-path", default=None, help="Defaults to <data-dir>/Y_FOSI_HR_JRA55.nc")
    p.add_argument("--weighted-grids-dir", default=fe.DEFAULT_WEIGHTED_GRIDS_DIR,
                    help="Where regridding weight files are cached/reused across runs.")
    p.add_argument("--output-dir", default=None, help="Defaults to <data-dir>/results/<PBS_JOBID or timestamp>")

    # Train/test years
    p.add_argument("--train-years", default=None, help='e.g. "1980-2005" or "1980,1985,1990-1995"')
    p.add_argument("--test-years", default=None, help='e.g. "2006-2014"')
    p.add_argument("--train-frac", type=float, default=0.8,
                    help="Used only if --train-years/--test-years are not given (random split fallback).")

    # Patches vs. single sub-domain
    patch_group = p.add_mutually_exclusive_group()
    patch_group.add_argument("--patches", dest="use_patches", action="store_true",
                              help="Use sliding-window patch extraction (default).")
    patch_group.add_argument("--no-patches", dest="use_patches", action="store_false",
                              help="Train on one full lat/lon sub-domain instead of patches.")
    p.set_defaults(use_patches=True)

    p.add_argument("--lat-min", type=float, default=None, help="Required with --no-patches.")
    p.add_argument("--lat-max", type=float, default=None, help="Required with --no-patches.")
    p.add_argument("--lon-min", type=float, default=None, help="Required with --no-patches.")
    p.add_argument("--lon-max", type=float, default=None, help="Required with --no-patches.")

    # Patch geometry (only used when --patches)
    p.add_argument("--context-size", type=int, nargs=2, default=list(fe.DEFAULT_CONTEXT_SIZE), metavar=("H", "W"))
    p.add_argument("--target-size", type=int, nargs=2, default=list(fe.DEFAULT_TARGET_SIZE), metavar=("H", "W"))
    p.add_argument("--stride", type=int, default=fe.DEFAULT_STRIDE)

    # Model / training hyperparameters
    p.add_argument("--k", type=int, default=6, help="Ensemble size during training.")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent-channels", type=int, default=8)

    # Evaluation
    p.add_argument("--k-eval", type=int, default=6, help="Ensemble size during evaluation.")
    p.add_argument("--eval-batch-size", type=int, default=16)

    p.add_argument("--no-figures", dest="make_figures", action="store_false", help="Skip figure generation (faster).")
    p.set_defaults(make_figures=True)
    p.add_argument("--no-eval-data", dest="save_eval_data", action="store_false",
                    help="Skip saving eval_data/ (raw arrays + tile geometry + candidate-point time series "
                         "for later notebook plotting). Saved by default.")
    p.set_defaults(save_eval_data=True)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    if not args.use_patches:
        missing = [name for name in ("lat_min", "lat_max", "lon_min", "lon_max") if getattr(args, name) is None]
        if missing:
            flags = ", ".join("--" + m.replace("_", "-") for m in missing)
            p.error(f"--no-patches requires {flags}")

    return args


def main():
    args = parse_args()

    x_path = args.x_path or os.path.join(args.data_dir, "X_FOSI_HR_JRA55_interp.nc")
    y_path = args.y_path or os.path.join(args.data_dir, "Y_FOSI_HR_JRA55.nc")

    job_tag = os.environ.get("PBS_JOBID", time.strftime("%Y%m%d_%H%M%S"))
    if args.train_years and args.test_years:
        run_tag = f"train_{args.train_years}_test_{args.test_years}_{job_tag}".replace(",", "-")
    else:
        run_tag = job_tag
    output_dir = args.output_dir or os.path.join(args.data_dir, "results", run_tag)

    subdomain = None
    if not args.use_patches:
        subdomain = {
            "lat_min": args.lat_min, "lat_max": args.lat_max,
            "lon_min": args.lon_min, "lon_max": args.lon_max,
        }

    config = argparse.Namespace(
        x_path=x_path,
        y_path=y_path,
        output_dir=output_dir,
        weighted_grids_dir=args.weighted_grids_dir,
        bbox=fe.DEFAULT_BBOX,
        bbox_regrid=fe.DEFAULT_BBOX_REGRID,
        use_patches=args.use_patches,
        subdomain=subdomain,
        context_size=tuple(args.context_size),
        target_size=tuple(args.target_size),
        stride=args.stride,
        train_years=fe.parse_years(args.train_years),
        test_years=fe.parse_years(args.test_years),
        train_frac=args.train_frac,
        k=args.k,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_channels=args.latent_channels,
        k_eval=args.k_eval,
        eval_batch_size=args.eval_batch_size,
        make_figures=args.make_figures,
        save_eval_data=args.save_eval_data,
        seed=args.seed,
    )

    fe.run_pipeline(config)


if __name__ == "__main__":
    main()
