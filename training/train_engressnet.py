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
    p.add_argument("--test-x-path", default=None,
                   help="Cross-dataset evaluation: evaluate on a different dataset than the model "
                        "trained on (e.g. train on FOSI, test on MESA-HR). Must be given together "
                        "with --test-y-path, and with --train-years/--test-years both set (no "
                        "random-split fallback). The test dataset must share a grid with --x-path.")
    p.add_argument("--test-y-path", default=None, help="See --test-x-path. Must be given together with it.")
    p.add_argument("--weighted-grids-dir", default=fe.DEFAULT_WEIGHTED_GRIDS_DIR, help="Where regridding weight files are cached/reused across runs.")
    p.add_argument("--output-dir", default=None, help="Defaults to <data-dir>/results/<PBS_JOBID or timestamp>")

    # Train/test years
    p.add_argument("--train-years", default=None, help='e.g. "1980-2005" or "1980,1985,1990-1995"')
    p.add_argument("--test-years", default=None, help='e.g. "2006-2014"')
    p.add_argument("--train-frac", type=float, default=0.8, help="Used only if --train-years/--test-years are not given (random split fallback).")
    p.add_argument("--months", default=None,
                   help='Restrict both train and test samples to these calendar months before '
                        'the year split, e.g. "3-7" for March-July. Comma/range syntax like '
                        '--train-years. No wraparound (e.g. "11-2") support. Default: all months.')

    # Patches vs. single sub-domain
    patch_group = p.add_mutually_exclusive_group()
    patch_group.add_argument("--patches", dest="use_patches", action="store_true", help="Use sliding-window patch extraction (default).")
    patch_group.add_argument("--no-patches", dest="use_patches", action="store_false", help="Train on one full lat/lon sub-domain instead of patches.")
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
    # lr/k/batch_size defaults come from optimization/trial_results.csv trial 11 (best rmse,
    # 0.1321, tied with trial 13) -- see the caveats in submit_engressnet.sh before trusting
    # these blindly on top of the coastal-loss/ocean_frac changes made after that HPO run.
    p.add_argument("--k", type=int, default=9, help="Ensemble size during training.")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0009566023621826124)
    p.add_argument("--coastal-width", type=int, default=5,
                   help="Ocean cells within this many high-res grid cells of land count as "
                        "'coastal' for both the training loss up-weighting and the reported "
                        "Coastal MAE/RMSE metrics.")
    p.add_argument("--coastal-boost", type=float, default=2.0,
                   help="Loss weight multiplier for coastal ocean cells (land stays at the "
                        "baseline weight of 1, other ocean cells are weighted 1).")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Power parameter of the energy-score loss (functions_engressnet.energy_loss). "
                        "Lowering below 1 shifts relative weight toward the ensemble-spread term vs. "
                        "the mean-accuracy term; default 1.0 matches prior behavior unchanged.")
    p.add_argument("--attention-end", dest="attention_end", action="store_true",
                   help="Sensitivity-test toggle: apply local windowed self-attention "
                        "(WindowedSelfAttention2d) to the final 32-channel decoder feature map, "
                        "after the mask+noise fusion. 'Sliding window' attention at the decoder "
                        "end rather than the encoder. Default off = unchanged architecture.")
    p.set_defaults(attention_end=False)
    p.add_argument("--attn-window-size", type=int, default=8,
                   help="Window size for --attention-end's local self-attention. No effect "
                        "unless --attention-end is set.")
    p.add_argument("--attn-num-heads", type=int, default=4,
                   help="Number of attention heads for --attention-end (32 channels must be "
                        "divisible by this). No effect unless --attention-end is set.")
    p.add_argument("--noise-channels", type=int, default=1,
                   help="2026-09-17 sensitivity-test knob: number of independent noise channels "
                        "drawn per decoder stage (see functions_engressnet.GaussianNoiseStage). "
                        "Default 1, the settled value for this single-target-variable model -- "
                        "not intended as a permanent per-run tuning parameter.")
    p.add_argument("--noise-kernel-size", type=int, default=5,
                   help="2026-09-17 sensitivity-test knob: fixed footprint (same at all four "
                        "decoder stages) of each GaussianNoiseStage's correlating kernel. "
                        "Default 5, the settled value -- not intended as a permanent per-run "
                        "tuning parameter.")
    p.add_argument("--late-mask-fusion", dest="late_mask_fusion", action="store_true",
                   help="2026-09-17 ablation-only toggle: reverts the final stage's land-mask "
                        "concatenation to how every earlier version of this model did it -- "
                        "after a noise-only fusion (and after --attention-end, if set) instead "
                        "of before -- while keeping the per-stage noise cascade unchanged "
                        "everywhere, including at the final stage. Isolates whether the "
                        "mask-timing change or the noise-cascade change drives any observed "
                        "difference. Default off = current architecture (mask folds in early). "
                        "Not intended to stay in the codebase once that question is answered.")
    p.set_defaults(late_mask_fusion=False)
    p.add_argument("--deterministic-mse", dest="deterministic_mse", action="store_true",
                   help="Train the same backbone as a deterministic baseline: all noise stages "
                        "zeroed, coastal-weighted MSE loss instead of the energy score, and "
                        "every evaluation member is the (identical) deterministic prediction.")
    # Evaluation
    p.add_argument("--k-eval", type=int, default=6, help="Ensemble size during evaluation.")
    p.add_argument("--eval-batch-size", type=int, default=16)

    p.add_argument("--no-figures", dest="make_figures", action="store_false", help="Skip figure generation (faster).")
    p.set_defaults(make_figures=True)
    p.add_argument("--no-eval-data", dest="save_eval_data", action="store_false",
                   help="Skip saving eval_data/ (raw arrays + tile geometry + candidate-point time series for later notebook plotting). Saved by default.")
    p.set_defaults(save_eval_data=True)
    p.add_argument("--seed", type=int, default=0)

    # Pretraining / transfer learning
    p.add_argument("--init-checkpoint", default=None,
                   help="Path to a model_state_dict.pt from a prior run (e.g. a MESA-trained "
                        "checkpoint) to initialize this run's model weights from, instead of "
                        "random init. Loaded strictly (architecture must match exactly, including "
                        "in_channels -- see --collapse-wind-vector if the source run used a "
                        "different predictor-channel schema).")
    p.add_argument("--collapse-wind-vector", dest="collapse_wind_vector", action="store_true",
                   help="Collapse FOSI's vector wind (u_10, v_10) into a single derived wind-speed "
                        "channel, matching MESA-HR's (hi_d, aice_d, wind_speed) 3-channel schema. "
                        "No-op on datasets that don't carry vector wind (e.g. MESA itself). Needed "
                        "so a checkpoint pretrained on one dataset can be --init-checkpoint-loaded "
                        "into a run on the other -- the two datasets otherwise present different "
                        "channel counts (see functions_engressnet.collapse_wind_vector_channel).")
    p.set_defaults(collapse_wind_vector=False)

    args = p.parse_args()

    if not args.use_patches:
        missing = [name for name in ("lat_min", "lat_max", "lon_min", "lon_max") if getattr(args, name) is None]
        if missing:
            flags = ", ".join("--" + m.replace("_", "-") for m in missing)
            p.error(f"--no-patches requires {flags}")

    if bool(args.test_x_path) != bool(args.test_y_path):
        p.error("--test-x-path and --test-y-path must be given together.")
    if args.test_x_path and not (args.train_years and args.test_years):
        p.error("--test-x-path/--test-y-path (cross-dataset evaluation) requires both "
                "--train-years and --test-years.")

    if args.attention_end and 32 % args.attn_num_heads != 0:
        p.error(f"--attn-num-heads ({args.attn_num_heads}) must divide the decoder's 32 channels.")

    return args


def main():
    args = parse_args()

    x_path = args.x_path or os.path.join(args.data_dir, "X_FOSI_HR_JRA55_interp.nc")
    y_path = args.y_path or os.path.join(args.data_dir, "Y_FOSI_HR_JRA55.nc")

    job_name = os.environ.get("PBS_JOBNAME", "engressnet")
    job_tag = os.environ.get("PBS_JOBID", time.strftime("%Y%m%d_%H%M%S"))
    if args.train_years and args.test_years:
        run_tag = f"{job_name}_{args.train_years}_{args.test_years}"
        if args.months:
            run_tag += f"_m{args.months}"
        run_tag = f"{run_tag}_{job_tag}".replace(",", "-")
    else:
        run_tag = f"{job_name}_{job_tag}"
    output_dir = args.output_dir or os.path.join(fe.DEFAULT_RESULTS_DIR, run_tag)

    subdomain = None
    if not args.use_patches:
        subdomain = {
            "lat_min": args.lat_min, "lat_max": args.lat_max,
            "lon_min": args.lon_min, "lon_max": args.lon_max,
        }

    config = argparse.Namespace(
        x_path=x_path,
        y_path=y_path,
        test_x_path=args.test_x_path,
        test_y_path=args.test_y_path,
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
        months=fe.parse_months(args.months),
        k=args.k,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        coastal_width=args.coastal_width,
        coastal_boost=args.coastal_boost,
        beta=args.beta,
        attention_end=args.attention_end,
        attn_window_size=args.attn_window_size,
        attn_num_heads=args.attn_num_heads,
        noise_channels=args.noise_channels,
        noise_kernel_size=args.noise_kernel_size,
        late_mask_fusion=args.late_mask_fusion,
        deterministic_mse=args.deterministic_mse,
        k_eval=args.k_eval,
        eval_batch_size=args.eval_batch_size,
        make_figures=args.make_figures,
        save_eval_data=args.save_eval_data,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
        collapse_wind_vector=args.collapse_wind_vector,
    )

    fe.run_pipeline(config)


if __name__ == "__main__":
    main()
