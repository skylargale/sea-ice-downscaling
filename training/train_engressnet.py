"""
train_engressnet.py

CLI entry point for EngressNet SIT downscaling training. Parses arguments
(training/test years, sub-domain bounds, model/training hyperparameters, ...)
and hands off to run_pipeline() in functions_engressnet.py, which does the
actual data loading, training, evaluation, and figure generation. The final
model's flags are in recommended_config.md; the PBS templates are
submit_engressnet_daily.sh (FOSI) and submit_engressnet_daily_mesa.sh (MESACLIP).

Example (sub-domain must be inside the ML domain: lat 60-80, lon -190 to -140):
    python train_engressnet.py --train-years 2000-2020 --test-years 2021 \\
        --lat-min 60 --lat-max 75 --lon-min -182 --lon-max -151

Flags of rejected experiments (patch mode, obs fine-tuning, CRPS/mean-MSE loss terms,
softplus output floor, time context, aux inputs, cosine lr, noise-lr warm-up, windowed
attention, wide final noise kernel, mult/add_feat noise, MC-dropout) were removed in the
2026-11 cleanup; the pre-cleanup code is in scratch downscaling/archive/records/.
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

    # Sub-domain (patch mode was removed in the 2026-11 cleanup; --no-patches is accepted and ignored)
    p.add_argument("--no-patches", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--lat-min", type=float, required=True)
    p.add_argument("--lat-max", type=float, required=True)
    p.add_argument("--lon-min", type=float, required=True)
    p.add_argument("--lon-max", type=float, required=True)

    # Model / training hyperparameters
    # lr/k/batch_size defaults come from an early (Version4) hyperparameter search; the final model
    # overrides k (see recommended_config.md).
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
    p.add_argument("--noise-channels", type=int, default=1,
                   help="Appendix ablation (2026-09-17): number of independent noise channels drawn per "
                        "decoder stage (see functions_engressnet.GaussianNoiseStage). Default 1 (final model).")
    p.add_argument("--noise-kernel-size", type=int, default=5,
                   help="Appendix ablation (2026-09-17): fixed footprint (same at all four decoder stages) "
                        "of each GaussianNoiseStage's correlating kernel. Default 5 (final model).")
    p.add_argument("--late-mask-fusion", dest="late_mask_fusion", action="store_true",
                   help="Appendix ablation (2026-09-17): concatenate the land mask after a noise-only "
                        "final fusion (every earlier version of this model) instead of before it. "
                        "Default off = final architecture.")
    p.set_defaults(late_mask_fusion=False)
    p.add_argument("--deterministic-mse", dest="deterministic_mse", action="store_true",
                   help="Train the same backbone as a deterministic baseline: all noise stages "
                        "zeroed, coastal-weighted MSE loss instead of the energy score, and "
                        "every evaluation member is the (identical) deterministic prediction.")
    # Final-model options (2026-09/10; all default off so older runs reproduce)
    p.add_argument("--noise-amp-map", dest="noise_amp_map", action="store_true",
                   help="Per-pixel noise amplitude predicted from each decoder stage's features (state-dependent spread).")
    p.set_defaults(noise_amp_map=False)
    p.add_argument("--energy-patch-size", type=int, default=0,
                   help="Score the energy term over ~N x N tiles instead of the whole field (0 = whole field).")
    p.add_argument("--noise-lr-mult", type=float, default=1.0,
                   help="Learning-rate multiplier for the noise scalars and amplitude-map heads.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="Exponential moving average of the weights with this decay (e.g. 0.999); evaluation and the saved "
                        "checkpoint use the EMA weights (last-step weights saved as model_state_dict_last_step.pt). 0 = off.")
    p.add_argument("--coarse-zero", action="store_true",
                   help="2026-10-23: at evaluation, set members to exactly 0 on ocean cells whose bilinear coarse SIT is exactly 0 "
                        "(the area-mean coarse input is 0 only where the high-res truth is 0). Training is unchanged. Off by default.")
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

    # 2026-11 co-author revisions. Every default reproduces the earlier pipeline exactly; the final model (FINAL2)
    # uses --train-random-days 1825 --random-days-seed 0 over --train-years 2000-2020.
    p.add_argument("--train-random-days", type=int, default=0,
                   help="F: train on this many random dates drawn from --train-years (the pool; --test-years excluded) instead "
                        "of every date in --train-years. All members of each chosen date are used. 0 = off (default).")
    p.add_argument("--random-days-seed", type=int, default=None,
                   help="F: seed for the random date draw (default: --seed, so each seed draws its own dates).")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Split each batch into this many equal chunks (forward/backward per chunk, one optimizer step): same "
                        "gradient, less GPU memory. 1 = off (default).")
    p.add_argument("--random-days-exclude-years", default=None,
                   help="With --train-random-days: years excluded from the date pool (default: --test-years). Inference of a "
                        "random-days model on other test years must pass the training run's test years here (e.g. 2021).")
    # final-round sensitivity tests (2026-11-15); defaults = final model
    p.add_argument("--land-threshold", type=float, default=0.1,
                   help="Ocean-fraction cutoff below which a 0.1 deg cell is land (final model 0.1).")
    p.add_argument("--global-noise", dest="global_noise", action="store_true",
                   help="One white-noise field at the output grid, resized to every decoder stage (sensitivity test).")
    p.add_argument("--encoder-depth", type=int, default=4, choices=(4, 5),
                   help="Encoder blocks; 5 adds a 4th max-pool and its decoder block (needs a 16-divisible crop).")
    p.add_argument("--attention-end", dest="attention_end", action="store_true",
                   help="Windowed self-attention on the fused output features (sensitivity test).")
    p.add_argument("--noise-ablate", default="",
                   help="H.1: comma-separated noise stages to switch off at inference only (d3,d2,d1,final). Empty = none.")

    args = p.parse_args()

    if args.train_random_days and not (args.train_years and args.test_years):
        p.error("--train-random-days needs --train-years (the pool) and --test-years")

    if bool(args.test_x_path) != bool(args.test_y_path):
        p.error("--test-x-path and --test-y-path must be given together.")
    if args.test_x_path and not (args.train_years and args.test_years):
        p.error("--test-x-path/--test-y-path (cross-dataset evaluation) requires both "
                "--train-years and --test-years.")

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
        use_patches=False,
        subdomain=subdomain,
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
        noise_amp_map=args.noise_amp_map,
        energy_patch_size=args.energy_patch_size,
        noise_lr_mult=args.noise_lr_mult,
        coarse_zero=args.coarse_zero,
        ema_decay=args.ema_decay,
        train_random_days=args.train_random_days,
        random_days_seed=args.random_days_seed,
        noise_ablate=[s.strip() for s in args.noise_ablate.split(",") if s.strip()],
        grad_accum_steps=args.grad_accum_steps,
        random_days_exclude_years=fe.parse_years(args.random_days_exclude_years) if args.random_days_exclude_years else None,
    )
    # final-round sensitivity options: recorded only when set, so final-model run configs are unchanged
    for key, default in (("land_threshold", 0.1), ("global_noise", False), ("encoder_depth", 4), ("attention_end", False)):
        if getattr(args, key) != default:
            setattr(config, key, getattr(args, key))

    fe.run_pipeline(config)


if __name__ == "__main__":
    main()
