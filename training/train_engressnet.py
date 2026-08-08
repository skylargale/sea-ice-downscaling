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
    p.add_argument("--weighted-grids-dir", default=fe.DEFAULT_WEIGHTED_GRIDS_DIR, help="Where regridding weight files are cached/reused across runs.")
    p.add_argument("--output-dir", default=None, help="Defaults to <data-dir>/results/<PBS_JOBID or timestamp>")

    # Train/test years
    p.add_argument("--train-years", default=None, help='e.g. "1980-2005" or "1980,1985,1990-1995"')
    p.add_argument("--test-years", default=None, help='e.g. "2006-2014"')
    p.add_argument("--train-frac", type=float, default=0.8, help="Used only if --train-years/--test-years are not given (random split fallback).")

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
    # lr/k/batch_size defaults come from hpo_echo/trial_results.csv trial 11 (best rmse,
    # 0.1321, tied with trial 13) -- see the caveats in submit_engressnet.sh before trusting
    # these blindly on top of the coastal-loss/ocean_frac changes made after that HPO run.
    p.add_argument("--k", type=int, default=9, help="Ensemble size during training.")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0009566023621826124)
    p.add_argument("--latent-channels", type=int, default=8)
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
    p.add_argument("--extra-layer", dest="extra_layer", action="store_true",
                   help="Sensitivity-test toggle: add a 4th UNet downsample/upsample stage "
                        "(1024-channel bottleneck) below the default 512-channel one. Requires "
                        "the low-res domain crop to be divisible by 16 (not just 8) in both dims "
                        "when --no-patches is used. Default off = unchanged architecture.")
    p.set_defaults(extra_layer=False)
    p.add_argument("--stochastic-refine", dest="stochastic_refine", action="store_true",
                   help="Sensitivity-test toggle: add a single-shot EnScale-style (Schillinger et "
                        "al., 2025) stochastic refinement stage -- sigma-scaled noise mixed by a "
                        "per-location learnable LocallyConnected2d layer (not a translation-invariant "
                        "conv), then a shared MLP -- layered on top of the existing latent-noise "
                        "pathway. Mutually exclusive with --enscale-net. Default off = unchanged "
                        "architecture.")
    p.set_defaults(stochastic_refine=False)
    p.add_argument("--enscale-net", dest="enscale_net", action="store_true",
                   help="Sensitivity-test toggle: replace the entire decoder with the full EnScale "
                        "(Schillinger et al., 2025) mechanism applied progressively at every 2x "
                        "upsampling step (still built on this UNet's own encoder/skip connections, "
                        "not the paper's separate skip-free pyramid) -- see functions_engressnet.UNet "
                        "docstring. Mutually exclusive with --stochastic-refine and --extra-layer. "
                        "Default off = unchanged architecture.")
    p.set_defaults(enscale_net=False)
    p.add_argument("--noise-sigma", type=float, default=1.0,
                   help="Fixed (non-learnable) standard deviation of the raw Gaussian noise fed "
                        "into every LocallyConnected2d mixing layer, for both --stochastic-refine "
                        "and --enscale-net. Deliberately not learned -- this is the sensitivity-test "
                        "'dial': everything about how the network *uses* the noise (the local "
                        "layer's weights, the shared MLP) stays fully trainable regardless of this "
                        "value. No effect if neither --stochastic-refine nor --enscale-net is set. "
                        "Default 1.0 = raw unit-Gaussian noise, as in the paper.")
    p.add_argument("--land-threshold", type=float, default=0.1,
                   help="A high-res grid cell is classified as land only if its regridded ocean "
                        "fraction is below this value (default 0.1, i.e. >90%% land). Tightened "
                        "from an original 0.5 majority-rule threshold, which called mixed coastal "
                        "cells 'land' even though they still carry real ice signal from their "
                        "ocean fraction -- hard-zeroing predictions there (see run_pipeline's "
                        "ocean_test masking) then disagreed with that residual truth signal, "
                        "inflating IIEE.")
    p.add_argument("--coastal-channel", dest="coastal_channel", action="store_true",
                   help="Sensitivity-test toggle: append a low-res ocean-fraction channel to X "
                        "so the encoder sees coastal structure from the first conv layer, instead "
                        "of land/ocean location only being known at the last layer (the land mask "
                        "concatenated before out_conv). Default off = unchanged in_channels.")
    p.set_defaults(coastal_channel=False)
    p.add_argument("--classification-head", dest="classification_head", action="store_true",
                   help="Sensitivity-test toggle: add an auxiliary land/open-ocean/ice "
                        "segmentation head branching off the final decoder feature map, trained "
                        "with an auxiliary cross-entropy loss (weight --classif-weight) alongside "
                        "the main energy-score loss, to give the decoder an explicit "
                        "boundary-aware signal. Default off = no head, no change to model output.")
    p.set_defaults(classification_head=False)
    p.add_argument("--classif-weight", type=float, default=0.1,
                   help="Weight of the auxiliary classification cross-entropy loss relative to "
                        "the main energy-score loss. No effect unless --classification-head is set.")
    p.add_argument("--attention-end", dest="attention_end", action="store_true",
                   help="Sensitivity-test toggle: apply local windowed self-attention "
                        "(WindowedSelfAttention2d) to the final 32-channel decoder feature map, "
                        "right before the land mask is concatenated. 'Sliding window' attention "
                        "at the decoder end rather than the encoder. Default off = unchanged "
                        "architecture.")
    p.set_defaults(attention_end=False)
    p.add_argument("--attn-window-size", type=int, default=8,
                   help="Window size for --attention-end's local self-attention. No effect "
                        "unless --attention-end is set.")
    p.add_argument("--attn-num-heads", type=int, default=4,
                   help="Number of attention heads for --attention-end (32 channels must be "
                        "divisible by this). No effect unless --attention-end is set.")
    p.add_argument("--calibrate-from", default=None,
                   help="Path to an already-trained model_state_dict.pt checkpoint. When set, "
                        "the model loads this checkpoint before training instead of starting from "
                        "scratch. Combine with --freeze-backbone for stage-2 'freeze backbone, "
                        "calibrate noise pathway only' fine-tuning (see "
                        "functions_engressnet.set_noise_only_trainable) -- typically with a "
                        "smaller --num-epochs/--lr than the original training run. Architecture "
                        "flags (--extra-layer/--stochastic-refine/--enscale-net/etc.) must match "
                        "the checkpoint's original run exactly, or load_state_dict will fail on a "
                        "shape mismatch.")
    p.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true",
                   help="Only used with --calibrate-from: freeze every parameter except the "
                        "noise-injection pathway (z_proj_*/concat_d*/local_noise_mix/refiner) so "
                        "training only recalibrates ensemble spread. Not supported with "
                        "--enscale-net (raises -- see set_noise_only_trainable).")
    p.set_defaults(freeze_backbone=False)

    # Evaluation
    p.add_argument("--k-eval", type=int, default=6, help="Ensemble size during evaluation.")
    p.add_argument("--eval-batch-size", type=int, default=16)

    p.add_argument("--no-figures", dest="make_figures", action="store_false", help="Skip figure generation (faster).")
    p.set_defaults(make_figures=True)
    p.add_argument("--no-eval-data", dest="save_eval_data", action="store_false",
                   help="Skip saving eval_data/ (raw arrays + tile geometry + candidate-point time series for later notebook plotting). Saved by default.")
    p.set_defaults(save_eval_data=True)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    if not args.use_patches:
        missing = [name for name in ("lat_min", "lat_max", "lon_min", "lon_max") if getattr(args, name) is None]
        if missing:
            flags = ", ".join("--" + m.replace("_", "-") for m in missing)
            p.error(f"--no-patches requires {flags}")

    if args.enscale_net and args.extra_layer:
        p.error("--enscale-net and --extra-layer are mutually exclusive.")
    if args.enscale_net and args.stochastic_refine:
        p.error("--enscale-net and --stochastic-refine are mutually exclusive "
                "(--enscale-net already injects EnScale-style noise at every decoder stage).")
    if args.attention_end and 32 % args.attn_num_heads != 0:
        p.error(f"--attn-num-heads ({args.attn_num_heads}) must divide the decoder's 32 channels.")
    if args.freeze_backbone and not args.calibrate_from:
        p.error("--freeze-backbone requires --calibrate-from (nothing to calibrate on top of).")
    if args.calibrate_from and args.enscale_net and args.freeze_backbone:
        p.error("--freeze-backbone is not supported with --enscale-net (see "
                "functions_engressnet.set_noise_only_trainable's docstring).")

    return args


def main():
    args = parse_args()

    x_path = args.x_path or os.path.join(args.data_dir, "X_FOSI_HR_JRA55_interp.nc")
    y_path = args.y_path or os.path.join(args.data_dir, "Y_FOSI_HR_JRA55.nc")

    job_name = os.environ.get("PBS_JOBNAME", "engressnet")
    job_tag = os.environ.get("PBS_JOBID", time.strftime("%Y%m%d_%H%M%S"))
    if args.train_years and args.test_years:
        run_tag = f"{job_name}_{args.train_years}_{args.test_years}_{job_tag}".replace(",", "-")
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
        extra_layer=args.extra_layer,
        stochastic_refine=args.stochastic_refine,
        enscale_net=args.enscale_net,
        noise_sigma=args.noise_sigma,
        coastal_width=args.coastal_width,
        coastal_boost=args.coastal_boost,
        beta=args.beta,
        land_threshold=args.land_threshold,
        coastal_channel=args.coastal_channel,
        classification_head=args.classification_head,
        classif_weight=args.classif_weight,
        attention_end=args.attention_end,
        attn_window_size=args.attn_window_size,
        attn_num_heads=args.attn_num_heads,
        calibrate_from=args.calibrate_from,
        freeze_backbone=args.freeze_backbone,
        k_eval=args.k_eval,
        eval_batch_size=args.eval_batch_size,
        make_figures=args.make_figures,
        save_eval_data=args.save_eval_data,
        seed=args.seed,
    )

    fe.run_pipeline(config)


if __name__ == "__main__":
    main()
