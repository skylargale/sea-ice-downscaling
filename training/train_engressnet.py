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
                        "docstring. Mutually exclusive with --stochastic-refine. "
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
    p.add_argument("--noise-smooth-kernel-size", type=int, default=3,
                   help="Kernel size of the avg_pool2d applied to the mixed noise after "
                        "LocallyConnected2d, for both --stochastic-refine and --enscale-net "
                        "(functions_engressnet.smooth_noise). Widening this (e.g. 7 or 9) smooths "
                        "more aggressively -- targets pixel-scale speckle in a single ensemble "
                        "member's output. Default 3 = original behavior.")
    p.add_argument("--noise-bias-smooth-weight", type=float, default=0.0,
                   help="Weight on a total-variation-style penalty applied during training to "
                        "LocallyConnected2d.bias in the --stochastic-refine noise-mixing layer "
                        "(functions_engressnet.noise_bias_smoothness_penalty). That bias is a "
                        "freely-learned per-location parameter added even in the fully "
                        "deterministic (eps=0) pass, so unlike --noise-sigma or "
                        "--noise-smooth-kernel-size, it can leave a persistent non-random rough "
                        "pattern that no runtime noise setting can remove -- confirmed empirically "
                        "2026-08-26 (visible in ensemble_figure.png's 'Deterministic' column, "
                        "concentrated in high-thickness/high-variability regions). Default 0.0 = "
                        "off, no effect on training. No effect if --stochastic-refine isn't set or "
                        "--enscale-net is.")
    p.add_argument("--noise-shared-bias", dest="noise_shared_bias", action="store_true",
                   help="Root-cause fix (2026-08-26) for the same persistent deterministic-pass "
                        "streaky texture --noise-bias-smooth-weight targets, but stronger: ties "
                        "LocallyConnected2d.bias in the --stochastic-refine noise-mixing layer to a "
                        "single shared vector instead of one independent value per grid location, "
                        "removing the degree of freedom that caused it by construction rather than "
                        "penalizing it statistically. The per-location *weight* (which only matters "
                        "when noise is actually nonzero) is unaffected. Default off = original "
                        "independent-per-location bias. No effect if --stochastic-refine isn't set "
                        "or --enscale-net is; incompatible with a nonzero --noise-bias-smooth-weight "
                        "(nothing left to penalize once the bias is shared -- silently a no-op).")
    p.set_defaults(noise_shared_bias=False)
    p.add_argument("--deep-mask-head", dest="deep_mask_head", action="store_true",
                   help="The high-res land mask is already concatenated into the feature map both "
                        "out_conv and the --stochastic-refine refiner consume, but only a single bare "
                        "3x3 conv (or a shallow 3-layer 1x1-conv MLP) processes that concatenation -- "
                        "no real depth to learn coastal-aware behavior. Inserts a small 2-layer conv "
                        "block (3x3 conv + InstanceNorm + ReLU, twice) right after the mask "
                        "concatenation, at full target resolution, before out_conv/refiner see it. "
                        "Not the same as --coastal-channel (which feeds a "
                        "low-res ocean-fraction proxy into the encoder's input, already tested with "
                        "no effect) -- this deepens processing of the real high-res mask at the "
                        "decoder's own resolution. Default off = unchanged architecture.")
    p.set_defaults(deep_mask_head=False)
    p.add_argument("--noise-mix-kernel", choices=["learned", "gaussian", "gaussian_fixed", "none"],
                   default="learned",
                   help="Which layer mixes --stochastic-refine's raw per-pixel noise into "
                        "spatially-correlated texture. 'learned' (default) = LocallyConnected2d, "
                        "the EnScale paper's independently-learned-per-location weight matrix -- "
                        "appropriate for sharp-fronted atmospheric fields, but nothing constrains "
                        "neighboring locations to mix noise similarly, and it's the confirmed source "
                        "of a persistent per-pixel artifact (see --noise-shared-bias). 'none' = no "
                        "mixing layer at all, learned or otherwise: draw noise_mix_channels "
                        "independent noise channels directly and smooth each with the existing "
                        "fixed-kernel --noise-smooth-kernel-size -- zero new learnable parameters, "
                        "the cheapest thing to try before concluding a learned mixing layer is needed. "
                        "'gaussian_fixed'/'gaussian' = GaussianNoiseMix, a genuinely distance-weighted "
                        "alternative (one shared, translation-invariant Gaussian kernel per channel) -- "
                        "'gaussian_fixed' freezes the per-channel smoothing scale at its initial value "
                        "(isolates whether the bell-curve *shape* helps over 'none''s box shape, holding "
                        "width fixed); 'gaussian' lets that scale be learned by backprop instead (isolates "
                        "whether *learning* the width helps, holding shape fixed vs. 'gaussian_fixed'). "
                        "No effect unless --stochastic-refine is set. --noise-shared-bias is a no-op with "
                        "any kernel other than 'learned' (none of the others has a bias term at all).")
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
                        "flags (--stochastic-refine/--enscale-net/etc.) must match "
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

    if bool(args.test_x_path) != bool(args.test_y_path):
        p.error("--test-x-path and --test-y-path must be given together.")
    if args.test_x_path and not (args.train_years and args.test_years):
        p.error("--test-x-path/--test-y-path (cross-dataset evaluation) requires both "
                "--train-years and --test-years.")

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
        latent_channels=args.latent_channels,
        stochastic_refine=args.stochastic_refine,
        enscale_net=args.enscale_net,
        noise_sigma=args.noise_sigma,
        noise_smooth_kernel_size=args.noise_smooth_kernel_size,
        noise_bias_smooth_weight=args.noise_bias_smooth_weight,
        noise_shared_bias=args.noise_shared_bias,
        deep_mask_head=args.deep_mask_head,
        noise_mix_kernel=args.noise_mix_kernel,
        coastal_width=args.coastal_width,
        coastal_boost=args.coastal_boost,
        beta=args.beta,
        land_threshold=args.land_threshold,
        coastal_channel=args.coastal_channel,
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
