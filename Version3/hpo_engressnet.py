"""
hpo_engressnet.py

Hyperparameter optimization for EngressNet using Optuna. Reuses the same
run_pipeline() as train_engressnet.py, so each trial runs the full
load -> split -> train -> evaluate pipeline with a sampled set of
hyperparameters (learning rate, training ensemble size K, batch size,
latent channel count) and reports back the test-set RMSE of the stochastic
UNet's ensemble mean for Optuna to minimize.

Trials run with make_figures=False and a short --trial-epochs to keep the
search fast; once you have a winning config, re-run train_engressnet.py
with those hyperparameters (and the usual --num-epochs) for the full
figures/metrics/checkpoint.

Install (if not already in downscaling_env):
    pip install optuna --break-system-packages

Examples:
    python hpo_engressnet.py --train-years 1980-2005 --test-years 2006-2014 \\
        --patches --n-trials 30

    python hpo_engressnet.py --train-years 1980-2005 --test-years 2006-2014 \\
        --no-patches --lat-min 65 --lat-max 72 --lon-min -170 --lon-max -155 \\
        --n-trials 30
"""

import argparse
import json
import os
import time

import torch

import functions_engressnet as fe

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "hpo_engressnet.py requires optuna. Install it with "
        "'pip install optuna --break-system-packages' (or add it to downscaling_env) and try again."
    ) from exc

torch.manual_seed(0)


def parse_args():
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for EngressNet.")

    p.add_argument("--data-dir", default=fe.DEFAULT_DATA_DIR)
    p.add_argument("--x-path", default=None)
    p.add_argument("--y-path", default=None)
    p.add_argument("--weighted-grids-dir", default=fe.DEFAULT_WEIGHTED_GRIDS_DIR)
    p.add_argument("--output-dir", default=None)

    p.add_argument("--train-years", default=None)
    p.add_argument("--test-years", default=None)
    p.add_argument("--train-frac", type=float, default=0.8)

    patch_group = p.add_mutually_exclusive_group()
    patch_group.add_argument("--patches", dest="use_patches", action="store_true")
    patch_group.add_argument("--no-patches", dest="use_patches", action="store_false")
    p.set_defaults(use_patches=True)

    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)

    p.add_argument("--context-size", type=int, nargs=2, default=list(fe.DEFAULT_CONTEXT_SIZE), metavar=("H", "W"))
    p.add_argument("--target-size", type=int, nargs=2, default=list(fe.DEFAULT_TARGET_SIZE), metavar=("H", "W"))
    p.add_argument("--stride", type=int, default=fe.DEFAULT_STRIDE)

    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--trial-epochs", type=int, default=8,
                    help="Epochs per trial (kept short; re-run train_engressnet.py with the winning config for a full run).")
    p.add_argument("--k-eval", type=int, default=3, help="Kept small during search to save time.")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if not args.use_patches:
        missing = [name for name in ("lat_min", "lat_max", "lon_min", "lon_max") if getattr(args, name) is None]
        if missing:
            flags = ", ".join("--" + m.replace("_", "-") for m in missing)
            p.error(f"--no-patches requires {flags}")
    return args


def build_base_config(args, output_dir):
    x_path = args.x_path or os.path.join(args.data_dir, "X_perfmodexp_interp.nc")
    y_path = args.y_path or os.path.join(args.data_dir, "Y_perfmodexp.nc")

    subdomain = None
    if not args.use_patches:
        subdomain = {"lat_min": args.lat_min, "lat_max": args.lat_max, "lon_min": args.lon_min, "lon_max": args.lon_max}

    return argparse.Namespace(
        x_path=x_path,
        y_path=y_path,
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
        num_epochs=args.trial_epochs,
        k_eval=args.k_eval,
        eval_batch_size=args.eval_batch_size,
        make_figures=False,
        seed=args.seed,
    )


def make_objective(base_config, output_dir):
    def objective(trial):
        config = argparse.Namespace(**vars(base_config))
        config.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        config.k = trial.suggest_categorical("k", [3, 6, 9])
        config.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        config.latent_channels = trial.suggest_categorical("latent_channels", [4, 8, 16])
        config.output_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(config.output_dir, exist_ok=True)

        result = fe.run_pipeline(config)
        trial_rmse = result["metrics_df"].set_index("Method").loc["Stochastic UNet Mean", "RMSE"]

        trial.set_user_attr("metrics_csv", os.path.join(config.output_dir, "metrics.csv"))
        return trial_rmse

    return objective


def main():
    args = parse_args()
    run_tag = os.environ.get("PBS_JOBID", time.strftime("%Y%m%d_%H%M%S"))
    output_dir = args.output_dir or os.path.join(args.data_dir, "results", f"hpo_{run_tag}")
    os.makedirs(output_dir, exist_ok=True)
    print("HPO output directory:", output_dir)

    base_config = build_base_config(args, output_dir)
    study = optuna.create_study(direction="minimize", study_name="engressnet_hpo")
    study.optimize(make_objective(base_config, output_dir), n_trials=args.n_trials)

    print("Best trial:", study.best_trial.number)
    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump({"best_value_rmse": study.best_value, "best_params": study.best_params}, f, indent=2)

    study.trials_dataframe().to_csv(os.path.join(output_dir, "trials.csv"), index=False)
    print("Saved best_params.json and trials.csv to", output_dir)


if __name__ == "__main__":
    main()
