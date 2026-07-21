"""
objective_engressnet.py

ECHO objective for EngressNet hyperparameter search.

This replaces the objective(trial) closure that used to live inside
hpo_engressnet.py's make_objective(). The trial logic is unchanged --
same four hyperparameters (lr, k, batch_size, latent_channels), same
run_pipeline() call, same metric (test-set RMSE of the Stochastic UNet
Mean) -- it's just adapted to run inside ECHO's distributed-PBS trial
launcher instead of a single optuna.Study.optimize() loop in one job.

Install (in downscaling_env, alongside optuna which ECHO depends on):
    pip install echo-opt --break-system-packages
"""

import argparse
import os
import sys

import torch

from echo.src.base_objective import BaseObjective

# ECHO's loader imports this file directly via importlib, which does NOT
# add this file's own directory to sys.path the way `python script.py`
# normally does. Without this, functions_engressnet.py (which lives
# alongside this file) isn't found regardless of the current working
# directory echo-run/echo-opt happen to be launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import functions_engressnet as fe


class Objective(BaseObjective):

    def __init__(self, config, metric="rmse"):
        # Initialize the base class
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):
        # ------------------------------------------------------------
        # Sample the same four hyperparameters as the original
        # make_objective() in hpo_engressnet.py, reading bounds/choices
        # from the optuna.parameters block in hyperparameters.yml.
        # ------------------------------------------------------------
        hp = conf["optuna"]["parameters"]

        lr_s = hp["lr"]["settings"]
        lr = trial.suggest_float(lr_s["name"], lr_s["low"], lr_s["high"], log=lr_s.get("log", True))

        k_s = hp["k"]["settings"]
        k = trial.suggest_categorical(k_s["name"], k_s["choices"])

        bs_s = hp["batch_size"]["settings"]
        batch_size = trial.suggest_categorical(bs_s["name"], bs_s["choices"])

        lc_s = hp["latent_channels"]["settings"]
        latent_channels = trial.suggest_categorical(lc_s["name"], lc_s["choices"])

        # ------------------------------------------------------------
        # Build the run_pipeline() config, mirroring build_base_config()
        # from hpo_engressnet.py, but reading fixed settings from the
        # "model" block of model_config.yml instead of argparse CLI flags.
        # ------------------------------------------------------------
        model_conf = conf["model"]

        torch.manual_seed(model_conf.get("seed", 0))

        use_patches = model_conf.get("use_patches", True)
        subdomain = model_conf.get("subdomain")
        if not use_patches:
            missing = [
                name for name in ("lat_min", "lat_max", "lon_min", "lon_max")
                if subdomain is None or subdomain.get(name) is None
            ]
            if missing:
                raise ValueError(
                    f"model.subdomain in model_config.yml is missing {missing}, "
                    "required when model.use_patches is False."
                )

        data_dir = model_conf.get("data_dir", fe.DEFAULT_DATA_DIR)
        x_path = model_conf.get("x_path") or os.path.join(data_dir, "X_FOSI_HR_JRA55_interp.nc")
        y_path = model_conf.get("y_path") or os.path.join(data_dir, "Y_FOSI_HR_JRA55.nc")

        # One subdirectory per trial, same convention as the original script
        output_dir = os.path.join(model_conf["output_dir"], f"trial_{trial.number}")
        os.makedirs(output_dir, exist_ok=True)

        run_config = argparse.Namespace(
            x_path=x_path,
            y_path=y_path,
            output_dir=output_dir,
            weighted_grids_dir=model_conf.get("weighted_grids_dir", fe.DEFAULT_WEIGHTED_GRIDS_DIR),
            bbox=model_conf.get("bbox", fe.DEFAULT_BBOX),
            bbox_regrid=model_conf.get("bbox_regrid", fe.DEFAULT_BBOX_REGRID),
            use_patches=use_patches,
            subdomain=subdomain,
            context_size=tuple(model_conf.get("context_size", fe.DEFAULT_CONTEXT_SIZE)),
            target_size=tuple(model_conf.get("target_size", fe.DEFAULT_TARGET_SIZE)),
            stride=model_conf.get("stride", fe.DEFAULT_STRIDE),
            train_years=fe.parse_years(model_conf.get("train_years")),
            test_years=fe.parse_years(model_conf.get("test_years")),
            train_frac=model_conf.get("train_frac", 0.8),
            num_epochs=model_conf.get("trial_epochs", 8),
            k_eval=model_conf.get("k_eval", 3),
            eval_batch_size=model_conf.get("eval_batch_size", 16),
            make_figures=False,
            save_eval_data=False,
            seed=model_conf.get("seed", 0),
            coastal_width=model_conf.get("coastal_width", 5),
            coastal_boost=model_conf.get("coastal_boost", 2.0),
            beta=model_conf.get("beta", 1.0),
            # sampled hyperparameters
            lr=lr,
            k=k,
            batch_size=batch_size,
            latent_channels=latent_channels,
        )

        result = fe.run_pipeline(run_config)
        rmse = result["metrics_df"].set_index("Method").loc["Stochastic UNet Mean", "RMSE"]

        # Kept for parity with the original script -- lets you trace a
        # trial back to its metrics.csv from the ECHO results dataframe.
        trial.set_user_attr("metrics_csv", os.path.join(output_dir, "metrics.csv"))

        # Key must match optuna.metric in hyperparameters.yml
        return {"rmse": float(rmse)}