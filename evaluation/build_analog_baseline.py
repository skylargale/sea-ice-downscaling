#!/usr/bin/env python3
"""
build_analog_baseline.py

Distributional baseline requested by the coauthor review: for each test-time
low-res predictor field X, find the K nearest-neighbor ("analog") X's in the
*training* period and use their corresponding truth Y fields as a K-member
predictive ensemble -- no neural net, no training loop. Common in the
climate-downscaling literature (constructed analogs); this mirrors the
EnScale-paper-style analog baseline the coauthor pointed to.

Deliberately duplicates only the thin orchestration glue from
functions_engressnet.run_pipeline() (load -> clip -> land mask -> split ->
normalize -> crop to sub-domain) rather than reimplementing any of it, then
reuses run_pipeline()'s own compute_metrics_table()/save_evaluation_data()
so the output is byte-for-byte the same fields.npz/metrics.csv schema as a
real training run's output_dir. That means run_daily_eval_batch.py works
against this run's eval_data/ completely unmodified -- rank histogram,
reliability diagram, spread-skill, CRPS, and per-member PSD spread all come
for free, directly comparable to the stochastic UNet's own numbers.

Only supports the no-patches / single-sub-domain path (use_patches=False),
matching the recommended-config MESA/FOSI runs this was built to compare
against -- patches mode has no single fixed X grid to build a KD-tree over
per tile in the same straightforward way.

Usage:
    python evaluation/build_analog_baseline.py \\
        --template-run results/MESA_stochastic_refine_sweep_avg/MESA_refine_avg_2000-2005_2021_5622261.casper-pbs \\
        --k 20
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import functions_engressnet as fe  # noqa: E402


def build_config(template_run, k, output_dir):
    with open(os.path.join(template_run, "run_config.json")) as f:
        cfg_dict = json.load(f)
    cfg_dict["k_eval"] = k
    cfg_dict["output_dir"] = output_dir
    cfg_dict["make_figures"] = False  # run_daily_eval_batch.py regenerates all figures from eval_data/
    return types.SimpleNamespace(**cfg_dict)


def load_and_prepare(config):
    """Replicates run_pipeline()'s load -> land-mask -> split -> normalize ->
    sub-domain-crop sequence exactly (same functions, same call order), but
    stops there -- no model, no training loop."""
    print("Loading data...")
    X_ds = fe.xr.open_dataset(config.x_path)
    Y_ds = fe.xr.open_dataset(config.y_path)
    X_da, Y_da = X_ds.X, Y_ds.Y
    X_time = X_da["time"]
    llat, llon = X_da.lat.values, X_da.lon.values
    hlat, hlon = Y_da.lat.values, Y_da.lon.values

    X = X_da.values
    Y = Y_da.values
    X[:, :, 0, :, :] = np.clip(X[:, :, 0, :, :], None, 6.0)
    Y = np.clip(Y, None, 6.0)

    print("Building land-sea mask...")
    land_mask = fe.build_land_sea_mask(
        hlat, hlon, config.bbox, config.bbox_regrid, config.weighted_grids_dir,
    )

    print("Splitting...")
    (X_train_fields, Y_train_fields, X_test_fields, Y_test_fields,
     time_train, time_test, member_train, member_test) = fe.split_train_test(
        X, Y, X_time, config.train_years, config.test_years, config.train_frac, seed=config.seed,
        months=getattr(config, "months", None),
    )

    X_mean = X_train_fields.mean(dim=(0, 2, 3), keepdim=True)
    X_std = X_train_fields.std(dim=(0, 2, 3), keepdim=True)
    Y_mean = Y_train_fields.mean(dim=(0, 2, 3), keepdim=True)
    Y_std = Y_train_fields.std(dim=(0, 2, 3), keepdim=True)

    X_train = (X_train_fields - X_mean) / (X_std + 1e-6)
    X_test = (X_test_fields - X_mean) / (X_std + 1e-6)
    Y_train = (Y_train_fields - Y_mean) / (Y_std + 1e-6)
    Y_test = (Y_test_fields - Y_mean) / (Y_std + 1e-6)

    if config.use_patches:
        raise NotImplementedError("Analog baseline only supports use_patches=False (single sub-domain).")

    print("Cropping to sub-domain (no patches)...")
    X_train, Y_train, mask_train, _, _ = fe.extract_full_domain(
        X_train, Y_train, land_mask, llon, llat, hlon, hlat, config.subdomain,
    )
    X_test, Y_test, mask_test, test_tile_ids, tile_geometry = fe.extract_full_domain(
        X_test, Y_test, land_mask, llon, llat, hlon, hlat, config.subdomain,
    )

    return dict(
        X_train=X_train, Y_train=Y_train, mask_train=mask_train,
        X_test=X_test, Y_test=Y_test, mask_test=mask_test,
        test_tile_ids=test_tile_ids, tile_geometry=tile_geometry,
        X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std,
        time_test=time_test, land_mask=land_mask, hlat=hlat, hlon=hlon, llat=llat, llon=llon,
    )


def build_analog_ensemble(d, k):
    """KD-tree nearest-neighbor analog search in normalized low-res X space --
    the same space the model itself sees, so distances are comparably scaled
    across channels. Returns K analog Y fields (physical units) per test
    sample, i.e. a synthetic ensemble the same shape as preds_all_phys."""
    N_train = d["X_train"].shape[0]
    N_test = d["X_test"].shape[0]
    X_train_flat = d["X_train"].reshape(N_train, -1).numpy()
    X_test_flat = d["X_test"].reshape(N_test, -1).numpy()

    print(f"Building KD-tree over {N_train} training-period analogs...")
    tree = cKDTree(X_train_flat)
    print(f"Querying {k} nearest analogs for each of {N_test} test samples...")
    _, nn_idx = tree.query(X_test_flat, k=k)  # (N_test, k)
    if k == 1:
        nn_idx = nn_idx[:, None]

    # Multiply by (Y_std + 1e-6), matching the (Y_std + 1e-6) divisor used to normalize in
    # load_and_preprocess -- see training/functions_engressnet.py's run_pipeline and
    # project-seaicedownscaling-v6-iiee-bugfix for why a bare Y_std here would leave true-zero
    # (open water) pixels at a tiny nonzero floor instead of exact 0.0.
    Y_train_phys = (d["Y_train"] * (d["Y_std"] + 1e-6) + d["Y_mean"]).clamp(min=0.0)  # (N_train, C, H, W)
    preds_all_phys = Y_train_phys[torch.as_tensor(nn_idx)]  # (N_test, k, C, H, W)

    ocean_test = (1.0 - d["mask_test"].to(preds_all_phys.dtype)).clamp(0.0, 1.0)
    preds_all_phys = preds_all_phys * ocean_test.unsqueeze(1)

    Y_pred_phys = preds_all_phys.mean(dim=1)
    Y_spread_phys = preds_all_phys.std(dim=1, unbiased=True)
    Y_pred_det_phys = preds_all_phys[:, 0]  # single nearest analog, as the "deterministic" counterpart

    sit_idx = 0
    X_test_sit_phys = d["X_test"][:, sit_idx:sit_idx + 1] * (d["X_std"][:, sit_idx:sit_idx + 1] + 1e-6) + d["X_mean"][:, sit_idx:sit_idx + 1]
    Y_base_phys = F.interpolate(X_test_sit_phys, size=d["Y_test"].shape[-2:], mode="bilinear", align_corners=False)

    Y_test_phys = d["Y_test"] * (d["Y_std"] + 1e-6) + d["Y_mean"]

    return dict(
        X_test_sit_phys=X_test_sit_phys, Y_base_phys=Y_base_phys, Y_pred_det_phys=Y_pred_det_phys,
        preds_all_phys=preds_all_phys, Y_pred_phys=Y_pred_phys, Y_spread_phys=Y_spread_phys,
        Y_test_phys=Y_test_phys, nn_idx=nn_idx,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-run", required=True, help="An existing run dir to clone data/sub-domain config from")
    parser.add_argument("--k", type=int, default=None, help="Number of analogs (default: template run's k_eval)")
    parser.add_argument("--output-dir", default=None, help="Default: results/MESA_analog_baseline/<same window suffix>")
    args = parser.parse_args()

    template_run = args.template_run.rstrip("/")
    with open(os.path.join(template_run, "run_config.json")) as f:
        template_cfg = json.load(f)
    k = args.k or template_cfg["k_eval"]

    window_name = os.path.basename(template_run)
    output_dir = args.output_dir or os.path.join("results", "MESA_analog_baseline", f"MESA_analog_{window_name}")
    os.makedirs(output_dir, exist_ok=True)

    config = build_config(template_run, k, output_dir)
    fe.save_run_config(config)

    d = load_and_prepare(config)
    a = build_analog_ensemble(d, k)

    mesaclip_run = fe.is_mesaclip_run(config.output_dir)
    metrics_df = fe.compute_metrics_table(
        a["Y_base_phys"], a["Y_pred_det_phys"], a["Y_pred_phys"], a["Y_spread_phys"], a["Y_test_phys"],
        mask_test=d["mask_test"], coastal_width=config.coastal_width,
        preds_all_phys=a["preds_all_phys"] if mesaclip_run else None, per_member=mesaclip_run,
    )
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print(metrics_df)

    print("Saving evaluation data for notebook plotting...")
    fe.save_evaluation_data(
        output_dir, a["X_test_sit_phys"], a["Y_base_phys"], a["Y_pred_det_phys"], a["preds_all_phys"],
        a["Y_pred_phys"], a["Y_test_phys"], d["mask_test"], d["test_tile_ids"], d["tile_geometry"],
        d["time_test"], d["land_mask"], d["hlat"], d["hlon"], d["llat"], d["llon"], config.bbox, config.use_patches,
    )
    print("All done. Outputs written to:", output_dir)


if __name__ == "__main__":
    main()
