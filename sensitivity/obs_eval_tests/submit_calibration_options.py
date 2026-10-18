"""
Calibration options for EngressNet against ICESat-2 (2026-10-07; all three run, then compared). Everything
starts from the recommended T2 10 px configuration (T2_statespread_p10_FOSI); only real data are used.

  1a noise lr   T2p10 retrained from scratch (4 windows) with the noise scalars + amplitude-map heads at 10x lr
                (their learned values never moved from init in every earlier run)        -> CAL1a_noiselr_FOSI
  1b wide final noise  1a + a 31-tap (edge-safe) kernel for the 0.1 deg noise stage, so its learnable width can reach the
                ~50-160 km scales the 5-tap kernel can't                                   -> CAL1b_widekernel_FOSI
     Each: training (perfect-model FOSI 2021) -> inference on PIOMAS / CS2SMOS / TOPAZ4-ML 2018-2022 (for the pooled product)
  2  obs-calibrated noise  T2p10 checkpoints, everything that sets the mean frozen, noise parameters trained only against
                ICESat-2 pass-cell means on PIOMAS input (pixel CRPS), 2 folds by winter (build_is2_calib_targets.py); each fold's
                run also infers PIOMAS 2018-2022 and is scored only on its held-out winters -> CAL2_obsnoise_fold{A,B}
  3  FOSI + MESA pooled  no new runs (existing t2p10_* and mesa_* samples), computed in calibration_options_compare.py
Then ICESat-2 cell sampling of every new batch -> observations/processing/calibration_options_compare.py.
Gated behind two 1-epoch GPU smoke runs (results/SMOKE_calib, delete afterwards).

usage: python submit_calibration_options.py [--submit]
"""
import argparse
import glob
import os
import subprocess

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
TRAINING, PROC = f"{V6}/training", f"{V6}/observations/processing"
D = "/glade/derecho/scratch/skygale/downscaling/obs_eval_tests"
PIO_X = f"{D}/X_PIOMAS_obs_2017-2022_cdrv6_daily_v4_avg_coastfill.nc"
PIO_Y = f"{D}/Y_PIOMAS_obs_2017-2022_cdrv6_daily_v2.nc"
INPUTS = {"piomas": (PIO_X, PIO_Y),
          **{p: (f"{D}/X_{p}_obs_2018-2022_daily_avg.nc", f"{D}/Y_{p}_obs_2018-2022_placeholder.nc") for p in ("c2s_smos", "topaz4ml")}}
WINDOWS = ["2000-2005", "2005-2010", "2010-2015", "2015-2020"]
COMMON = {"DATA_VARIANT": "avg", "K": 20, "K_EVAL": 20, "BETA": 0.8, "COASTAL_WIDTH": 5, "COASTAL_BOOST": 2.0,
          "LAT_MIN": 60, "LAT_MAX": 75, "LON_MIN": -182, "LON_MAX": -151, "SEED": 0, "NOISE_AMP_MAP": "true",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
OPT1 = {"1a": ("CAL1a_noiselr_FOSI", "cal1a", {"NOISE_LR_MULT": 10}),
        "1b": ("CAL1b_widekernel_FOSI", "cal1b", {"NOISE_LR_MULT": 10, "NOISE_KERNEL_SIZE_FINAL": 31})}
OPT2 = {"NUM_EPOCHS": 15, "TRAIN_ONLY_NOISE": "true", "OBS_FT_ONLY": "true", "OBS_FT_X_PATH": PIO_X, "OBS_FT_WEIGHT": 1,
        "TEST_YEARS": "2018-2022", "TEST_X_PATH": PIO_X, "TEST_Y_PATH": PIO_Y}


def t2p10_ckpt(w):
    ck = glob.glob(f"{V6}/results/T2_statespread_p10_FOSI/*_{w}_2021_*/model_state_dict.pt")
    assert len(ck) == 1, (w, ck)
    return os.path.relpath(ck[0], TRAINING)


def qsub(args, submit, cwd=TRAINING):
    print("qsub " + " ".join(args)[:230])
    if not submit:
        return f"DRY{len(args)}.casper-pbs"
    return subprocess.run(["qsub"] + args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def env(v):
    return ",".join(f"{k}={x}" for k, x in v.items())


def jid(j):
    return j.split(".")[0]


def main():
    a = argparse.ArgumentParser(); a.add_argument("--submit", action="store_true"); a = a.parse_args()
    s = a.submit
    # smoke: 1 epoch of 1b (wide kernel + noise lr) and of option 2 (frozen, obs-only), both full pipelines
    sm1 = qsub(["-N", "smk_cal1b", "-m", "n", "-l", "walltime=01:00:00", "-v", env({**COMMON, "BATCH_NAME": "SMOKE_calib",
               "TRAIN_YEARS": "2000-2005", "TEST_YEARS": 2021, "NUM_EPOCHS": 1, "ENERGY_PATCH_SIZE": 10, **OPT1["1b"][2]}),
               "submit_engressnet_daily.sh"], s)
    sm2 = qsub(["-N", "smk_cal2", "-m", "n", "-l", "walltime=01:00:00", "-v", env({**COMMON, **OPT2, "BATCH_NAME": "SMOKE_calib",
               "TRAIN_YEARS": "2000-2005", "NUM_EPOCHS": 1, "INIT_CHECKPOINT": t2p10_ckpt("2000-2005"),
               "OBS_FT_TARGET_PATH": f"{D}/T_IS2cells_mean_foldA.nc"}), "submit_engressnet_daily.sh"], s)
    gate = f"afterok:{jid(sm1)}:{jid(sm2)}"
    leaves, steps = [], []
    for key, (batch, short, flags) in OPT1.items():
        for w in WINDOWS:
            name = f"{short}_{w[2:4]}"
            tr = qsub(["-N", name, "-m", "n", "-l", "walltime=03:00:00", "-W", f"depend={gate}",
                       "-v", env({**COMMON, "BATCH_NAME": batch, "TRAIN_YEARS": w, "TEST_YEARS": 2021, "NUM_EPOCHS": 20,
                                  "ENERGY_PATCH_SIZE": 10, **flags}), "submit_engressnet_daily.sh"], s)
            ck = f"results/{batch}/{name}_{w}_2021_{tr}/model_state_dict.pt"
            for inp, (x, y) in INPUTS.items():
                ib = f"PIOMAS_2018-2022_{batch}" if inp == "piomas" else f"INPUTSWAP_{inp}_{short}_4win"
                leaves.append(qsub(["-N", f"I{short}{inp[:3]}{w[2:4]}", "-m", "n", "-l", "walltime=02:00:00", "-W", f"depend=afterok:{jid(tr)}",
                                    "-v", env({**COMMON, "BATCH_NAME": ib, "TRAIN_YEARS": w, "TEST_YEARS": "2018-2022", "NUM_EPOCHS": 0,
                                               "INIT_CHECKPOINT": ck, "TEST_X_PATH": x, "TEST_Y_PATH": y, **flags}),
                                    "submit_engressnet_daily.sh"], s))
            print(f"JOBS {key} {w} train={tr}")
        steps += [f"cells:PIOMAS_2018-2022_{batch}:{short}_piomas", f"cells:INPUTSWAP_c2s_smos_{short}_4win:{short}_c2s_smos",
                  f"cells:INPUTSWAP_topaz4ml_{short}_4win:{short}_topaz4ml"]
    for fold in ("A", "B"):
        batch = f"CAL2_obsnoise_fold{fold}"
        for w in WINDOWS:
            leaves.append(qsub(["-N", f"cal2{fold}_{w[2:4]}", "-m", "n", "-l", "walltime=03:00:00", "-W", f"depend={gate}",
                                "-v", env({**COMMON, **OPT2, "BATCH_NAME": batch, "TRAIN_YEARS": w, "INIT_CHECKPOINT": t2p10_ckpt(w),
                                           "OBS_FT_TARGET_PATH": f"{D}/T_IS2cells_mean_fold{fold}.nc"}), "submit_engressnet_daily.sh"], s))
        steps.append(f"cells:{batch}:cal2{fold}_piomas")
    smp = qsub(["-N", "cal_sample", "-W", "depend=afterany:" + ":".join(jid(j) for j in leaves), "-l", "walltime=08:00:00",
                "-v", f"STEPS={' '.join(steps)}", "submit_alongtrack_ensemble.sh"], s, PROC)
    print(f"smoke {sm1} {sm2} | {len(leaves)} train/infer leaves | sampling {smp}")


if __name__ == "__main__":
    main()
