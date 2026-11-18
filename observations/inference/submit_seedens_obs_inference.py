"""
2026-11-17: the seed-ensemble baseline on the ICESat-2 testbed. Runs the 9 extra det-MSE U-Nets (results/FINAL2_FOSI_detmse_seedens,
seeds 1-9, trained on the same 1,825 random days as the final model) on PIOMAS 2018-2022 with exactly the settings of
submit_final2_obs_inference.py (inference only, coarse-zero rule, normalization from the training days), into one batch
results/PIOMAS_2018-2022_FINAL2_FOSI_detmse_seedens; seed 0 is the existing results/PIOMAS_2018-2022_FINAL2_FOSI_detmse.
Then samples the ICESat-2 pass-cells: ICESat2_IS2SITDAT4_cells/eval/final2_seedens_piomas_{cells,bins}.npz.
Scored by evaluation/revisions/is2_seed_ensemble.py.

usage: python submit_seedens_obs_inference.py [--submit]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submit_final2_obs_inference import V6, PROC, P1822, qsub, common  # noqa: E402

BATCH = "PIOMAS_2018-2022_FINAL2_FOSI_detmse_seedens"


def main():
    a = argparse.ArgumentParser(); a.add_argument("--submit", action="store_true"); a = a.parse_args()
    for p in P1822:
        assert os.path.exists(p), p
    jobs = []
    for rc in sorted(glob.glob(f"{V6}/results/FINAL2_FOSI_detmse_seedens/*/run_config.json")):
        c, v = common(rc)
        assert v["DETERMINISTIC_MSE"] == "true" and v["RANDOM_DAYS_SEED"] == 0, (rc, v["DETERMINISTIC_MSE"], v["RANDOM_DAYS_SEED"])
        vv = {**v, "BATCH_NAME": BATCH, "TEST_YEARS": "2018-2022", "TEST_X_PATH": P1822[0], "TEST_Y_PATH": P1822[1]}
        jobs.append(qsub(["-N", f"I2se{c['seed']}pio", "-m", "n", "-l", "walltime=03:00:00",
                          "-v", ",".join(f"{k}={x}" for k, x in vv.items()), "submit_engressnet_daily.sh"], a.submit))
    smp = qsub(["-N", "seedens_cells", "-W", "depend=afterok:" + ":".join(j.split(".")[0] for j in jobs),
                "-l", "walltime=08:00:00", "-v", f"STEPS=cells:{BATCH}:final2_seedens_piomas", "submit_alongtrack_ensemble.sh"],
               a.submit, PROC)
    print(f"inference jobs {jobs} | sampling {smp}")


if __name__ == "__main__":
    main()
