"""
2026-11-18 K-sweep: re-run the final EngressNet seeds on the 2021 test year with more members, for scoring at K = 20, 50, 100, 200
(evaluation/revisions/k_sweep.py). Inference only (NUM_EPOCHS=0, INIT_CHECKPOINT), same settings as each training run (normalization
from the run's 1,825 training days, coarse-zero rule) via submit_final2_obs_inference.common(); only SEED changes, which reseeds the
noise draws (the training days have their own RANDOM_DAYS_SEED).
  FOSI: 10 seeds (FINAL2_FOSI_random1825 + SENS2_FOSI_final s1-9), one run each with K_EVAL=200 -> results/KSWEEP_FOSI
  MESA: 5 seeds (FINAL2_MESA_random1825 + SENS2_MESA_final s1-4), two runs each with K_EVAL=100 (200 members; one K=200 run would not
        fit in memory) -> results/KSWEEP_MESA
The fields.npz files are large (FOSI ~7 GB, MESA ~17 GB each); delete the batches after scoring.

usage: python submit_ksweep.py [--submit]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "observations", "inference"))
from submit_final2_obs_inference import V6, qsub, common  # noqa: E402


def main():
    a = argparse.ArgumentParser(); a.add_argument("--submit", action="store_true"); a = a.parse_args()
    jobs = []
    for ds, tmpl, plan in (("FOSI", "submit_engressnet_daily.sh", [(200, 1000)]),
                           ("MESA", "submit_engressnet_daily_mesa.sh", [(100, 1000), (100, 2000)])):
        rcs = glob.glob(f"{V6}/results/FINAL2_{ds}_random1825/*/run_config.json") + sorted(glob.glob(f"{V6}/results/SENS2_{ds}_final/*/run_config.json"))
        for rc in rcs:
            c, v = common(rc)
            assert v["DETERMINISTIC_MSE"] == "false", rc
            for k_eval, seed_off in plan:
                vv = {**v, "BATCH_NAME": f"KSWEEP_{ds}", "TEST_YEARS": "2021", "K_EVAL": k_eval, "SEED": c["seed"] + seed_off}
                jobs.append(qsub(["-N", f"KS{ds[0]}{c['seed']}_{seed_off // 1000}", "-m", "n", "-l", "walltime=03:00:00",
                                  "-v", ",".join(f"{k}={x}" for k, x in vv.items()), tmpl], a.submit))
    print(f"{len(jobs)} jobs:", " ".join(j.split('.')[0] for j in jobs))


if __name__ == "__main__":
    main()
