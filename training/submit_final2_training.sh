#!/bin/bash
# Final model (FINAL2, 2026-11-14): all training runs behind the paper's results, one dataset or both.
# Run it directly from anywhere (not with qsub); without --submit it only prints the qsub commands.
#
#   ./submit_final2_training.sh [MESA|FOSI|both] [--submit]
#
# Per dataset (reconstructed from the run_config.json of the FINAL2 batches; the runs themselves were submitted by hand):
#   results/FINAL2_<DS>_random1825          EngressNet, 1825 random days from 2000-2020 (seed 0), test 2021, K = 20
#   results/FINAL2_<DS>_detmse_random1825   deterministic U-Net (det-MSE + EMA) on the same days
#   results/FINAL2_<DS>_detmse_seedens      seeds 1-9 of the same det-MSE model on the same days (seed 0 = the run above):
#                                           the 10-model seed-ensemble baseline
# Afterwards: evaluation/revisions/submit_final2_eval_chain.sh (truth fix, per-run tables, corrected metrics), then
# submit_final2_metrics.sh, figures/figures_v2/submit_make_v2_figures.sh, and observations/inference/submit_final2_obs_inference.py for the
# PIOMAS / ICESat-2 application (that one needs RANDOM_DAYS_EXCLUDE_YEARS=2021).
set -euo pipefail
cd "$(dirname "$0")"
WHICH="${1:-both}"; SUBMIT="${2:-}"
COMMON="DATA_VARIANT=avg,TRAIN_YEARS=2000-2020,TEST_YEARS=2021,TRAIN_RANDOM_DAYS=1825,RANDOM_DAYS_SEED=0,BETA=0.8,COARSE_ZERO=true"
ENG="$COMMON,NOISE_AMP_MAP=true,ENERGY_PATCH_SIZE=10,NOISE_LR_MULT=10,K=20,K_EVAL=20"
DET="$COMMON,DETERMINISTIC_MSE=true,K_EVAL=2"

run() {  # run <template> <jobname> <walltime> <vars>
    local cmd=(qsub -N "$2" -l "walltime=$3" -v "$4" "$1")
    echo "${cmd[*]}"
    if [ "$SUBMIT" = --submit ]; then "${cmd[@]}"; fi
}

for DS in MESA FOSI; do
    [ "$WHICH" = both ] || [ "$WHICH" = "$DS" ] || continue
    if [ "$DS" = MESA ]; then T=submit_engressnet_daily_mesa.sh; EMA="EMA_DECAY=0.9995"; L=M; WE=10:00:00; WD=02:00:00
    else T=submit_engressnet_daily.sh; EMA="EMA_DECAY=0.997"; L=F; WE=04:00:00; WD=01:00:00; fi
    run $T "F2eng_$L" $WE "$ENG,$EMA,SEED=0,BATCH_NAME=FINAL2_${DS}_random1825"
    run $T "F2det_$L" $WD "$DET,$EMA,SEED=0,BATCH_NAME=FINAL2_${DS}_detmse_random1825"
    for s in 1 2 3 4 5 6 7 8 9; do
        run $T "F2se_${L}_s$s" $WD "$DET,$EMA,SEED=$s,BATCH_NAME=FINAL2_${DS}_detmse_seedens"
    done
done
