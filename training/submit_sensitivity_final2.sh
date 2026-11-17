#!/bin/bash
# Final round (2026-11): Appendix A sensitivity tests redone on the final design (FINAL2). Every run is the FINAL2 EngressNet
# configuration with ONE setting changed. Run it directly (not with qsub) from anywhere:
#   bash submit_sensitivity_final2.sh [FOSI|MESA|both] [phase1|topup|only:<config>] [--submit]
# Without --submit it only prints the qsub commands. Batches: results/SENS2_<DS>_<config>/ (seed in the run name).
#
# phase1: FOSI every config x seeds 0-2, final config seeds 1-9 (seed 0 = results/FINAL2_FOSI_random1825);
#         MESA every config x seed 0, final config seeds 1-4 (seed 0 = results/FINAL2_MESA_random1825).
# only:<config>: seeds 0-2 of one config on both datasets (rerun of a failed config, e.g. only:attn).
# topup:  MESA seeds 1-2 for the configs listed in TOPUP (filled in after phase 1, for rows inside the MESA seed band).
#
# Configs that need code not in functions_engressnet.py yet are marked [new flag]; they are added default-off and checked
# bit-identical on the final-model flags before anything is submitted. "conservative truth" needs the 2021 conservative
# targets (the existing conservative_targets/ files are bilinear in 2021); built first by processing/build_conservative_2021.py.
set -euo pipefail
cd "$(dirname "$0")"
WHICH="${1:-both}"; PHASE="${2:-phase1}"; SUBMIT="${3:-}"
TD=/glade/derecho/scratch/skygale/downscaling/training_data
FINAL="DATA_VARIANT=avg,TRAIN_YEARS=2000-2020,TEST_YEARS=2021,TRAIN_RANDOM_DAYS=1825,RANDOM_DAYS_SEED=0,BETA=0.8,COARSE_ZERO=true"
FINAL="$FINAL,NOISE_AMP_MAP=true,ENERGY_PATCH_SIZE=10,NOISE_LR_MULT=10,K=20,K_EVAL=20"

# name | change from the final configuration (env vars; later values override FINAL) | walltime factor
CONFIGS=(
  "interp|DATA_VARIANT=interp|1"
  "land05|LAND_THRESHOLD=0.5|1"                                    # [new flag] ocean-fraction cutoff 0.1 -> 0.5
  "construth|Y_PATH_OVERRIDE=@CONS@|1"                             # conservative truth (train + test)
  "season|MONTHS=3-7|1"                                            # Mar-Jul only; 1,825 days drawn from Mar-Jul 2000-2020 [new: draw within season]
  "bigdomain|LAT_MIN=60,LAT_MAX=75,LON_MIN=-190,LON_MAX=-143,GRAD_ACCUM_STEPS=2|2"  # 16 x 48 coarse cells (190-143W, 1.5x the area; scored on the paper domain)
  "depth5|ENCODER_DEPTH=5|1.3"                                     # [new flag] +1 encoder/decoder block
  "attn|ATTENTION_END=true,GRAD_ACCUM_STEPS=4|2.5"                 # [new flag] end-of-decoder windowed self-attention (restored); first try (no grad-accum) ran out of V100 memory at step 1 -> same batch in 4 chunks
  "uncorr|NOISE_KERNEL_SIZE=1|1"                                   # no spatially correlated noise (1x1 kernel = white noise)
  "globalz|GLOBAL_NOISE=true|1"                                    # [new flag] one noise field shared by all stages
  "latefusion|LATE_MASK_FUSION=true|1"
  "nch2|NOISE_CHANNELS=2|1"
  "k3|NOISE_KERNEL_SIZE=3|1"
  "k9|NOISE_KERNEL_SIZE=9|1"
  "tile30|ENERGY_PATCH_SIZE=30|1"
  "tile15|ENERGY_PATCH_SIZE=15|1"
  "tile5|ENERGY_PATCH_SIZE=5|1"
  "noampmap|NOISE_AMP_MAP=false|1"
  "nlr1|NOISE_LR_MULT=1|1"
  "noema|EMA_DECAY=0|1"
)
# 2026-11-16 (first top-up, jobs 6687043-72): TOPUP=(bigdomain construth depth5 globalz k3 k9 land05 nch2 nlr1 noampmap season tile15 tile30 tile5 uncorr), the MESA configs whose phase-1 RMSE or CRPS differed from the 5-seed final mean by less than the final seed range
TOPUP=(interp latefusion noema)   # 2026-11-16 second top-up (user request): the remaining single-seed MESA rows, so every row has 3 seeds

run() {  # run <template> <jobname> <walltime> <vars>
    local cmd=(qsub -N "$2" -l "walltime=$3" -v "$4" "$1")
    echo "${cmd[*]}"
    if [ "$SUBMIT" = --submit ]; then "${cmd[@]}"; fi
}
hours() { python3 -c "import math;print(f'{math.ceil($1*$2):02d}:00:00')"; }
merge() {  # merge "<A=1,B=2>" "<B=3,C=4>" ... -> one list, each variable once, later values win (qsub -v keeps no order promise)
    python3 -c "import sys;d={}
for a in sys.argv[1:]:
    for kv in filter(None, a.split(',')): k,v=kv.split('=',1); d.pop(k,None); d[k]=v
print(','.join(f'{k}={v}' for k,v in d.items()))" "$@"
}

n=0
for DS in FOSI MESA; do
    [ "$WHICH" = both ] || [ "$WHICH" = "$DS" ] || continue
    if [ "$DS" = MESA ]; then T=submit_engressnet_daily_mesa.sh; EMA="EMA_DECAY=0.9995"; L=M; BASE_H=8
        CONS=$TD/conservative_targets/Y_MESA_HR_daily_conservative_2000_2021.nc
        if [[ "$PHASE" == only:* ]]; then SEEDS_CFG=(0 1 2); SEEDS_FINAL=(); CFG_LIST=("${PHASE#only:}")
        elif [ "$PHASE" = topup ]; then SEEDS_CFG=(1 2); SEEDS_FINAL=(); CFG_LIST=("${TOPUP[@]}"); else SEEDS_CFG=(0); SEEDS_FINAL=(1 2 3 4); CFG_LIST=(); fi
    else T=submit_engressnet_daily.sh; EMA="EMA_DECAY=0.997"; L=F; BASE_H=2
        CONS=$TD/conservative_targets/Y_FOSI_HR_JRA55_daily_conservative_2000_2021.nc
        [ "$PHASE" = topup ] && continue
        SEEDS_CFG=(0 1 2); SEEDS_FINAL=(1 2 3 4 5 6 7 8 9); CFG_LIST=()
        if [[ "$PHASE" == only:* ]]; then SEEDS_FINAL=(); CFG_LIST=("${PHASE#only:}"); fi
    fi
    for s in "${SEEDS_FINAL[@]}"; do
        run $T "S2${L}_final_s$s" "$(hours $BASE_H 1)" "$(merge "$FINAL" "$EMA" "SEED=$s,BATCH_NAME=SENS2_${DS}_final")"; n=$((n+1))
    done
    for c in "${CONFIGS[@]}"; do
        IFS='|' read -r name change fac <<< "$c"
        if [ "$PHASE" != phase1 ] && [[ ! " ${CFG_LIST[*]} " =~ " $name " ]]; then continue; fi
        change="${change//@CONS@/$CONS}"
        for s in "${SEEDS_CFG[@]}"; do
            run $T "S2${L}_${name}_s$s" "$(hours $BASE_H $fac)" "$(merge "$FINAL" "$EMA" "$change" "SEED=$s,BATCH_NAME=SENS2_${DS}_${name}")"; n=$((n+1))
        done
    done
done
echo "# $n jobs"
