#!/bin/bash
# ==============================================================
# submit_daily_combo_sweep.sh
#
# Full-factorial extension of submit_daily_sweep.sh: sweeps extra_layer,
# domain size, and enscale-lite (stochastic_refine) on/off in every
# combination, on top of the same 4 training windows x 2 data variants,
# all tested on 2021.
#
# Axes:
#   extra_layer   : off / on            (--extra-layer, 4th UNet stage)
#   domain        : medium / tiny       (16x32 vs 16x16 low-res crop)
#   enscale_lite  : off / on            (--stochastic-refine, noise_sigma=1.0 default)
#   train window  : 2000-2005, 2005-2010, 2010-2015, 2015-2020 (test: 2021)
#   data variant  : interp, avg
#
# 2 x 2 x 2 x 4 x 2 = 64 minus 16 infeasible (see below) = 48 jobs.
#
# Domain definitions (low-res grid, 1 deg/cell on the daily FOSI grid):
#   medium: lat 60-75, lon -182 to -151  -> 16 x 32  (matches FOSI_2conv baseline)
#   tiny:   lat 60-75, lon -172 to -157  -> 16 x 16  (half the area; still
#           covers Kivalina/Shishmaref/Kotzebue/Nome/Point Hope)
#   NOTE: the original monthly-sweep "tiny" box (lat62-69/lon-172to-157,
#   8x16) is NOT reused here -- it crops to lr_h=8, which is below the
#   minimum-16 requirement extract_full_domain() enforces when
#   extra_layer=True, and would crash every extra_layer x tiny-domain arm.
#
#   extra_layer x tiny is SKIPPED entirely (not just avoided by choice of
#   box): extract_full_domain() requires not just "both dims a multiple of
#   the divisor" but also that the bottleneck (lr_h/divisor, lr_w/divisor)
#   have MORE THAN ONE spatial element (InstanceNorm2d can't compute a
#   variance over a single point). With divisor=16 (extra_layer=True), a
#   16x16 crop bottlenecks to exactly (1, 1) -- one pixel -- so it fails
#   that second check no matter how the box is chosen, unless one dimension
#   is widened to 32+, at which point it's no longer smaller than the
#   medium (16x32) domain. There is no "tiny" domain distinct from medium
#   that's simultaneously extra_layer-compatible at this grid resolution --
#   found the hard way (2026-08-06): 16 jobs (8 interp, 8 avg) were
#   submitted expecting this to work, failed/were cancelled, and this loop
#   was fixed to skip the combination instead of resubmitting it.
#
# Each arm is a single run (no repeats). Batches land in
# results/FOSI_daily_combo_<variant>/, run tag encodes the arm via PBS job
# name (FOSI_el<0|1>_d<med|tiny>_es<0|1>) plus train/test years/job id.
#
# Usage:
#   ./submit_daily_combo_sweep.sh              # dry run: prints every qsub command
#   ./submit_daily_combo_sweep.sh --submit      # actually submits all jobs
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

SPLITS=(
    "2000-2005"
    "2005-2010"
    "2010-2015"
    "2015-2020"
)
TEST_YEARS="2021"
VARIANTS=("interp" "avg")
EXTRA_LAYER_VALUES=("false" "true")
ENSCALE_VALUES=("false" "true")
# domain_name:LAT_MIN:LAT_MAX:LON_MIN:LON_MAX
DOMAINS=(
    "med:60:75:-182:-151"
    "tiny:60:75:-172:-157"
)

# Baseline (FOSI_2conv) values, held fixed across every arm.
BASE_NUM_EPOCHS=20
BASE_K=20
BASE_K_EVAL=20
BASE_BETA=0.8

n_jobs=0

submit_job () {
    # $1 = PBS job name, $2 = comma-separated NAME=VALUE PBS -v pairs
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily.sh
    fi
    n_jobs=$((n_jobs + 1))
}

for variant in "${VARIANTS[@]}"; do
    batch_name="FOSI_daily_combo_${variant}"
    for domain_spec in "${DOMAINS[@]}"; do
        IFS=":" read -r dname lat_min lat_max lon_min lon_max <<< "$domain_spec"
        for extra_layer in "${EXTRA_LAYER_VALUES[@]}"; do
            el_tag=$([ "$extra_layer" = true ] && echo "1" || echo "0")
            if [ "$extra_layer" = true ] && [ "$dname" = tiny ]; then
                echo "# skipping extra_layer=true x domain=tiny (16x16 bottlenecks to 1x1, InstanceNorm2d rejects it) -- see header comment"
                continue
            fi
            for enscale in "${ENSCALE_VALUES[@]}"; do
                es_tag=$([ "$enscale" = true ] && echo "1" || echo "0")
                job_name="FOSI_el${el_tag}_d${dname}_es${es_tag}"
                for train_years in "${SPLITS[@]}"; do
                    submit_job "$job_name" \
                        "BATCH_NAME=${batch_name},DATA_VARIANT=${variant},TRAIN_YEARS=${train_years},TEST_YEARS=${TEST_YEARS},NUM_EPOCHS=${BASE_NUM_EPOCHS},K=${BASE_K},K_EVAL=${BASE_K_EVAL},BETA=${BASE_BETA},LAT_MIN=${lat_min},LAT_MAX=${lat_max},LON_MIN=${lon_min},LON_MAX=${lon_max},EXTRA_LAYER=${extra_layer},STOCHASTIC_REFINE=${enscale}"
                done
            done
        done
    done
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
