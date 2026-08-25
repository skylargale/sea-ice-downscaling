#!/bin/bash
# ==============================================================
# submit_daily_combo_sweep_mesa.sh
#
# MESA-HR twin of submit_daily_combo_sweep.sh: same full-factorial sweep
# (extra_layer x domain x enscale_lite x 4 train windows x 2 data variants
# = 64 jobs), but against the stitched HIST+RCP8.5 daily MESA data instead
# of FOSI. Requires X_MESA_HR_daily_{interp,avg}.nc / Y_MESA_HR_daily.nc to
# exist first -- run, in order:
#   1. processing/submit_build_X_Y_from_MESA-HR_daily_rcp85.sh
#   2. processing/submit_stitch_MESA_HR_daily.sh
# before this script.
#
# Axes (identical definitions to the FOSI combo sweep -- see that script for
# the domain-sizing derivation):
#   extra_layer   : off / on
#   domain        : medium (16x32, lat60-75/lon-182to-151) /
#                   tiny   (16x16, lat60-75/lon-172to-157)
#   enscale_lite  : off / on
#   train window  : 2000-2005, 2005-2010, 2010-2015, 2015-2020 (test: 2021)
#   data variant  : interp, avg
#
# 2 x 2 x 2 x 4 x 2 = 64 minus 16 infeasible = 48 jobs. Batches land in
# results/MESA_daily_combo_<variant>/.
#
# extra_layer x tiny is SKIPPED (not just avoided by box choice): the tiny
# domain (16x16) bottlenecks to exactly 1x1 under extra_layer's divisor=16,
# which InstanceNorm2d rejects (can't compute a variance over one point) --
# there's no "tiny" box distinct from medium (16x32) that's simultaneously
# extra_layer-compatible at this grid resolution. Found the hard way against
# the FOSI version of this sweep (2026-08-06, see submit_daily_combo_sweep.sh)
# before this MESA sweep was ever submitted -- fixed here from the start.
#
# Usage:
#   ./submit_daily_combo_sweep_mesa.sh              # dry run
#   ./submit_daily_combo_sweep_mesa.sh --submit      # actually submit
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
DOMAINS=(
    "med:60:75:-182:-151"
    "tiny:60:75:-172:-157"
)

BASE_NUM_EPOCHS=20
BASE_K=20
BASE_K_EVAL=20
BASE_BETA=0.8

n_jobs=0

submit_job () {
    local job_name="$1"; shift
    local vlist="$1"; shift
    echo "qsub -N ${job_name} -v ${vlist} submit_engressnet_daily_mesa.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "${job_name}" -v "${vlist}" submit_engressnet_daily_mesa.sh
    fi
    n_jobs=$((n_jobs + 1))
}

for variant in "${VARIANTS[@]}"; do
    batch_name="MESA_daily_combo_${variant}"
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
                job_name="MESA_el${el_tag}_d${dname}_es${es_tag}"
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
