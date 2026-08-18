#!/bin/bash
# ==============================================================
# submit_stochastic_refine_seed_variance_mesa.sh
#
# MESA-HR twin of submit_stochastic_refine_seed_variance.sh: SEED=1..4 at
# the recommended config's original single split (2000-2005->2021, interp).
# SEED=0 already exists (MESA_enscale_lite_smooth).
#
# 4 new seeds, into results/MESA_stochastic_refine_seed_variance/.
#
# Usage:
#   ./submit_stochastic_refine_seed_variance_mesa.sh              # dry run
#   ./submit_stochastic_refine_seed_variance_mesa.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

SEEDS=(1 2 3 4)
BATCH_NAME="MESA_stochastic_refine_seed_variance"
COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TRAIN_YEARS=2000-2005,TEST_YEARS=2021,STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0"

n_jobs=0
for seed in "${SEEDS[@]}"; do
    echo "qsub -N MESA_refine_seed${seed} -v ${COMMON},SEED=${seed} submit_engressnet_daily_mesa.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "MESA_refine_seed${seed}" -v "${COMMON},SEED=${seed}" submit_engressnet_daily_mesa.sh
    fi
    n_jobs=$((n_jobs + 1))
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
