#!/bin/bash
# ==============================================================
# submit_stochastic_refine_seed_variance.sh
#
# Repeated-seed variance check for the recommended config
# (STOCHASTIC_REFINE=true, NOISE_SIGMA=1.0), addressing the
# recommended_config.md caveat that run-to-run variance was never
# quantified (only inferred indirectly from two independent EnScaleNet
# sweeps differing ~1.5% in RMSE). Same single split already used for the
# recommended config's original evidence (2000-2005->2021, interp, medium
# domain) so results are directly comparable to what's already in the
# table -- SEED=0 already exists (FOSI_enscale_lite_smooth/
# MESA_enscale_lite_smooth, not resubmitted here), this adds SEED=1..4 to
# make a 5-seed sample. Uses the SEED passthrough added to
# submit_engressnet_daily.sh/_mesa.sh in this same session.
#
# 4 new seeds, FOSI only -- MESA twin is
# submit_stochastic_refine_seed_variance_mesa.sh.
#
# Usage:
#   ./submit_stochastic_refine_seed_variance.sh              # dry run
#   ./submit_stochastic_refine_seed_variance.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

SEEDS=(1 2 3 4)
BATCH_NAME="FOSI_stochastic_refine_seed_variance"
COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TRAIN_YEARS=2000-2005,TEST_YEARS=2021,STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0"

n_jobs=0
for seed in "${SEEDS[@]}"; do
    echo "qsub -N FOSI_refine_seed${seed} -v ${COMMON},SEED=${seed} submit_engressnet_daily.sh"
    if [ "$SUBMIT" = true ]; then
        qsub -N "FOSI_refine_seed${seed}" -v "${COMMON},SEED=${seed}" submit_engressnet_daily.sh
    fi
    n_jobs=$((n_jobs + 1))
done

echo ""
echo "Total jobs: ${n_jobs}"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub these)"
fi
