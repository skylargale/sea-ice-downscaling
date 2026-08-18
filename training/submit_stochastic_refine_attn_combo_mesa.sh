#!/bin/bash
# ==============================================================
# submit_stochastic_refine_attn_combo_mesa.sh
#
# MESA-HR twin of submit_stochastic_refine_attn_combo.sh: single-split
# viability check for STOCHASTIC_REFINE=true + ATTENTION_END=true together.
#
# 1 job, into results/MESA_stochastic_refine_attn_combo/.
#
# Usage:
#   ./submit_stochastic_refine_attn_combo_mesa.sh              # dry run
#   ./submit_stochastic_refine_attn_combo_mesa.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

BATCH_NAME="MESA_stochastic_refine_attn_combo"
COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TRAIN_YEARS=2000-2005,TEST_YEARS=2021,STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0,ATTENTION_END=true"

echo "qsub -N MESA_refine_attn -v ${COMMON} submit_engressnet_daily_mesa.sh"
if [ "$SUBMIT" = true ]; then
    qsub -N "MESA_refine_attn" -v "${COMMON}" submit_engressnet_daily_mesa.sh
fi

echo ""
echo "Total jobs: 1"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub this)"
fi
