#!/bin/bash
# ==============================================================
# submit_stochastic_refine_attn_combo.sh
#
# Viability check: STOCHASTIC_REFINE=true + ATTENTION_END=true together.
# Not mutually exclusive in functions_engressnet.py (only enscale_net is
# mutually exclusive with extra_layer/stochastic_refine) -- untested
# combination of the two toggles that each independently improved
# calibration (Spread/Error) through different mechanisms: the refiner via
# spatially-coherent per-location noise mixing, attention-end via windowed
# self-attention at the decoder end (see recommended_config.md's
# `MESA_attn_end_avg` writeup for the attention-end calibration signal).
# Single split first (2000-2005->2021, interp, medium domain) to check the
# combo trains cleanly and shows a real signal before committing to a full
# 4-window sweep -- MESA twin is
# submit_stochastic_refine_attn_combo_mesa.sh.
#
# 1 job, into results/FOSI_stochastic_refine_attn_combo/.
#
# Usage:
#   ./submit_stochastic_refine_attn_combo.sh              # dry run
#   ./submit_stochastic_refine_attn_combo.sh --submit      # actually submit
# ==============================================================

set -euo pipefail
cd "$(dirname "$0")"

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
fi

mkdir -p logs

BATCH_NAME="FOSI_stochastic_refine_attn_combo"
COMMON="BATCH_NAME=${BATCH_NAME},DATA_VARIANT=interp,TRAIN_YEARS=2000-2005,TEST_YEARS=2021,STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0,ATTENTION_END=true"

echo "qsub -N FOSI_refine_attn -v ${COMMON} submit_engressnet_daily.sh"
if [ "$SUBMIT" = true ]; then
    qsub -N "FOSI_refine_attn" -v "${COMMON}" submit_engressnet_daily.sh
fi

echo ""
echo "Total jobs: 1"
if [ "$SUBMIT" = false ]; then
    echo "(dry run -- rerun with --submit to actually qsub this)"
fi
