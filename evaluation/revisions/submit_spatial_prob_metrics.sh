#!/bin/bash
# Revisions 2026-11, D (+E scoring): CPU job running evaluation/revisions/spatial_prob_metrics.py for one dataset/window.
#   qsub -v DATASET=MESA,WINDOW=2015-2020 evaluation/revisions/submit_spatial_prob_metrics.sh
# Extra ensembles (E baselines, H modes) once their batches exist, e.g.
#   qsub -v DATASET=MESA,WINDOW=2015-2020,SKIP_CORE=true,EXTRA="MC-dropout U-Net=REV_E2_MESA_mcdropout;EngressNet (mult noise)=REV_H_MESA_mult",SEEDWIN=MESA_detmse_ema:REV_E1_MESA_detmse_seed1:REV_E1_MESA_detmse_seed2 ...
# (EXTRA entries separated by ';', SEEDWIN batches by ':' -- qsub -v splits on commas.)
#PBS -N D_metrics
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=120GB
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
set -uo pipefail
V6=/glade/work/skygale/projects/SeaIceDownscaling
module load conda
conda activate seaice-downscaling
export NCPUS=4
ARGS=(--dataset "${DATASET:?}" --window "${WINDOW:?}")
[ "${SKIP_CORE:-false}" = true ] && ARGS+=(--skip-core)
if [ -n "${EXTRA:-}" ]; then
    IFS=';' read -ra EX <<< "$EXTRA"
    for e in "${EX[@]}"; do ARGS+=(--extra "$e"); done
fi
[ -n "${SEEDWIN:-}" ] && ARGS+=(--seedwin "${SEEDWIN//:/,}")
[ -n "${TAG:-}" ] && ARGS+=(--tag "$TAG")
echo "start $(date) | ${ARGS[*]}"
python -u $V6/evaluation/revisions/spatial_prob_metrics.py "${ARGS[@]}"
echo "done $(date)"
