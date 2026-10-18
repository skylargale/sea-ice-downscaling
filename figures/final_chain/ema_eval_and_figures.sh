#!/bin/bash
# 2026-10-18 EMA model: daily evaluation outputs + 4-window aggregate figures (rank histograms, spread-skill and CRPS by regime,
# PSD, community time series) for the adopted EngressNet + EMA batches and the matched det-MSE + EMA baseline.
# Same steps as final_mesa_eval_and_figures.sh (2026-10-07), without the paper-figure notebook (re-run that once RUN is switched).
#PBS -N ema_eval_figs
#PBS -l select=1:ncpus=4:mem=200GB
#PBS -l walltime=10:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
set -uo pipefail
module load conda
conda activate seaice-downscaling
V6=/glade/work/skygale/projects/SeaIceDownscaling
cd $V6
for B in MESA_final_ema FOSI_final_ema MESA_detmse_ema FOSI_detmse_ema; do
    python evaluation/run_daily_eval_batch.py --batch-dir results/$B || echo "!!!!! daily eval failed for $B"
done
for B in MESA_final_ema FOSI_final_ema; do
    python -u evaluation/build_4window_aggregate_figures.py --batch-dir results/$B --save-dir evaluation/saved_figs/$B/_4window_aggregate || echo "!!!!! aggregate failed for $B"
done
echo "EMA EVAL + FIGURES DONE $(date)"
