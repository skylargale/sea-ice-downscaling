#!/bin/bash
# Final-results chain (2026-10-07), part 1: evaluation outputs for the final MESA (and FOSI) batches, corrected calibration,
# then re-execute the paper figure notebook figures/all_figures_by_run.ipynb (RUN=MESA_final_avg) into final_figures/.
#PBS -N final_mesa_figs
#PBS -l select=1:ncpus=4:mem=160GB
#PBS -l walltime=08:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
set -uo pipefail
module load conda
conda activate seaice-downscaling
V6=/glade/work/skygale/projects/SeaIceDownscaling
cd $V6
for B in MESA_final_avg CAL1a_noiselr_FOSI; do
    python evaluation/run_daily_eval_batch.py --batch-dir results/$B || echo "!!!!! daily eval failed for $B"
    python -u evaluation/build_4window_aggregate_figures.py --batch-dir results/$B --save-dir evaluation/saved_figs/$B/_4window_aggregate || echo "!!!!! aggregate failed for $B"
done
python -u evaluation/recompute_spread_skill.py --out-dir evaluation/corrected_calibration results/MESA_final_avg results/MESA_deterministic_mse_avg results/FOSI_deterministic_mse_avg
cd $V6/figures
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=seaice-downscaling \
    --ExecutePreprocessor.timeout=14400 all_figures_by_run.ipynb || echo "!!!!! paper figure notebook failed"
echo "FINAL MESA EVAL + FIGURES DONE $(date)"
