#!/bin/bash
# 2026-10-08 rerun: seasonal-cycle CSV (was missing), paper figure notebook, final summary table.
#PBS -N final_rerun
#PBS -l select=1:ncpus=4:mem=160GB
#PBS -l walltime=04:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
set -uo pipefail
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/figures
python -u build_piomas_fosi_mesa_seasonal_cycle.py || echo "!!!!! seasonal cycle failed"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=seaice-downscaling \
    --ExecutePreprocessor.timeout=14400 all_figures_by_run.ipynb || echo "!!!!! paper figure notebook failed"
python -u final_chain/final_results_summary.py || echo "!!!!! summary failed"
echo "FINAL RERUN DONE $(date)"
