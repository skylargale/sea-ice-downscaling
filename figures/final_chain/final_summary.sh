#!/bin/bash
# Final-results chain (2026-10-07), part 3: the final results table.
#PBS -N final_summary
#PBS -l select=1:ncpus=2:mem=96GB
#PBS -l walltime=02:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
module load conda
conda activate seaice-downscaling
python -u /glade/work/skygale/projects/SeaIceDownscaling/figures/final_chain/final_results_summary.py
echo "FINAL SUMMARY DONE $(date)"
