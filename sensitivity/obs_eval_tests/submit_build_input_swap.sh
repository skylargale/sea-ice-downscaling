#!/bin/bash
# usage: qsub -v STEPS="topaz4ml c2s_smos petty_gpsat" submit_build_input_swap.sh
#PBS -N build_swap
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=96GB
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/sensitivity_logs/
set -euo pipefail
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/sensitivity/obs_eval_tests
python build_input_swap.py ${STEPS}
