#!/bin/bash
# Revisions 2026-11: score all revision runs (summarize_rev_runs.py) for one experiment: qsub -v EXP=H revisions/submit_summarize_rev_runs.sh
#PBS -N rev_summ
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=150GB
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
python -u evaluation/revisions/summarize_rev_runs.py ${EXP:?}
