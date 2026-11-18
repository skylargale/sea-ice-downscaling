#!/bin/bash
# Scores the K-sweep re-inference (k_sweep.py). Submit from evaluation/ with -W depend=afterok:<KSWEEP jobs>.
#PBS -N k_sweep
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=300GB
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
python -u evaluation/revisions/k_sweep.py && echo "K SWEEP DONE $(date)" || echo "!!!!! k_sweep failed"
