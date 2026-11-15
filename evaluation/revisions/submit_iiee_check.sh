#!/bin/bash
# Revisions 2026-11 follow-up 2 (IIEE fairness / bilinear) -- CPU job. qsub evaluation/revisions/submit_iiee_check.sh
#PBS -N iiee_check
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=150GB
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
python -u evaluation/revisions/iiee_check.py
