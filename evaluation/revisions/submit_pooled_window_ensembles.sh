#!/bin/bash
#PBS -N pooled_ens
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=150GB
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
python -u evaluation/revisions/pooled_window_ensembles.py
