#!/bin/bash
#PBS -N fig7_final
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=96GB
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/figures
python -u build_fig7_final_piomas_vs_icesat2.py
