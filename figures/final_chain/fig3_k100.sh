#!/bin/bash
#PBS -N fig3_k100
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=200GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/figures
python -u build_fig3_example_fields_k100.py
