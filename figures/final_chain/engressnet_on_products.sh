#!/bin/bash
#PBS -N eng_on_prod
#PBS -l select=1:ncpus=2:mem=96GB
#PBS -l walltime=02:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/figures
python -u build_engressnet_on_products.py
