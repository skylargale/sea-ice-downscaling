#!/bin/bash
# Builds the 2000-2021 conservative-truth targets (processing/build_conservative_2021.py). Submit from processing/.
#PBS -N cons2021
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=96GB
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/processing
python -u build_conservative_2021.py MESA FOSI && echo "CONS2021 DONE $(date)" || echo "!!!!! cons2021 failed"
