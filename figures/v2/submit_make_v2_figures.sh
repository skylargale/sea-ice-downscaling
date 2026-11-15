#!/bin/bash
# Makes every figures/v2 figure: runs figures/v2/make_v2_figures.py top to bottom on a compute node (the map figures need
# ~64 GB). The model is FINAL_SET in the Setup cell of that file (final2 = paper figures, written to figures/v2/).
# Submit from figures/v2/:  qsub submit_make_v2_figures.sh
#PBS -N make_v2_figs
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=200GB
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/figures/v2
python -u make_v2_figures.py && echo "FIGURES DONE $(date)" || echo "!!!!! figures failed"
