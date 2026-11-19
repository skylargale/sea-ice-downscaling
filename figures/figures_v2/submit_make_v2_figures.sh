#!/bin/bash
# Makes every figures/figures_v2 figure: builds the figure data (build_figure_data.py), then runs make_v2_figures.py top to bottom on a compute node (the map figures need
# ~64 GB). The model is FINAL_SET in the Setup cell of that file (final2 = paper figures, written to figures/figures_v2/).
# Submit from figures/figures_v2/:  qsub submit_make_v2_figures.sh
#PBS -N make_v2_figs
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=200GB
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling/figures/figures_v2
python -u build_figure_data.py && (cd ../.. && python -u evaluation/revisions/sit_type_metrics.py && python -u evaluation/revisions/psd_dates.py) && python -u make_v2_figures.py && echo "FIGURES DONE $(date)" || echo "!!!!! figures failed"
