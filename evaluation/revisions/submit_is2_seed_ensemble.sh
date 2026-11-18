#!/bin/bash
# Scores EngressNet vs the seed ensemble on the ICESat-2 testbed (is2_seed_ensemble.py). Submit from evaluation/ with
# -W depend=afterok:<seedens_cells job> (observations/inference/submit_seedens_obs_inference.py).
#PBS -N is2_seedens
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=40GB
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
python -u evaluation/revisions/is2_seed_ensemble.py && echo "IS2 SEEDENS DONE $(date)" || echo "!!!!! is2_seed_ensemble failed"
