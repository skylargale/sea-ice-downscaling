#!/bin/bash
#PBS -N final2_time
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:ngpus=1:mem=64GB:gpu_type=v100
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
export PYTHONPATH="/glade/work/skygale/projects/SeaIceDownscaling/evaluation:${PYTHONPATH:-}"
cd /glade/work/skygale/projects/SeaIceDownscaling
python -u evaluation/revisions/time_inference_final2.py
