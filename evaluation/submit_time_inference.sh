#!/bin/bash
#PBS -N time_inference
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB:gpu_type=v100
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o .logs/time_inference.OU
#PBS -m n

set -euo pipefail
module load conda
conda activate seaice-downscaling
export PYTHONPATH="/glade/work/skygale/projects/SeaIceDownscaling/Version6/evaluation:${PYTHONPATH:-}"
cd /glade/work/skygale/projects/SeaIceDownscaling/Version6/evaluation
python time_inference.py
