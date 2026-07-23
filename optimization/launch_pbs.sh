#!/bin/bash -l
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_type=a100_80gb
#PBS -l walltime=04:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -N echo_engressnet
cd "$PBS_O_WORKDIR"
module load conda
conda activate downscaling_env
echo-run hyperparameters.yml model_config.yml -n $PBS_JOBID
