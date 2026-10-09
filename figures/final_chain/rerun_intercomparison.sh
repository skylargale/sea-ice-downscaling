#!/bin/bash
#PBS -N intercomp_nb
#PBS -l select=1:ncpus=4:mem=128GB
#PBS -l walltime=03:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
module load conda
conda activate seaice-downscaling
O=/glade/work/skygale/projects/SeaIceDownscaling/observations
cd $O
python notebook_src/py2nb.py notebook_src/observed_sit_intercomparison_src.py observed_sit_intercomparison.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=seaice-downscaling \
    --ExecutePreprocessor.timeout=14400 observed_sit_intercomparison.ipynb || echo "!!!!! intercomparison notebook failed"
echo "INTERCOMPARISON DONE $(date)"
