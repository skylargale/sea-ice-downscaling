#!/bin/bash
# Final-results chain (2026-10-07), part 2: re-extract model data from the final batches (perfect model FOSI/MESA 2021,
# PIOMAS 2010-2023 application, along-track matchups) and re-execute observations/observed_sit_intercomparison.ipynb.
#PBS -N final_piomas_nb
#PBS -l select=1:ncpus=4:mem=160GB
#PBS -l walltime=08:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
set -uo pipefail
module load conda
conda activate seaice-downscaling
O=/glade/work/skygale/projects/SeaIceDownscaling/observations
cd $O/processing
python -u extract_models.py perfect piomas_app alongtrack_match || echo "!!!!! extract_models failed"
cd $O
python notebook_src/py2nb.py notebook_src/observed_sit_intercomparison_src.py observed_sit_intercomparison.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=seaice-downscaling \
    --ExecutePreprocessor.timeout=14400 observed_sit_intercomparison.ipynb || echo "!!!!! intercomparison notebook failed"
echo "FINAL PIOMAS EXTRACT + NOTEBOOK DONE $(date)"
