#!/bin/bash
# Final round: after the sensitivity runs (training/submit_sensitivity_final2.sh) finish, restore the original truth in every
# SENS2 batch (evaluation/restore_raw_truth.py) and score all runs (sensitivity_final2_metrics.py, headline convention).
# Submit from evaluation/revisions/ (optionally with -W depend=afterany:<jobids>).
#PBS -N sens2_eval
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=200GB
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
export PYTHONPATH="$PWD/evaluation:${PYTHONPATH:-}"
python -u evaluation/restore_raw_truth.py results/SENS2_* || echo "!!!!! restore failed"
python -u evaluation/revisions/sensitivity_final2_metrics.py ${DATASETS:-FOSI MESA} || echo "!!!!! scoring failed"
echo "SENS2 EVAL DONE $(date)"
