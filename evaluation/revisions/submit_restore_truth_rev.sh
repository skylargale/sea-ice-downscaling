#!/bin/bash
# Revisions 2026-11: original-truth fix (evaluation/restore_raw_truth.py) on every new REV_* perfect-model batch, 6 batches
# at a time. Backups of fields.npz go to restore_raw_truth.py's BACKUP (scratch outputs/backups/...).
#PBS -N rev_truth
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=300GB
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
ls -d results/REV_*/ | xargs -P 6 -I{} sh -c 'python -u evaluation/restore_raw_truth.py {} > logs/restore_$(basename {}).log 2>&1; echo "finished {}"'
grep -h -i -E "skip|differ|error" logs/restore_REV_*.log | head -40
echo "ALL DONE $(date)"
