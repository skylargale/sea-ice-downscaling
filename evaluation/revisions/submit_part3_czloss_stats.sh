#!/bin/bash
# Final round part 3, item 5: restore the raw truth of the coarse-zero-in-loss run (results/P3_FOSI_czloss) and score it with
# part3_perfect_model_stats.py next to the final model. Submit from evaluation/ with -W depend=afterok:<training job>.
#PBS -N p3_czstats
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=64GB
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
python -u evaluation/restore_raw_truth.py results/P3_FOSI_czloss || echo "!!!!! restore failed"
RUN=$(ls -d results/P3_FOSI_czloss/*/ | head -1)
python -u evaluation/revisions/part3_perfect_model_stats.py \
    "FOSI final (zero rule at inference only)=results/FINAL2_FOSI_random1825/F2eng_F_2000-2020_2021_6666801.casper-pbs" \
    "FOSI zero rule in the loss too=${RUN%/}" > results/revisions_2026-11/part3_czloss_stats.txt 2>&1 || echo "!!!!! stats failed"
cat results/revisions_2026-11/part3_czloss_stats.txt
echo "P3 CZ STATS DONE $(date)"
