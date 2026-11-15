#!/bin/bash
# 2026-11 revised final model (1825 random days 2000-2020, seed 0): perfect-model evaluation chain for one dataset.
#   qsub -v DS=FOSI revisions/submit_final2_eval_chain.sh        (DS=FOSI or MESA)
# 1) original-truth fix on the new batches  2) run_daily_eval_batch.py (per-run saved_figs tables)
# 3) build_4window_aggregate_figures.py on the EngressNet batch (single run; Fig. 6/7 CSVs under saved_figs/<batch>/_4window_aggregate)
# 4) recompute_spread_skill.py + recompute_ocean_metrics.py -> evaluation/corrected_calibration/<batch>{,_ocean_metrics}.csv
#PBS -N final2_eval
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=200GB
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
set -uo pipefail
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
DS=${DS:?}
ENG=results/FINAL2_${DS}_random1825; DET=results/FINAL2_${DS}_detmse_random1825; SE=results/FINAL2_${DS}_detmse_seedens
echo "start $(date) $DS"
for b in $ENG $DET $SE; do python -u evaluation/restore_raw_truth.py $b || echo "!!!!! restore failed $b"; done
for b in $ENG $DET; do python -u evaluation/run_daily_eval_batch.py --batch-dir $b || echo "!!!!! daily eval failed $b"; done
python -u evaluation/build_4window_aggregate_figures.py --batch-dir $ENG --save-dir evaluation/saved_figs/$(basename $ENG)/_4window_aggregate || echo "!!!!! aggregate failed"
python -u evaluation/recompute_spread_skill.py --out-dir evaluation/corrected_calibration $ENG $DET || echo "!!!!! spread-skill failed"
python -u evaluation/recompute_ocean_metrics.py --out-dir evaluation/corrected_calibration $ENG $DET || echo "!!!!! ocean metrics failed"
echo "CHAIN DONE $(date)"
