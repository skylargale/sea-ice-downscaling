#!/bin/bash
# Revised final model (FINAL_SET=final2): D metrics (+ 10-model seed ensemble), IIEE check, Table 4 community values.
#PBS -N final2_metrics
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=150GB
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
cd /glade/work/skygale/projects/SeaIceDownscaling
export FINAL_SET=final2 NCPUS=4
for ds in MESA FOSI; do
  python -u evaluation/revisions/spatial_prob_metrics.py --dataset $ds --window 2000-2020 --seedwin FINAL2_${ds}_detmse_random1825,FINAL2_${ds}_detmse_seedens || echo "!!!!! D $ds failed"
done
python -u evaluation/revisions/iiee_check.py || echo "!!!!! iiee failed"
python -u evaluation/revisions/community_table4.py || echo "!!!!! table4 failed"
echo "METRICS DONE $(date)"
