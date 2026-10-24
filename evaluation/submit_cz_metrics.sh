#!/bin/bash
# After apply_coarse_zero_fields.py: recompute the corrected ocean metrics / spread-skill CSVs, the ensemble-mean sweep summary
# (20-member K=20 rows are the headline numbers, 2026-10-24), and the regime / rank-histogram tables for the final batches; email the summary.
#PBS -N cz_metrics
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=2:mem=200GB
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o .logs/

set -uo pipefail
module load conda
conda activate seaice-downscaling
V6=/glade/work/skygale/projects/SeaIceDownscaling
cd $V6
C=evaluation/corrected_calibration
B="results/FOSI_final_ema results/MESA_final_ema results/FOSI_detmse_ema results/MESA_detmse_ema"
python -u evaluation/recompute_ocean_metrics.py --out-dir $C $B || echo "!!!!! recompute_ocean_metrics failed"
python -u evaluation/recompute_spread_skill.py --out-dir $C $B || echo "!!!!! recompute_spread_skill failed"
python -u evaluation/summarize_ensmean_sweep.py > /dev/null || echo "!!!!! summarize failed"
for D in FOSI MESA; do
    python -u evaluation/compare_output_floor.py results/${D}_final_ema > $C/coarse_zero_final_${D}_regimes.txt 2>&1 || echo "!!!!! compare $D failed"
    PYTHONPATH=evaluation python -u evaluation/rank_hist_breakdown.py results/${D}_final_ema > $C/coarse_zero_final_${D}_rank_hist.txt 2>&1 || echo "!!!!! rank hist $D failed"
done
cat $C/coarse_zero_final_FOSI_regimes.txt $C/coarse_zero_final_FOSI_rank_hist.txt $C/coarse_zero_final_MESA_regimes.txt \
    $C/coarse_zero_final_MESA_rank_hist.txt $C/ensmean_sweep_summary.txt > $C/coarse_zero_final_summary.txt
for a in ${MAIL_TO:-}; do  # qsub -v MAIL_TO="a@b c@d" to email the summary
    mailx -s "EngressNet coarse-zero (output rule): perfect-model metrics" "$a" < $C/coarse_zero_final_summary.txt
done
echo "CZ METRICS DONE $(date)"
