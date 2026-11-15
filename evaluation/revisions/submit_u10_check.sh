#!/bin/bash
# Revisions 2026-11, C.2 follow-up: does corner-based conservative U10 change the downscaled SIT? No retraining.
# 1) build the 2021 X file with conservative U10 (build_u10_conservative_2021.py; checks member order first)
# 2) per training window: inference with the final MESACLIP checkpoint on the original 2021 inputs and on the
#    conservative-U10 inputs, through the identical --test-x-path code path (seed 0, K_EVAL=20, coarse-zero rule)
# 3) compare_u10_runs.py: RMSE / CRPS / spread-skill of each, and how much the ensemble-mean SIT moves
#   qsub evaluation/revisions/submit_u10_check.sh      (from the project root's evaluation/ folder or anywhere: absolute -o)
#PBS -N U10_check
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=1:mem=200GB:gpu_type=v100
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
set -uo pipefail
V6=/glade/work/skygale/projects/SeaIceDownscaling
D=/glade/derecho/scratch/skygale/downscaling
OUT=$D/outputs/revisions_2026-11/U10_check
XO=$D/training_data/X_MESA_HR_daily_avg.nc
XC=$D/training_data/X_MESA_HR_daily_avg_2021_U10conservative.nc
Y=$D/training_data/Y_MESA_HR_daily.nc
module load conda
conda activate seaice-downscaling
export PYTHONPATH="$V6/evaluation:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date)"
[ -f "$XC" ] || python -u $V6/evaluation/revisions/build_u10_conservative_2021.py || { echo "!!!!! build failed"; exit 1; }
cd $V6/training
for W in 2000-2005 2005-2010 2010-2015 2015-2020; do
    CKPT=$(ls $V6/results/MESA_final_ema/*_${W}_2021_*/model_state_dict.pt)
    for V in orig cons; do
        [ -f "$OUT/${W}_${V}/eval_data/fields.npz" ] && { echo "skip $W $V (done)"; continue; }
        TX=$XO; [ $V = cons ] && TX=$XC
        echo "===== $W $V ($(date))"
        python -u train_engressnet.py --x-path "$XO" --y-path "$Y" --test-x-path "$TX" --test-y-path "$Y" \
            --train-years "$W" --test-years 2021 --no-patches --lat-min 60 --lat-max 75 --lon-min -182 --lon-max -151 \
            --k 20 --k-eval 20 --beta 0.8 --seed 0 --num-epochs 0 --noise-amp-map --coarse-zero --no-figures \
            --init-checkpoint "$CKPT" --output-dir "$OUT/${W}_${V}" || echo "!!!!! $W $V failed"
    done
done
python -u $V6/evaluation/revisions/compare_u10_runs.py "$OUT" || echo "!!!!! compare failed"
echo "done $(date)"
