#!/bin/bash
# Revisions 2026-11, H.1: does the concatenated per-stage noise actually drive the ensemble spread?
# Evaluation only (no training): loads one final EngressNet + EMA checkpoint and re-evaluates the 2021 test year 5 times,
# with no stage ablated and with each decoder stage's noise switched off in turn (--noise-ablate, inference only),
# K_EVAL=20, coarse-zero rule on, same flags as the paper model. Outputs on scratch; then CRPS / corrected spread-skill
# per configuration via evaluation/revisions/summarize_noise_ablation.py.
#
# Submit (one job per dataset and window), e.g.:
#   qsub -v DATASET=MESA,WINDOW=2015-2020 evaluation/revisions/submit_noise_ablation.sh
#   qsub -v DATASET=FOSI,WINDOW=2015-2020 evaluation/revisions/submit_noise_ablation.sh
#PBS -N H1_ablate
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=1:mem=200GB:gpu_type=v100
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
set -uo pipefail
DATASET="${DATASET:-MESA}"
WINDOW="${WINDOW:-2015-2020}"
ABLATIONS="${ABLATIONS:-none d3 d2 d1 final}"
V6=/glade/work/skygale/projects/SeaIceDownscaling
OUT=/glade/derecho/scratch/skygale/downscaling/outputs/revisions_2026-11/H1_ablation/${DATASET}_${WINDOW}
D=/glade/derecho/scratch/skygale/downscaling
case "$DATASET" in
    MESA) BATCH=MESA_final_ema; X=$D/training_data/X_MESA_HR_daily_avg.nc; Y=$D/training_data/Y_MESA_HR_daily.nc ;;
    FOSI) BATCH=FOSI_final_ema; X=$D/training_data/X_FOSI_HR_JRA55_daily_avg.nc; Y=$D/training_data/Y_FOSI_HR_JRA55_daily.nc ;;
    *) echo "DATASET must be MESA or FOSI"; exit 1 ;;
esac
CKPT=$(ls $V6/results/$BATCH/*_${WINDOW}_2021_*/model_state_dict.pt)
[ "$(echo "$CKPT" | wc -l)" = 1 ] || { echo "expected one checkpoint, got: $CKPT"; exit 1; }
[ -f "$Y" ] || { echo "missing $Y"; exit 1; }
echo "Job started on $(hostname) at $(date) | $DATASET $WINDOW | checkpoint $CKPT"
module load conda
conda activate seaice-downscaling
export PYTHONPATH="$V6/evaluation:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $V6/training
mkdir -p "$OUT"
for A in $ABLATIONS; do
    ABL=""; [ "$A" != none ] && ABL="$A"
    echo "===== ablate: $A ($(date))"
    python -u train_engressnet.py --x-path "$X" --y-path "$Y" --train-years "$WINDOW" --test-years 2021 \
        --no-patches --lat-min 60 --lat-max 75 --lon-min -182 --lon-max -151 \
        --k 20 --k-eval 20 --beta 0.8 --seed 0 --num-epochs 0 --noise-amp-map --coarse-zero --no-figures \
        --init-checkpoint "$CKPT" --noise-ablate "$ABL" --output-dir "$OUT/ablate_${A}" \
        || echo "!!!!! ablation $A failed"
done
python -u $V6/evaluation/revisions/summarize_noise_ablation.py "$OUT" || echo "!!!!! summary failed"
echo "Job finished at $(date)"
