#!/bin/bash
# ==============================================================
# submit_stochastic_refine_bilinear_2015_2020_avg_coastalboost1.sh
#
# Test 2 (coastal_boost ablation): retrains the exact recommended config
# (STOCHASTIC_REFINE=true, NOISE_SIGMA=1.0, DATA_VARIANT=avg, train
# 2015-2019, test 2020) that produced
# results/FOSI_stochastic_refine_bilinear_2000_2020_avg/FOSI_refine_bilin_avg_2015-2019_2020_5631173.casper-pbs/model_state_dict.pt
# -- the checkpoint used for every Paragraph 7 PIOMAS-observational-
# application run -- with ONE change: COASTAL_BOOST=1.0 instead of the
# default 2.0.
#
# Why: results/PIOMAS_obs_2020/cryosat2_validation/network_vs_bilinear_coastal_bias.csv
# showed the network amplifies PIOMAS's documented coastal thin-bias
# relative to bilinear in 6/7 months (RMSE ~19% worse in the coastal band).
# coastal_boost up-weights the training loss for ocean cells near the coast
# -- the leading hypothesis for why the network learned to be "decisive"
# at the coast in a way that doesn't transfer usefully when the real-world
# PIOMAS input is coastally biased. This tests whether removing that
# up-weighting (coastal_boost=1.0, i.e. no special coastal treatment)
# reduces the amplification effect, at the cost of the coastal accuracy
# gains coastal_boost=2.0 provides in the perfect-model (FOSI-only) setting.
#
# After this finishes, rerun the same --calibrate-from + --num-epochs 0
# PIOMAS-observational inference against this new checkpoint (see
# submit_infer_piomas_obs_2020_coastalboost1.sh), then the same CryoSat-2
# comparison scripts used for the original checkpoint.
#
# Submit with:  qsub submit_stochastic_refine_bilinear_2015_2020_avg_coastalboost1.sh
# ==============================================================

#PBS -N FOSI_refine_bilin_avg_cb1
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=128GB:gpu_type=v100
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR/../.."
mkdir -p logs

DATA_DIR="/glade/derecho/scratch/skygale/Downscaling_Data"

PYTHONPATH="$PWD/evaluation:${PYTHONPATH:-}" python training/train_engressnet.py \
    --x-path "${DATA_DIR}/X_FOSI_HR_JRA55_daily_avg.nc" \
    --y-path "${DATA_DIR}/Y_FOSI_HR_JRA55_daily.nc" \
    --train-years "2015-2019" \
    --test-years "2020" \
    --no-patches --lat-min 60 --lat-max 75 --lon-min -182 --lon-max -151 \
    --stochastic-refine --noise-sigma 1.0 \
    --num-epochs 20 --k 20 --k-eval 20 --beta 0.8 \
    --coastal-width 5 --coastal-boost 1.0 --land-threshold 0.1 \
    --seed 0 \
    --output-dir "results/FOSI_stochastic_refine_bilinear_2000_2020_avg_coastalboost1/FOSI_refine_bilin_avg_cb1_2015-2019_2020_${PBS_JOBID:-local}"

echo "Job finished at $(date)"
