#!/bin/bash
# ==============================================================
# PBS batch submission script for evaluation/build_analog_baseline.py --
# the distributional (nearest-neighbor analog) baseline requested in the
# coauthor review. CPU-only: no model, no training loop, just a KD-tree
# lookup, so no GPU is requested (unlike the actual training submit
# scripts). Mem sized to match the MESA training jobs' own budget, since
# this script materializes the same full X/Y arrays via functions_engressnet's
# own load path before cropping to the sub-domain (Y_MESA_HR_daily.nc's full
# array alone is ~24GB in memory).
#
# Submit with:
#   qsub -v TEMPLATE_RUN="results/MESA_stochastic_refine_sweep_avg/MESA_refine_avg_2000-2005_2021_5622261.casper-pbs" submit_analog_baseline.sh
# Optional K override (default: template run's own k_eval, 20) and OUTPUT_DIR override (default:
# build_analog_baseline.py's own results/MESA_analog_baseline/MESA_analog_<window> naming) -- for
# a K-sensitivity sweep, set both so different K values don't collide:
#   qsub -v TEMPLATE_RUN=...,K=5,OUTPUT_DIR="results/MESA_analog_baseline_k5/MESA_analog_k5_<window>" submit_analog_baseline.sh
# ==============================================================

#PBS -N analog_baseline
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=256GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o .logs/
#PBS -m abe
#PBS -M skycgale@uw.edu

set -euo pipefail

TEMPLATE_RUN="${TEMPLATE_RUN:?Must set TEMPLATE_RUN, e.g. -v TEMPLATE_RUN=results/MESA_stochastic_refine_sweep_avg/<window>}"
K="${K:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

module load conda
conda activate downscaling_env

# Hardcoded absolute path, not $PBS_O_WORKDIR-relative or BASH_SOURCE-based -- see
# submit_daily_eval_batch.sh's matching comment for why (PBS spools a submitted script at
# qsub time, so BASH_SOURCE-based self-location breaks under real execution).
cd "/glade/work/skygale/projects/SeaIceDownscaling/Version6"

echo "Template run: $TEMPLATE_RUN"
echo "K override: ${K:-none, using the template run k_eval}"
echo "Output dir override: ${OUTPUT_DIR:-none, using the default}"
echo "Job started at $(date)"

ARGS=(--template-run "$TEMPLATE_RUN")
[ -n "$K" ] && ARGS+=(--k "$K")
[ -n "$OUTPUT_DIR" ] && ARGS+=(--output-dir "$OUTPUT_DIR")
python evaluation/build_analog_baseline.py "${ARGS[@]}"

echo "Job finished at $(date)"
