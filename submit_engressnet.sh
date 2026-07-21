#!/bin/bash
# ==============================================================
# PBS batch submission script for train_engressnet.py (Version4) on Casper
#
# Submit with:  qsub submit_engressnet.sh
# Check status: qstat -u $USER
#
# Try a different train/test split without editing this file:
# qsub -v TRAIN_YEARS="1958-1990",TEST_YEARS="1991-2005" submit_engressnet.sh
# ==============================================================

#PBS -N FOSI_nopatches
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=64GB:gpu_type=v100
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -m abe
#PBS -M skycgale@uw.edu

# No -o here on purpose: a fixed log filename would be shared (and clobbered) across
# concurrently-submitted jobs, e.g. a batch of runs with different TRAIN_YEARS/TEST_YEARS.
# Omitting -o falls back to PBS's own default, which embeds the job ID
# (FOSI_nopatches.o<jobid>), so every submission gets its own log file automatically.

set -euo pipefail

# FOSI_BGC HR record covers 1958-2022 (single realization)
# Accepts "YYYY-YYYY" ranges and/or comma-separated years, e.g. "1980-2000" or "1980-1990,1995,2000-2005"
# Leave BOTH blank ("") to fall back to a random 80/20 train/test split
TRAIN_YEARS="${TRAIN_YEARS:-1960-2000}"
TEST_YEARS="${TEST_YEARS:-2014-2022}"

# "true"  -> sliding-window patch extraction
# "false" -> train directly on one lat/lon sub-domain (required for the
#            domain-mean SIT time series and the candidate-point
#            [Kivalina/Shishmaref/Kotzebue/Nome] time series -- both are
#            skipped under patches=True, see save_evaluation_data() in
#            functions_engressnet.py)
USE_PATCHES="${USE_PATCHES:-false}"

# Sub-domain covering all 5 candidate coastal communities (Kivalina,
# Shishmaref, Kotzebue, Nome, Point Hope: lat 64.5-68.3, lon -166.8 to
# -162.6), widened from the original 8x16 crop to give the encoder more
# spatial context beyond just the coastal points themselves. Chosen so the
# 1-degree low-res crop is exactly 16x32 -- the UNet's 3 stride-2 pooling
# layers require both crop dims to be a multiple of 8 (and the resulting
# bottleneck to have more than 1 total spatial element, i.e. NOT both dims
# exactly 8 -- see extract_full_domain() in functions_engressnet.py). Must
# fall fully within the ML domain (lat 60-80, lon -190 to -140).
LAT_MIN="${LAT_MIN:-60}"
LAT_MAX="${LAT_MAX:-75}"
LON_MIN="${LON_MIN:--182}"
LON_MAX="${LON_MAX:--151}"

NUM_EPOCHS="${NUM_EPOCHS:-20}"

# Ensemble size during training (energy_loss's pairwise diversity term). K=9 matches
# hpo_echo/trial_results.csv trial 11 (best rmse, 0.1321, tied with trial 13) -- lr and
# batch_size were updated the same way, as train_engressnet.py's own --lr/--batch-size
# defaults, since they aren't overridden here. Caveat: that HPO run (2026-07-15) predates
# the coastal-weighted loss and the ocean_frac 5th X channel added since, and used
# train_years=1980-2005/test_years=2006-2014 (8 trial epochs) rather than this script's
# defaults -- worth a fresh HPO pass under the current setup if results don't hold up.
K="${K:-9}"

# Ensemble size at evaluation time (separate from K above, which is only used during
# training). train_engressnet.py's own --k-eval default is 6.
K_EVAL="${K_EVAL:-6}"

# Power parameter of the energy-score loss (functions_engressnet.energy_loss).
# train_engressnet.py's own --beta default is 1.0 (unchanged behavior); lowering it
# shifts relative weight toward the ensemble-spread term vs. the mean-accuracy term,
# which is the lever being tried to close the "Stochastic Mean underperforms
# Deterministic" gap seen in the coastal-fixed runs.
BETA="${BETA:-1.0}"

# Coastal-focused training: ocean cells within COASTAL_WIDTH high-res grid cells of land
# get COASTAL_BOOST x the loss weight of other ocean cells; land itself stays at the
# baseline weight of 1 (an earlier version zeroed it out and regressed badly -- see
# build_coastal_weight_map()'s docstring in functions_engressnet.py). The same
# COASTAL_WIDTH also defines the "Coastal MAE"/"Coastal RMSE" columns in metrics.csv, so
# the reported number tracks the region the loss actually prioritizes.
COASTAL_WIDTH="${COASTAL_WIDTH:-5}"
COASTAL_BOOST="${COASTAL_BOOST:-2.0}"

# A high-res cell is classified as land only if its regridded ocean fraction is below
# this value. Tightened from an original 0.5 majority-rule threshold to 0.1 (>90% land)
# -- the 0.5 rule called mixed coastal cells "land" even though they still carry real
# ice signal from their ocean fraction, so hard-zeroing model predictions there (see
# run_pipeline's ocean_test masking) disagreed with that residual truth signal and
# inflated IIEE, even though domain-wide accuracy and the visual land artifact were fine.
LAND_THRESHOLD="${LAND_THRESHOLD:-0.1}"

# ==============================================================

echo "Job started on $(hostname) at $(date)"
echo "PBS_JOBID: ${PBS_JOBID:-not set}"
echo "Train years: ${TRAIN_YEARS:-<random split>}   Test years: ${TEST_YEARS:-<random split>}"
echo "USE_PATCHES: ${USE_PATCHES}"
echo "K (train ensemble size): ${K}   K_EVAL (eval ensemble size): ${K_EVAL}"
echo "Beta: ${BETA}"
echo "Coastal width / boost: ${COASTAL_WIDTH} / ${COASTAL_BOOST}"
echo "Land threshold: ${LAND_THRESHOLD}"
echo "Sub-domain: lat ${LAT_MIN}-${LAT_MAX}, lon ${LON_MIN}-${LON_MAX}"

module load conda
conda activate downscaling_env

cd "$PBS_O_WORKDIR"

ARGS=(--num-epochs "$NUM_EPOCHS" --k "$K" --k-eval "$K_EVAL" --beta "$BETA" --coastal-width "$COASTAL_WIDTH" --coastal-boost "$COASTAL_BOOST" --land-threshold "$LAND_THRESHOLD")
[ -n "$TRAIN_YEARS" ] && ARGS+=(--train-years "$TRAIN_YEARS")
[ -n "$TEST_YEARS" ] && ARGS+=(--test-years "$TEST_YEARS")

if [ "$USE_PATCHES" = true ]; then
    ARGS+=(--patches)
else
    ARGS+=(--no-patches --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lon-min "$LON_MIN" --lon-max "$LON_MAX")
fi

python train_engressnet.py "${ARGS[@]}"

echo "Job finished at $(date)"
