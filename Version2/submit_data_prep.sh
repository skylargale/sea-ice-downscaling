#!/bin/bash
#PBS -N engressnet_data_prep
#PBS -q casper
#PBS -l select=1:ncpus=64:mem=256GB
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/_projects/SeaIceDownscaling/logs/data_prep.log

# ── Edit these if your paths differ ───────────────────────────────────────────
NOTEBOOK=/glade/work/skygale/_projects/SeaIceDownscaling/engressnet_data_prep.ipynb
CONDA_ENV=downscaling_env
LOG_DIR=/glade/work/skygale/_projects/SeaIceDownscaling/logs
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p "${LOG_DIR}"

echo "==== Job started: $(date) ===="
echo "Host:    $(hostname)"
echo "Job ID:  ${PBS_JOBID}"
echo "Cores:   $(nproc)"

# Activate conda environment
module load conda
conda activate "${CONDA_ENV}"
echo "Python:  $(which python)"
echo "Kernel env: ${CONDA_ENV}"

# Convert the notebook to a plain Python script.
# nbconvert strips cell outputs and magic commands; the resulting .py
# is placed alongside the notebook and deleted on clean exit.
SCRIPT="${NOTEBOOK%.ipynb}.py"

echo ""
echo "==== Converting notebook to script ===="
jupyter nbconvert \
    --to script \
    --output "${SCRIPT%.py}" \
    "${NOTEBOOK}"

# nbconvert keeps IPython magics like %matplotlib inline as-is, which
# crash a plain Python run. Strip them out.
sed -i '/^[[:space:]]*%/d;/^[[:space:]]*get_ipython/d' "${SCRIPT}"

echo "Script written to: ${SCRIPT}"

# Run the script, routing stdout/stderr to the PBS log (-j oe above)
echo ""
echo "==== Running script ===="
python "${SCRIPT}"

echo ""
echo "==== Job finished: $(date) ===="

# Clean up the generated .py (the notebook is the source of truth)
rm -f "${SCRIPT}"