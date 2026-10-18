#!/bin/bash
# 2026-10-18 EMA model, part 2: ICESat-2 scoring notebooks, sub-grid variability, Fig. 7, paper figure notebook (RUN=MESA_final_ema),
# and the final results table. Needs the EMA observation batches/cells (observations/inference/submit_ema_obs_inference.py), the
# PIOMAS 2010-2023 EMA application batches, and ema_eval_and_figures.sh.
#PBS -N ema_obs_paper
#PBS -l select=1:ncpus=4:mem=200GB
#PBS -l walltime=10:00:00
#PBS -A P93300065
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/observation_logs/
set -uo pipefail
module load conda
conda activate seaice-downscaling
V6=/glade/work/skygale/projects/SeaIceDownscaling
O=$V6/observations
run_nb() {  # src.py -> notebook, execute in place
    python $O/notebook_src/py2nb.py $O/notebook_src/$1_src.py $O/$1.ipynb && \
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=seaice-downscaling \
        --ExecutePreprocessor.timeout=14400 $O/$1.ipynb || echo "!!!!! $1 failed"
}
cd $O
run_nb engressnet_obs_evaluation
run_nb engressnet_inputs_and_ceiling
cd $V6/figures
python -u build_cice_ceiling_bar.py || echo "!!!!! cice ceiling bar failed"
python -u build_fig7_final_piomas_application.py || echo "!!!!! fig7 application failed"
python -u build_fig7_final_piomas_vs_icesat2.py || echo "!!!!! fig7 vs icesat2 failed"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=seaice-downscaling \
    --ExecutePreprocessor.timeout=14400 all_figures_by_run.ipynb || echo "!!!!! paper figure notebook failed"
python -u final_chain/final_results_summary.py || echo "!!!!! summary failed"
echo "EMA OBS + PAPER FIGURES DONE $(date)"
