#!/bin/bash
# Final round (2026-11-16): trim the final model's PIOMAS 2018-2022 output to the ICESat-2 period (trim_piomas_to_is2_window.py),
# then re-sample the ICESat-2 pass-cells/bins from the trimmed output under a check name and confirm they equal the cell files
# Table 5 / Fig. 9 / is2_spread_error.py use (final2_piomas, final2_detmse_piomas). Submit from evaluation/revisions/.
#PBS -N trim_piomas
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=64GB
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
V6=/glade/work/skygale/projects/SeaIceDownscaling
IS2E=/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells/eval
cd $V6 && python -u evaluation/revisions/trim_piomas_to_is2_window.py || exit 1
cd $V6/observations/processing
python -u extract_alongtrack_ensemble.py cells PIOMAS_2018-2022_FINAL2_FOSI trimcheck_final2_piomas || exit 1
python -u extract_alongtrack_ensemble.py cells PIOMAS_2018-2022_FINAL2_FOSI_detmse trimcheck_final2_detmse_piomas || exit 1
python - <<EOF
import numpy as np
ok = True
for old, new in (("final2_piomas", "trimcheck_final2_piomas"), ("final2_detmse_piomas", "trimcheck_final2_detmse_piomas")):
    for kind in ("cells", "bins"):
        a, b = np.load(f"$IS2E/{old}_{kind}.npz"), np.load(f"$IS2E/{new}_{kind}.npz")
        for k in ("members", "ensmean", "bilinear", "truth"):
            same = np.array_equal(a[k], b[k], equal_nan=True)
            ok &= same
            print(old, kind, k, "identical" if same else "!!!!! DIFFERENT")
print("TRIM CHECK", "PASSED: Table 5 inputs identical" if ok else "FAILED")
EOF
rm -f $IS2E/trimcheck_*.npz
echo "TRIM DONE $(date)"
