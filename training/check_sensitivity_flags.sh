#!/bin/bash
# Final round (2026-11-15): checks before submitting training/submit_sensitivity_final2.sh.
#  1. bit-identity: the final-model flags give identical weights and eval fields with the pre-change library
#     (copy in $OLD) and the current one (CPU, same seed, small FOSI run);
#  2. smoke tests: each new option runs end to end (1 epoch, few days).
# Submit from training/:  qsub -v OLD=<dir with the pre-change train_engressnet.py + functions_engressnet.py> check_sensitivity_flags.sh
#PBS -N sens2_check
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=160GB
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
export PYTHONPATH="/glade/work/skygale/projects/SeaIceDownscaling/evaluation:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
NEW=/glade/work/skygale/projects/SeaIceDownscaling/training
OUT=${OUT:-/glade/derecho/scratch/skygale/downscaling/outputs/sens2_check}
TD=/glade/derecho/scratch/skygale/downscaling/training_data
rm -rf "$OUT"; mkdir -p "$OUT"
BASE=(--x-path $TD/X_FOSI_HR_JRA55_daily_avg.nc --y-path $TD/Y_FOSI_HR_JRA55_daily.nc
      --lat-min 60 --lat-max 75 --lon-min -182 --lon-max -151 --train-years 2000-2020 --test-years 2021
      --train-random-days 24 --random-days-seed 0 --k 4 --k-eval 3 --num-epochs 1 --batch-size 4 --beta 0.8
      --noise-amp-map --energy-patch-size 10 --noise-lr-mult 10 --ema-decay 0.997 --coarse-zero --seed 0 --no-figures)
# 24 training days, 1 epoch, K=4 keep it small; the test year is the full 2021 as in the final model.

echo "== 1. bit-identity (final-model flags)"
(cd "$OLD" && python -u train_engressnet.py "${BASE[@]}" --output-dir "$OUT/old") > "$OUT/old.log" 2>&1 || echo "!!!!! old run failed"
(cd "$NEW" && python -u train_engressnet.py "${BASE[@]}" --output-dir "$OUT/new") > "$OUT/new.log" 2>&1 || echo "!!!!! new run failed"
python - "$OUT" <<'EOF'
import sys, numpy as np, torch, os
o = sys.argv[1]
ok = True
for f in ("model_state_dict.pt", "model_state_dict_last_step.pt"):
    a, b = torch.load(f"{o}/old/{f}", map_location="cpu"), torch.load(f"{o}/new/{f}", map_location="cpu")
    same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    print(f, "identical" if same else "DIFFERENT"); ok &= same
a, b = np.load(f"{o}/old/eval_data/fields.npz"), np.load(f"{o}/new/eval_data/fields.npz")
for k in a.files:
    same = np.array_equal(a[k], b[k], equal_nan=True)
    if not same:
        print("fields", k, "DIFFERENT"); ok = False
print("fields.npz", len(a.files), "arrays compared")
print("BIT-IDENTICAL" if ok else "!!!!! NOT BIT-IDENTICAL")
EOF

echo "== 2. smoke tests"
smoke() {  # smoke <name> <extra args...>
    local n=$1; shift
    (cd "$NEW" && python -u train_engressnet.py "${BASE[@]}" "$@" --output-dir "$OUT/smoke_$n") > "$OUT/smoke_$n.log" 2>&1 \
        && echo "smoke $n: ok ($(grep -c . "$OUT/smoke_$n.log") log lines)" || { echo "!!!!! smoke $n FAILED"; tail -20 "$OUT/smoke_$n.log"; }
}
smoke land05 --land-threshold 0.5
smoke globalz --global-noise
smoke depth5 --encoder-depth 5
smoke attn --attention-end
smoke uncorr --noise-kernel-size 1
smoke bigdomain --lon-min -190 --lon-max -143 --grad-accum-steps 2
smoke season --months 3-7
python - "$OUT" <<'EOF'
import sys, json, numpy as np, pandas as pd
o = sys.argv[1]
s = pd.read_csv(f"{o}/smoke_season/train_day_mask.csv") if False else None
for n in ("land05", "globalz", "depth5", "attn", "uncorr", "bigdomain", "season"):
    try:
        z = np.load(f"{o}/smoke_{n}/eval_data/fields.npz")
        print(n, "Y_pred", z["Y_pred_phys"].shape, "members", z["preds_all_phys"].shape, "ocean frac", float((z["mask_test"][:, 0] <= 0.5).mean()).__round__(3))
    except Exception as e:
        print(n, "no fields:", e)
EOF
grep -h "random training days" "$OUT/smoke_season.log" "$OUT/new.log"
echo "CHECKS DONE $(date)"
