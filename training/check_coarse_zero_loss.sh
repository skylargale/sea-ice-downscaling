#!/bin/bash
# Final round part 3 (2026-11-17): checks for --coarse-zero-loss (made from check_coarse_zero_loss.sh).
#  1. bit-identity: the final-model flags give identical weights and eval fields with the pre-change library
#     (copy in $OLD) and the current one (CPU, same seed, small FOSI run);
#  2. smoke tests: --coarse-zero-loss runs end to end on both training-loop branches (plain and grad-accum).
# Submit from training/:  qsub -v OLD=<dir with the pre-change train_engressnet.py + functions_engressnet.py> check_coarse_zero_loss.sh
#PBS -N czloss_check
#PBS -A P93300065
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=160GB
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /glade/work/skygale/projects/SeaIceDownscaling/logs/
module load conda
conda activate seaice-downscaling
export PYTHONPATH="/glade/work/skygale/projects/SeaIceDownscaling/evaluation:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
NEW=/glade/work/skygale/projects/SeaIceDownscaling/training
OUT=${OUT:-/glade/derecho/scratch/skygale/downscaling/outputs/czloss_check}
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
smoke czloss --coarse-zero-loss
smoke czloss_accum --coarse-zero-loss --grad-accum-steps 2
grep -h "coarse-zero rule in the loss" "$OUT/smoke_czloss.log" "$OUT/smoke_czloss_accum.log"
python - "$OUT" <<'EOF'
import sys, torch
o = sys.argv[1]
a = torch.load(f"{o}/new/model_state_dict.pt", map_location="cpu")
for n in ("czloss", "czloss_accum"):
    b = torch.load(f"{o}/smoke_{n}/model_state_dict.pt", map_location="cpu")
    print(n, "weights differ from the final-model run (expected):", any(not torch.equal(a[k], b[k]) for k in a))
EOF
echo "CHECKS DONE $(date)"
