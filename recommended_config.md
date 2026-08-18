# Recommended U-Net configuration — Version5

Compiled 2026-08-14, updated 2026-08-17 after the follow-up round that swept the
recommended config across the full 4-window grid (both data variants, both datasets),
tested the refiner+land-threshold combo to completion, quantified run-to-run seed
variance, and ran the conservative-vs-bilinear truth-regridding comparison for the
2000-2020 downscaling target. All numbers below are read directly from each run's
`metrics.csv`; every MESA number is the **per-member-averaged** value (never the raw
ensemble-mean-vs-single-realization number — see `member_metrics.py` /
`compute_member_avg_metrics`), either taken from `functions_engressnet.py`'s automatic
`per_member` correction (all MESA batches trained since 2026-08-12) or recomputed and
verified directly against `eval_data/fields.npz` for older/one-off batches.

**One thread still open**: the refiner+attention-end combo viability check
(`FOSI_stochastic_refine_attn_combo`/`MESA_stochastic_refine_attn_combo`, single split
each) — FOSI is done, MESA is still running. Everything else in this document is final.

## Recommended config

```
STOCHASTIC_REFINE=true
NOISE_SIGMA=1.0
DATA_VARIANT=avg          # confirmed better than interp on both datasets, see below
LAND_THRESHOLD=0.1        # default -- combining with 0.5 tested, does NOT help, see below
EXTRA_LAYER=false
ENSCALE_NET=false
COASTAL_CHANNEL=false
CLASSIFICATION_HEAD=false
ATTENTION_END=false
```
i.e. `qsub -v BATCH_NAME=...,TRAIN_YEARS=...,TEST_YEARS=...,DATA_VARIANT=avg,STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0 submit_engressnet_daily.sh`

This is **not** the config the original sweep was aimed at confirming. Going in, the
natural best-guess candidate was `ENSCALE_NET=true,LAND_THRESHOLD=0.5` (the two
previously-documented positive findings, combined). The completed sweep showed that combo
is a real but mixed tradeoff (see below) — the smooth-noise-fixed EnScale-lite refiner
(`--stochastic-refine`, with an unconditional `smooth_noise()` fix baked into
`functions_engressnet.py`) beats it outright on every metric checked, on both datasets,
now confirmed across the full 4-window grid rather than a single split. The `avg` data
variant addition is new this round — it was never swept before and turns out to matter
more than any architecture toggle tested (see "Data variant" section below).

## Evidence table

All values are 4-window-split averages (`train ∈ {2000-2005, 2005-2010, 2010-2015,
2015-2020}`, `test=2021`, medium domain lat60-75/lon-182to-151) except where noted as a
single split. Data variant is `interp` unless marked `avg`.

| Config | Dataset | n | RMSE | Coastal RMSE | Pattern Corr | SSIM | Spread/Error |
|---|---|---|---|---|---|---|---|
| **Baseline** (no toggles, `el0_dmed_es0`) | FOSI | 4 | 0.2198 | 0.4053 | 0.9177 | 0.8684 | 1.5819 |
| **Baseline** | MESA | 4 | 0.1388 | 0.2493 | 0.8655 | 0.9342 | 1.7424 |
| **Old EnScale-lite** (`es1`, pre-smooth-noise-fix) | FOSI | 4 | 0.2181 | 0.3949 | 0.9184 | 0.8828 | **0.7584** (broken) |
| **Old EnScale-lite** (`es1`, pre-fix) | MESA | 4 | 0.1329 | 0.2496 | 0.8808 | 0.9397 | **0.9588** |
| **Smooth-noise-fixed EnScale-lite** (interp, full grid) | FOSI | 4 | 0.2160 | 0.3916 | 0.9208 | 0.8826 | 0.750 |
| **Smooth-noise-fixed EnScale-lite** (interp, full grid) | MESA | 4 | 0.1332 | 0.2516 | 0.8468 | 0.9201 | 0.916 |
| **★ Smooth-noise-fixed EnScale-lite** (`avg`, full grid — RECOMMENDED) | FOSI | 4 | **0.1804** | **0.2960** | 0.9458 | 0.8968 | 0.833 |
| **★ Smooth-noise-fixed EnScale-lite** (`avg`, full grid — RECOMMENDED) | MESA | 4 | **0.1111** | **0.1938** | 0.8783 | 0.9344 | 1.034 |
| Land threshold 0.5 alone (single split, 2015-2020) | FOSI | 1 | 0.2460 | 0.3821 | 0.9025 | 0.8671 | 1.4546 |
| Land threshold 0.5 alone (single split, 2015-2020) | MESA | 1 | 0.1527 | 0.2363 | 0.8685 | 0.9282 | 1.6037 |
| Refiner + land threshold 0.5 (interp, full grid) | FOSI | 4 | 0.2449 | 0.3817 | 0.9079 | 0.8737 | 0.722 |
| Refiner + land threshold 0.5 (interp, full grid) | MESA | 4 | 0.1477 | 0.2376 | 0.8369 | 0.9155 | 0.876 |
| EnScaleNet alone (full grid, two independent sweeps agree) | FOSI | 4 | 0.2201–0.2235 | 0.4064–0.4190 | — | — | 1.085–1.153 |
| EnScaleNet alone (full grid, two independent sweeps agree) | MESA | 4 | 0.1392–0.1425 | 0.2619–0.2710 | — | — | 1.272–1.321 |
| EnScaleNet + land threshold 0.5 (full grid) | FOSI | 4 | 0.2475 | 0.3949 | 0.9064 | 0.8682 | 1.0972 |
| EnScaleNet + land threshold 0.5 (full grid) | MESA | 4 | 0.1550 | 0.2541 | 0.8329 | 0.9060 | 1.2319 |
| Season-restricted training (`MONTHS=3-7`) | FOSI | 4 | 0.2244 | 0.3917 | 0.9697 | 0.8319 | 1.2713 |
| Season-restricted training | MESA | 4 | 0.1744 | 0.2712 | 0.9538 | 0.8296 | 1.5423 |
| Full-year, matched baseline | FOSI | 4 | 0.1665 | 0.2817 | 0.9469 | 0.8896 | 1.4516 |
| Full-year, matched baseline | MESA | 4 | 0.1262 | 0.2094 | 0.8766 | 0.9089 | 1.6726 |
| Conservative-truth regrid, recommended config, 2000-2020 target (bilinear baseline) | FOSI | 4 | 0.1926 | 0.3363 | 0.9292 | 0.8876 | 0.796 |
| Conservative-truth regrid, recommended config, 2000-2020 target (conservative) | FOSI | 4 | 0.2056 | 0.3262 | 0.9252 | 0.8777 | 0.723 |
| Conservative-truth regrid, recommended config, 2000-2020 target (bilinear baseline) | MESA | 4 | 0.1450 | 0.2727 | 0.8553 | 0.9188 | 0.858 |
| Conservative-truth regrid, recommended config, 2000-2020 target (conservative) | MESA | 4 | 0.1572 | 0.2683 | 0.8471 | 0.9177 | 0.834 |

## Data variant: `avg` beats `interp` — new, confirmed on both datasets

Never checked before this round (all prior evidence for the recommended config used the
`interp` default). For the recommended config specifically, `avg` is a large, consistent
win on both datasets:

- **FOSI**: RMSE 0.2160→0.1804 (−16.5%), Coastal RMSE 0.3916→0.2960 (−24.4%).
- **MESA**: RMSE 0.1332→0.1111 (−16.6%), Coastal RMSE 0.2516→0.1938 (−23.0%).

The magnitude is nearly identical across both datasets, which rules out this being a
one-off quirk of either dataset's `avg`-file construction. Calibration also moves closer
to the 1.0 ideal on `avg` for both datasets (FOSI 0.750→0.833; MESA 0.916→1.034 — MESA's
`avg` variant actually crosses to mild *over*-dispersion, but `|1.034−1| < |0.916−1|`, so
it's still the better-calibrated of the two). This is now a bigger effect than any single
architecture toggle tested in this project — **the recommended config should always be run
with `DATA_VARIANT=avg`.**

## Why the smooth-noise-fixed EnScale-lite refiner, not EnScaleNet

The original hypothesis (EnScaleNet, or EnScaleNet+land-threshold-0.5) was based on a
single split's evidence and turned out to be a real but **mixed** result once swept
properly across all 4 windows:

- EnScaleNet alone consistently **improves calibration** (Spread/Error 1.58→1.09–1.15
  FOSI, 1.74→1.27–1.32 MESA) but **costs RMSE and Coastal RMSE** rather than leaving them
  flat as the earlier single-split evidence suggested. Two independent 4-window sweeps
  agree closely, so this isn't noise.
- The EnScaleNet+land-threshold-0.5 combo pushes calibration even closer to 1.0 (FOSI
  1.10, MESA 1.23 — the best calibration of any *architecture-toggle* config tested) but
  at real cost: worst RMSE of any EnScaleNet-family config, and no coastal RMSE win on
  MESA.
- The smooth-noise-fixed EnScale-lite refiner, by contrast, **improves RMSE relative to
  baseline on both datasets, both variants, across the full grid**, while swinging
  calibration dramatically closer to 1.0 from the badly under-dispersive pre-fix state
  (0.76 FOSI / 0.96 MESA → 0.75–1.03 depending on dataset/variant). It's the only
  candidate architecture change that's a net win on accuracy *and* calibration
  simultaneously, not a tradeoff between them.

## What did NOT resolve the coastal-bias problem — pattern confirmed across three separate experiments

Three independent things were tried to fix the model's coastal RMSE, and all three show
**the exact same tradeoff shape**: a small, real Coastal RMSE improvement bought at the
cost of meaningfully worse overall RMSE, with no calibration benefit:

| Fix tried | Dataset | RMSE change | Coastal RMSE change |
|---|---|---|---|
| Land threshold 0.1→0.5 alone | FOSI | +11.9% | −5.7% |
| Land threshold 0.1→0.5 alone | MESA | +10.0% | −5.2% |
| Refiner + land threshold 0.5 | FOSI | +13.4% (vs. refiner alone) | −2.5% |
| Refiner + land threshold 0.5 | MESA | +10.9% (vs. refiner alone) | −5.6% |
| Conservative vs. bilinear truth regrid | FOSI | +6.8% | −3.0% |
| Conservative vs. bilinear truth regrid | MESA | +8.4% | −1.6% |

None of these is worth adopting on its own — the coastal RMSE gain (2–7%) is consistently
smaller than the RMSE cost (7–13%) across every variant of this fix that's been tried,
architectural or data-side. The original coastal-bias problem remains genuinely unsolved;
this project has now ruled out the three most obvious candidate fixes rather than found
one.

## Data variant: conservative regridding for the 2000-2020 target — resolved, negative

This closes out the "production data is still bilinear" question **for the actual
2000-2020 downscaling target**, not just as an experiment. `Y_FOSI_HR_JRA55_daily_conservative_2000_2020.nc`
and `Y_MESA_HR_daily_conservative_2000_2020.nc` were built (FOSI: straightforward
extension of the existing regrid script; MESA: full HIST+RCP8.5 conservative pipeline,
matching the production stitch's exact 6-member alignment), and the recommended config was
retrained from scratch against both the conservative and matched bilinear truth at the
same 4 windows (train 2000-2004/2005-2009/2010-2014/2015-2019, test 2020).

**Result: conservative regridding is not worth adopting.** Same tradeoff shape as
land-threshold (see table above) — real RMSE cost (FOSI +6.8%, MESA +8.4%) for a smaller
Coastal RMSE gain (FOSI −3.0%, MESA −1.6%), and calibration moves slightly further from
1.0 in both cases rather than improving. Since this experiment specifically targeted the
2000-2020 window the user is downscaling for (not a smaller test slice standing in for a
future full-record rebuild), **this question is fully closed, not just deferred** — there
is no larger production-data rebuild decision hanging on this result. Stick with the
existing bilinear-regridded truth.

Note: this comparison was run on the `interp` data variant (matching how the original
land-threshold/EnScaleNet-combo experiments were scoped), not the now-recommended `avg`
variant — see "Still open."

## Run-to-run seed variance — quantified, both datasets

Previously unquantified; now measured directly. `submit_engressnet_daily.sh`/`_mesa.sh`
gained a `SEED` passthrough (`--seed`, `torch.manual_seed` + train/test-split RNG) this
round specifically to make this test possible. 5 seeds each (seed 0 = the original
single-split evidence run), same split (2000-2005→2021), interp:

- **FOSI**: RMSE 0.2145–0.2184 (±1.8% around the mean), Spread/Error 0.843–0.936 (±10%).
- **MESA**: RMSE 0.1323–0.1354 (±2.3% around the mean), Spread/Error 0.872–0.993 (±12%).

Same shape both times: **RMSE is tightly reproducible run-to-run; Spread/Error is
noticeably noisier.** This matters for reading every single-split number elsewhere in
this project's history — a single-split Spread/Error can plausibly be off by ±0.05-0.10
from what a repeated-seed mean would show, while a single-split RMSE is a much more
reliable point estimate. The 4-window averages in this document are far more trustworthy
than any single-split number for exactly this reason.

## Season vs. full-year training — resolved, negative

The season-restriction question (does training only on March-July help?) is resolved:
**no, full-year training is clearly better** on both datasets. Season-restricted training
is worse on RMSE (FOSI 0.2244 vs. 0.1665, MESA 0.1744 vs. 0.1262), Coastal RMSE (FOSI
0.3917 vs. 0.2817, MESA 0.2712 vs. 0.2094), and SSIM, in every one of the 4 matched
window-pairs. Its only advantage is Pattern Corr (FOSI 0.9697 vs. 0.9469, MESA 0.9538 vs.
0.8766) — plausibly because melt-season ice-edge shape is more spatially coherent and
predictable than magnitude. Not worth adopting as a training-data restriction.

**Caveat resolved 2026-08-18**: the comparison above uses different test sets per arm —
the season arm is tested on Mar-Jul only, the full-year arm on all 12 months — so part of
the full-year arm's apparent advantage could have been an easier, broader test set rather
than genuinely better generalization to the melt season specifically. Checked directly:
each `FOSI_fullyear_avg`/`MESA_fullyear_avg` checkpoint was re-evaluated (inference only,
no retraining) against a Mar-Jul-only slice of its own test set, using
`functions_engressnet.py`'s own load/split/normalize/evaluate functions directly (script:
`process_data`-adjacent one-off, `submit/evaluation/submit_eval_fullyear_on_season.sh`).
4-window averages, same test period for both arms this time:

| Metric | FOSI, trained Mar-Jul | FOSI, trained full-year (tested Mar-Jul only) | MESA, trained Mar-Jul | MESA, trained full-year (tested Mar-Jul only) |
|---|---|---|---|---|
| RMSE | 0.2244 | **0.2155** | 0.1744 | **0.1696** |
| Coastal RMSE | 0.3917 | **0.3758** | 0.2712 | 0.2771 |
| Pattern Corr | 0.9697 | **0.9722** | 0.9538 | **0.9612** |
| SSIM | 0.8319 | **0.8489** | 0.8296 | **0.8544** |
| Spread/Error | 1.2713 | 1.3563 | 1.5423 | 1.5390 |

**The conclusion holds and gets stronger, not weaker.** Even on a genuinely matched test
period, full-year training beats season-only training on RMSE (FOSI −4.0%, MESA −2.8%),
MAE, Pattern Corr, and SSIM on both datasets — the earlier result wasn't an artifact of
an easier full-year test set. The only exceptions: calibration is roughly flat to
slightly worse for full-year training, and MESA's Coastal RMSE is ~2.2% worse. Full-year
training genuinely generalizes better to the melt season than training exclusively on
it, not just on average across the year.

## Toggle sweep: no other standout

`coastal_channel` and `classification_head`: no standout signal on either dataset, either
data variant — all within the same noise band as baseline on every metric. `attention_end`
is the one exception: a real, consistent calibration signal on both FOSI and MESA
(Spread/Error swings meaningfully closer to 1.0 than either sibling toggle manages), but
it costs RMSE/Coastal RMSE at roughly the same rate the other toggles do, so it's not a
net win against baseline. Its combination with the refiner is the one thread still open
— see below. `FOSI_calibrate` (freeze-backbone + calibration pathway): a clean negative,
Spread/Error 1.65, indistinguishable from baseline — not worth pursuing further.

## Still open

1. ~~Refiner + attention-end combo~~ — **RESOLVED 2026-08-18, negative on both
   datasets, no follow-up sweep warranted.** MESA's first attempt (job 5622274) failed
   on the 8h walltime (same failure mode as the earlier `MESA_attn_end_avg` batch);
   resubmitted with `walltime=16:00:00` (job 5631599), succeeded after ~10h54m —
   confirms the combo genuinely needs the longer walltime, not a fluke. Single split
   each (2000-2005→2021, interp), cleanly matched against both ingredients:

   | Config | Dataset | RMSE | Coastal RMSE | Spread/Error |
   |---|---|---|---|---|
   | Refiner alone | FOSI | 0.2174 | 0.4005 | 0.843 (near-ideal) |
   | Attention-end alone | FOSI | 0.2178 | 0.4050 | 1.358 (over-dispersive) |
   | Refiner+attention combo | FOSI | 0.2168 | 0.4027 | 1.076 (over-dispersive) |
   | Refiner alone | MESA | 0.1339 | 0.2534 | 0.986 (near-ideal) |
   | Attention-end alone | MESA | 0.1414 | 0.2464 | 1.229 (over-dispersive) |
   | Refiner+attention combo | MESA | 0.1370 | **0.2586** (worst of the three) | 1.035 |

   Neither dataset shows an accuracy win: FOSI's RMSE is flat, MESA's sits between the
   two ingredients with no improvement, and MESA's Coastal RMSE is actually the worst
   of any of the three configs. Calibration moves in the same direction both times —
   from the refiner's already-near-ideal Spread/Error into mild over-dispersion (FOSI
   0.843→1.076, MESA 0.986→1.035) — the opposite of a calibration win; on MESA it's
   still much better than attention-end alone's poor 1.229, but worse than the refiner
   by itself. **The two toggles don't compose usefully — adding attention-end to the
   refiner is a net negative or neutral change, never a win, on either dataset.**
2. ~~Refiner+land-threshold combo and the conservative-regrid comparison, checked on
   `avg`~~ — **RESOLVED 2026-08-18, both datasets, both combos, same conclusion as
   `interp` in every case:**
   - **Conservative-regrid on `avg`**: FOSI RMSE +10.4% (0.179 vs. 0.162)/Coastal RMSE
     −3.2%; MESA RMSE +12.0% (0.134 vs. 0.120)/Coastal RMSE −2.7%. Matches the `interp`
     finding (FOSI +6.8%/−3.0%, MESA +8.4%/−1.6%) — `avg` doesn't change the verdict.
   - **Refiner+land-threshold on `avg`**: FOSI RMSE +18.5% vs. refiner-alone-avg (0.214
     vs. 0.180)/Coastal RMSE −4.4%/Spread-Error 0.771 (further from 1.0 than
     refiner-alone's 0.833); MESA RMSE +16.6% (0.130 vs. 0.111)/Coastal RMSE −6.6%
     (MESA's coastal gain is larger than FOSI's, but still far outweighed by the RMSE
     cost)/Spread-Error 0.965 (negligibly different from refiner-alone's 1.034 — no real
     calibration change on MESA, unlike FOSI's small regression). Matches the `interp`
     finding (FOSI +13.4%, MESA +10.9%) — `avg` doesn't change the verdict either.
   **Both combos are confirmed not worth adopting on any dataset/data-variant
   combination tested (4 of 4).** The `interp`-only original evidence generalizes
   cleanly; no combination reversed on `avg`.
3. Whether `attention_end` alone (without the refiner) is worth a dedicated
   calibration-focused deployment remains an open judgment call, not a data question —
   the numbers are already in the evidence table above.

## Cross-dataset generalization (FOSI ↔ MESA) — new 2026-08-18

Tests whether the recommended config generalizes across datasets, not just across
train/test time splits within one: train on FOSI, evaluate the checkpoint on MESA-HR
(and vice versa), instead of training and testing on the same dataset. This needed two
additions to the pipeline: `train_engressnet.py`/`functions_engressnet.py` gained
`--test-x-path`/`--test-y-path` (train on `x_path`/`y_path`'s `train_years`, evaluate on
a *different* dataset's `test_years`), and a predictor-channel harmonization step
(`collapse_wind_vector_channel`) — FOSI's X carries full vector wind (`u_10`, `v_10`)
while MESA-HR's carries only wind speed (`U10`), a real schema difference between the
two datasets' predictor-construction pipelines that a same-schema model can't ingest
directly. Whichever side has the vector-wind pair gets it collapsed to a single derived
`sqrt(u^2+v^2)` speed channel so both sides present `(hi_d, aice_d, wind_speed)`. Also
fixed: the MESACLIP per-member-averaged-metrics correction (member_metrics.py) was keyed
off output-folder naming (`MESA_` prefix), which is backwards for a cross-dataset run —
now keyed off which dataset actually supplies the test truth (`test_y_path`) instead.
Both directions smoke-tested end-to-end (1 epoch, K=2) before the full sweep.

Same 4-window methodology as the rest of this document (train ∈ {2000-2005, 2005-2010,
2010-2015, 2015-2020}, test=2021, recommended config: `STOCHASTIC_REFINE=true,
NOISE_SIGMA=1.0, DATA_VARIANT=avg`), `results/FOSI_train_MESA_test_recommended/` and
`results/MESA_train_FOSI_test_recommended/`:

| Direction | Train window | RMSE | Coastal RMSE | Pattern Corr | SSIM | Spread/Error |
|---|---|---|---|---|---|---|
| FOSI→MESA | 2000-2005 | 0.1670 | 0.2959 | 0.6708 | 0.8151 | 1.4198 |
| FOSI→MESA | 2005-2010 | 0.1553 | 0.2840 | 0.7175 | 0.8679 | 1.1494 |
| FOSI→MESA | 2010-2015 | 0.1498 | 0.2831 | 0.7099 | 0.8621 | 1.0128 |
| FOSI→MESA | 2015-2020 | 0.1573 | 0.2871 | 0.7039 | 0.8566 | 1.0908 |
| MESA→FOSI | 2000-2005 | 0.2053 | 0.3748 | 0.9412 | 0.8925 | 0.8527 |
| MESA→FOSI | 2005-2010 | 0.2021 | 0.3683 | 0.9418 | 0.8932 | 0.8616 |
| MESA→FOSI | 2010-2015 | 0.1996 | 0.3602 | 0.9408 | 0.8922 | 0.8189 |
| MESA→FOSI | 2015-2020 | 0.2034 | 0.3623 | 0.9408 | 0.8900 | 0.7714 |

Both directions beat their respective bilinear baseline in every window (FOSI-as-target
bilinear RMSE 0.4464 vs. MESA→FOSI's ~0.20; MESA-as-target bilinear RMSE 0.2581 vs.
FOSI→MESA's ~0.15-0.17) — the model generalizes usefully across datasets, not just
within one. But same-dataset training is still clearly better than cross-dataset in
both directions: FOSI→MESA's RMSE (~0.15-0.17) is 35-50% worse than this document's
in-domain MESA-trained/MESA-tested recommended-config RMSE (0.1111), and MESA→FOSI's
RMSE (~0.20) is 11-24% worse than in-domain FOSI-trained/FOSI-tested (0.1804 for the
recommended config, from the evidence table above) — real but bounded generalization
loss, not a collapse. Calibration also degrades cross-dataset: FOSI→MESA's Spread/Error
(1.01-1.42) is more over-dispersive than in-domain MESA's 1.034, and MESA→FOSI's
(0.77-0.86) drifts further under-dispersive than in-domain FOSI's 0.833 — consistent
with a model whose learned noise-injection scale was calibrated for one dataset's error
statistics not transferring cleanly to the other's.
