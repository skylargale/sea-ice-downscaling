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
NOISE_SHARED_BIAS=true    # added 2026-09-12 -- root-cause fix for a patchy per-pixel artifact,
                          # also a net accuracy/calibration win, not just cosmetic -- see below
DATA_VARIANT=avg          # confirmed better than interp on both datasets, see below
LAND_THRESHOLD=0.1        # default -- combining with 0.5 tested, does NOT help, see below
EXTRA_LAYER=false
ENSCALE_NET=false
COASTAL_CHANNEL=false
CLASSIFICATION_HEAD=false
ATTENTION_END=false
```
i.e. `qsub -v BATCH_NAME=...,TRAIN_YEARS=...,TEST_YEARS=...,DATA_VARIANT=avg,STOCHASTIC_REFINE=true,NOISE_SIGMA=1.0,NOISE_SHARED_BIAS=true submit_engressnet_daily.sh`

**Checkpoints to actually use for the paper**: `results/MESA_stochastic_refine_sweep_avg_sharedbias/`
(MESA). For FOSI, `results/FOSI_stochastic_refine_sweep_avg_sharedbias/` is a **symlink** (added
2026-09-15) to `results/FOSI_daily_combo_avg_recheck_currentcode_sharedbias/` (the item-10
confirmation rerun, jobs 5910301-304) rather than a dedicated retrain — a dedicated
`submit_stochastic_refine_sweep_mesa_sharedbias.sh`-style FOSI retrain turned out unnecessary:
the recheck already has FOSI run with identical toggles/domain/hyperparameters to the recommended
config, `NOISE_SHARED_BIAS=true` included, and the shared-bias effect on FOSI is essentially flat
(RMSE 0.1802 pre-fix -> 0.1805 with both fixes, vs. MESA's real ~3% win, plausibly because FOSI's
truth is a smooth deterministic single trajectory rather than MESA's per-member-averaged
comparison against a noisy single-realization truth, so the per-member texture artifact barely
shows up in FOSI's metrics either way). The symlink exists purely so any script/notebook that
expects a checkpoint at the "official" path finds one, without spending GPU-hours re-deriving a
result already in hand. Don't use the pre-fix `results/MESA_stochastic_refine_sweep_avg/` or
`results/FOSI_stochastic_refine_sweep_avg/` going forward except for before/after comparison.

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

**Limitation to state explicitly in the paper** (documented 2026-09-15, no new fix attempted):
coastal error is a real, quantified, and currently unsolved weakness of this method, not an
oversight — say so directly rather than letting a reviewer find it. Concretely: the recommended
config's Coastal RMSE (0.189 MESA, 0.296 FOSI, after the shared-bias fix) runs roughly 1.7-2x the
domain-wide RMSE (0.108 MESA, 0.180 FOSI) — errors concentrate specifically at the coast. Three
structurally different fixes (loss-side land-threshold tightening, architecture-side
refiner+land-threshold combo, data-side conservative-vs-bilinear truth regridding) were tried and
all three failed the identical way: a small coastal gain bought at a disproportionate cost to
overall accuracy, with no calibration benefit either. That consistency across three unrelated
intervention types is itself informative — it suggests the coastal error isn't a simple
regridding or loss-weighting artifact fixable by tuning existing knobs, but something more
structural (e.g. the fundamental information loss of downscaling a coarse coastal cell that mixes
land and ocean, or a genuine limit of this receptive-field size at the coastline). Worth framing
as a specific direction for future work rather than an incidental caveat: e.g. an explicit
coastal loss term with a different functional form (not just re-weighting existing MSE/energy
loss), or a boundary-aware architectural change (e.g. the `deep_mask_head` toggle from the
2026-09-12 round already shows a real IIEE win from more mask-aware decoder capacity, at a
calibration cost -- see below -- suggesting the mask-processing-depth axis is more promising than
the three loss/data-side levers already exhausted).

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

- **FOSI**: RMSE 0.2145–0.2184 (±1.8% around the mean), Spread/Error 0.811–0.936 (±11%,
  **corrected 2026-09-15** from a previously-stated 0.843 lower bound that didn't match any of
  the 5 actual seed runs when directly recomputed — the true minimum is seed 0's own
  `results/FOSI_stochastic_refine_sweep_interp/FOSI_refine_interp_2000-2005_2021_5622249.casper-pbs`
  at 0.8114; mean across all 5 seeds is 0.8807).
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
`processing`-adjacent one-off, `submit/evaluation/submit_eval_fullyear_on_season.sh`).
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
   | Refiner alone | FOSI | 0.2161 | 0.3978 | 0.881 (near-ideal) |
   | Attention-end alone | FOSI | 0.2178 | 0.4050 | 1.358 (over-dispersive) |
   | Refiner+attention combo | FOSI | 0.2168 | 0.4027 | 1.076 (over-dispersive) |
   | Refiner alone | MESA | 0.1339 | 0.2534 | 0.986 (near-ideal) |
   | Attention-end alone | MESA | 0.1414 | 0.2464 | 1.229 (over-dispersive) |
   | Refiner+attention combo | MESA | 0.1370 | **0.2586** (worst of the three) | 1.035 |

   Neither dataset shows an accuracy win: FOSI's RMSE is flat, MESA's sits between the
   two ingredients with no improvement, and MESA's Coastal RMSE is actually the worst
   of any of the three configs. Calibration moves in the same direction both times —
   from the refiner's already-near-ideal Spread/Error into mild over-dispersion (FOSI
   0.881→1.076, MESA 0.986→1.035) — the opposite of a calibration win; on MESA it's
   still much better than attention-end alone's poor 1.229, but worse than the refiner
   by itself. **The two toggles don't compose usefully — adding attention-end to the
   refiner is a net negative or neutral change, never a win, on either dataset.**
   (**Corrected 2026-09-15**: FOSI's "Refiner alone" row originally read 0.2174/0.4005/0.843 --
   verified directly against `results/`, this didn't match any actual run, including the 5-seed
   variance set for this exact window/config. Replaced with the mean of that 5-seed set
   (0.2161/0.3978/0.881), the closest verified single-split-equivalent reference actually on
   record; individual seeds range 0.2145-0.2184 / 0.3923-0.4070 / 0.811-0.936, so the qualitative
   conclusion above — refiner+attention moves further from ideal calibration than the refiner
   alone, in every seed — is unaffected by which exact reference point is used.)
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

## Distributional evaluation and baseline (2026-09 review round)

Added in response to a coauthor review asking for a distributional baseline and richer
distributional evaluation, beyond the existing Spread/Error and rank histogram:

- **New permanent evaluation sections** in `evaluation/run_daily_eval_batch.py`: section 16
  (CRPS, the standard proper-scoring-rule metric, broken out by the same SIT-regime bins as
  the Spread/Error section) and section 13b (per-ensemble-member isotropic PSD, checking
  whether individual stochastic draws — not just the ensemble mean, which is all section 13's
  domain-wide PSD comparison covers — reproduce realistic small-scale spatial texture, and
  whether the *spread* of member spectra is itself calibrated). Both apply automatically to
  every future run through this script, not just the ones evaluated below.
- **Distributional baseline**: `evaluation/build_analog_baseline.py`, a nearest-neighbor
  ("analog") method — for each test-time low-res X, finds the K=20 nearest analogs in the
  *training* period by KD-tree and uses their corresponding truth Y fields as the predictive
  ensemble. No training loop; reuses `functions_engressnet.py`'s own load/split/normalize/crop
  path and `compute_metrics_table`/`save_evaluation_data`, so its output is a real run
  (`results/MESA_analog_baseline/`) that every existing evaluation tool works against
  unmodified.

4-window average (MESA, `avg` variant, same methodology as the rest of this document):

| Metric | Stochastic UNet (recommended) | Analog baseline (K=20) | Ratio |
|---|---|---|---|
| RMSE | 0.1111 | 0.2700 | 2.43x worse |
| Coastal RMSE | 0.1938 | 0.4404 | 2.27x worse |
| Pattern Corr | 0.8783 | 0.6654 | 0.76x |
| SSIM | 0.9344 | 0.8380 | 0.90x |
| Spread/Error (1.0 = ideal) | 1.034 | 1.341 | more over-dispersive |
| CRPS, domain-wide (m) | 0.0350 | 0.0844 | 2.41x worse |
| CRPS, open water / thin / moderate / thick ice (m) | 0.0058 / 0.0506 / 0.0731 / 0.0955 | 0.0148 / 0.1319 / 0.1788 / 0.2212 | ~2.3-2.6x worse in every regime |

For reference, bilinear (identical baseline in both batches, since it doesn't depend on the
model) scores RMSE 0.2581 / Coastal RMSE 0.2651 — **the analog baseline is worse than plain
bilinear interpolation** on both, despite being a genuinely distributional method: matching on
the coarse low-res field doesn't guarantee the historical analog's fine-scale structure
resembles the current one. The stochastic UNet beats the analog baseline by a consistent
2.2-2.6x on every proper-scoring/calibration metric, in every SIT regime.

## Deterministic-vs-stochastic accuracy/calibration trade-off — write-up for the paper (2026-09-15)

**A real, quantified accuracy-vs-calibration trade-off, not a free lunch.** The plain
Deterministic UNet (same backbone, no `stochastic_refine`) beats the Stochastic UNet's own
ensemble mean on raw accuracy, on both the pre-fix and shared-bias-fixed checkpoints:

| Checkpoint | Deterministic RMSE | Stochastic-mean RMSE | Relative gap |
|---|---|---|---|
| Pre-fix (original recommended, MESA) | 0.1022 | 0.1111 | ~9% |
| **Shared-bias fix (current recommended, MESA)** | **0.1042** | **0.1080** | **~3.6%** |

The shared-bias fix narrowed this gap by more than half (9%→3.6%) as a side effect of fixing the
patchy-bias artifact — worth mentioning as an incidental benefit, though calibration (not
accuracy) was the fix's actual target. The gap is still larger than the ±2% seed-run-to-run noise
band quantified above, so it's a real, structural cost, not sampling noise.

**CRPS does not rescue this, and a reviewer could make the same point independently.** CRPS
reduces exactly to MAE for a point forecast — there's no separate "CRPS" to compute for a
deterministic model, its MAE already is the number that would go in that column. On the pre-fix
checkpoint, the Deterministic UNet's MAE (0.031) was *lower* than the stochastic ensemble's real
CRPS (0.035 domain-wide) — meaning a naive side-by-side of "CRPS" would have nominally favored
the point forecast. (The equivalent comparison on the shared-bias-fixed checkpoint needs the
CRPS rerun in progress, item 1 above, to state precisely — update this table once that lands.)

**The actual, defensible case for the stochastic method has to rest on calibration evidence
specifically, not on CRPS or RMSE.** The Deterministic UNet has no Spread/Error at all — not a
tie on calibration, a structural inability to represent uncertainty in the first place. The
paper's argument for the method should lean on the reliability diagram (per-threshold Brier
scores), rank histogram (domain-wide flatness), and Spread/Error (near-ideal calibration, ~0.97
post-fix) — i.e., "the ensemble tells you when to trust it, which a point forecast structurally
cannot" — rather than implying the stochastic method is simply more accurate, which the raw
numbers don't support and a careful reader would notice.

**This isn't a one-off quirk of `stochastic_refine`** — the same shape (calibration gain, real
accuracy cost) was independently seen with `attention_end` in the toggle sweep above, and again
with `deep_mask_head` in the 2026-09-12 round (IIEE improves, Spread/Error worsens). Three
unrelated architectural changes producing the same trade-off shape is reasonable evidence this is
a genuine property of pushing calibration in this architecture family, not a fixable artifact of
any one toggle — worth stating as a general finding, not just a caveat specific to the
recommended config.

## Known issue found 2026-09-11: recommended checkpoints predate the noise-shared-bias fix

`functions_engressnet.py`'s `LocallyConnected2d`/`noise_bias_smoothness_penalty` docstrings
document a root-cause fix (2026-08-26) for a persistent, non-random, spatially-rough texture
that survives even `stochastic_refine`'s fully deterministic (eps=0) pass, caused by its
noise-mixing layer's per-location bias having no smoothness constraint — fixed by
`--noise-shared-bias` (ties that bias to one shared vector instead of one independent value per
grid cell). **Confirmed directly against `model_state_dict.pt`**: both
`results/MESA_stochastic_refine_sweep_avg/` (written 2026-08-17) and
`results/FOSI_stochastic_refine_sweep_avg/` (same era) have `local_noise_mix.bias` shaped
`(46500, 4)` — one independent value per grid cell, i.e. **pre-fix**, nine days before the fix
landed. This plausibly explains this round's per-member PSD finding above (members' spectra
were suspiciously similar to each other, with an unexplained excess of high-wavenumber energy
in the member mean vs. truth) as a symptom of this specific bug rather than generic
under-dispersion.

**Retrain complete (2026-09-15) — a real win, adopted as the new MESA ★ recommendation.** Full
numbers (RMSE, Coastal RMSE, IIEE, Spread/Error) are in "Shared-bias / deep-mask-head /
noise-mix results" below, rather than duplicated here. Headline: RMSE 0.1111→0.1080 (−2.8%),
calibration 1.034→0.972 (closer to ideal from the other side). **FOSI's equivalent checkpoints
still have not been submitted** — `results/FOSI_stochastic_refine_sweep_avg_sharedbias/` does not
exist yet; the MESA-only result should not be assumed to carry over to FOSI. **CRPS-by-regime and
per-member PSD spread (sections 16/13b) also have not yet been re-run against this batch** — do
that before claiming the texture-calibration bug is fully resolved, not just RMSE/Pattern
Corr/Spread-Error.

Two more architecture experiments launched alongside it, same 4-window MESA/avg methodology,
each testing one toggle in isolation on top of the shared-bias fix:
- `submit/training/submit_deep_mask_head_sweep_mesa.sh` (`--deep-mask-head`, into
  `results/MESA_deep_mask_head_sweep_avg/`) -- adds a 2-layer conv block processing the
  high-res land mask right after it's concatenated in, giving the network real depth to learn
  coastal-aware behavior instead of one bare 3x3 conv. Not the same idea as `coastal_channel`
  (a low-res proxy fed to the *encoder*, already tested with no effect, see the toggle sweep
  above) -- this deepens processing of the real high-res mask at the *decoder's* resolution.
- `submit/training/submit_noise_mix_none_sweep_mesa.sh` (`--noise-mix-kernel none`, into
  `results/MESA_noise_mix_none_sweep_avg/`) -- drops `LocallyConnected2d` (the confirmed source
  of the patchy-member artifact, see above) entirely rather than just fixing its bias: no
  learned spatial-mixing layer at all, just the existing fixed-kernel `smooth_noise()` applied
  to independently-drawn noise channels. Zero new learnable parameters. Motivated by a direct
  visual check (2026-09-11): a baseline run with no `LocallyConnected2d` at all
  (`stochastic_refine=False`, `results/MESA_daily_combo_avg/MESA_el0_dmed_es0_2000-2005_2021_5529180.casper-pbs/ensemble_figure.png`)
  shows no patchy per-pixel texture, while `stochastic_refine=True` runs do -- pointing at
  `LocallyConnected2d` specifically, not just its bias term. A third option,
  `GaussianNoiseMix` (`--noise-mix-kernel gaussian`, a genuinely distance-weighted *learned*
  alternative -- one shared Gaussian kernel per channel, only a learnable smoothing-scale
  parameter, so it cannot produce a sharp per-pixel discontinuity by construction), is
  implemented and available but deliberately not launched yet -- only worth its extra
  complexity if `none`'s fixed, manually-chosen smoothing scale proves too rigid. (Launched
  anyway shortly after, into `results/MESA_noise_mix_gaussian_fixed_sweep_avg/` -- see results
  below.)

Results for all three (RMSE/Coastal RMSE/IIEE/Spread-Error, none beats the plain shared-bias
fix) are in "Shared-bias / deep-mask-head / noise-mix results" below, not duplicated here.

**Infrastructure bug found and fixed while launching these (2026-09-11)**: all three sweeps
initially failed immediately at import time (`ModuleNotFoundError: No module named
'member_metrics'`), before any GPU time was used. Root cause: `functions_engressnet.py` needs
`evaluation/member_metrics.py` as a sibling import, but `submit/training/submit_engressnet_daily_mesa.sh`/
`submit_engressnet_daily.sh` (the general-purpose per-run templates every sweep driver calls)
never put `evaluation/` on `PYTHONPATH` -- they only worked historically because the original
recommended-config sweep (2026-08-17) predates the 2026-08-25 stage reorg that split
`functions_engressnet.py` and `member_metrics.py` into separate folders. A handful of one-off
scripts (e.g. `submit_mesa_noisesharedbias_kernel9.sh`) happened to set `PYTHONPATH` manually
and so kept working after the reorg; the general templates did not, and apparently no one ran
the general sweep-driver path for a MESA/FOSI job between the reorg and today. Fixed in both
templates by resolving `evaluation/` relative to the script's own real file location (not
`$PBS_O_WORKDIR`, which varies by invocation convention) and exporting it via `PYTHONPATH`
before `cd`ing. All three sweeps were resubmitted after the fix.

## Potential things to check (open, as of 2026-09-11)

1. ~~Confirm the noise-shared-bias retrain actually changes the numbers~~ — **RESOLVED
   2026-09-15, MESA only.** It does: RMSE 0.1111→0.1080, Coastal RMSE and IIEE both improve too
   (see "Shared-bias / deep-mask-head / noise-mix results" below), not cosmetic.

   **CRPS-by-regime rerun** (sections 16/13b, now against `results/MESA_stochastic_refine_sweep_avg_sharedbias/`):
   domain-wide CRPS is essentially unchanged (0.0350 pre-fix -> 0.0358 post-fix, within noise) and
   per-regime values move similarly little (open water 0.0058->0.0062, thin 0.0506->0.0516,
   moderate 0.0731->0.0735, thick 0.0955->0.0972) — the shared-bias fix's accuracy/calibration
   gains (RMSE, Coastal RMSE, Spread/Error, IIEE) don't show up as a CRPS improvement, consistent
   with CRPS folding sharpness and calibration into one number that can move in offsetting
   directions.

   **Per-member PSD spread rerun**: the specific hypothesis that motivated the fix — that the
   "members are suspiciously similar to each other, member-mean overshoots truth's high-wavenumber
   energy" finding was a *symptom of the bias bug* — is only **partially confirmed**. At the
   shortest resolved wavelength (~23 km): truth=0.000339, member-mean pre-fix=0.000499 (47%
   excess), member-mean post-fix=0.000472 (39% excess) — a real but modest reduction, not the
   fix this was hoped to be. The member-to-member spread itself (p10-p90 band) is **still razor
   thin post-fix** (e.g. 0.000470-0.000473 at that wavelength) — essentially unchanged from
   pre-fix. So: `noise_shared_bias` fixed the *pixel-level* spatial-domain calibration
   (Spread/Error 1.034->0.972, a real win) but did **not** fix the *spectral-domain* one —
   individual members still produce near-identical spatial textures to each other. This lines up
   with the noise-mix ablation finding above (calibration needs per-location learned mixing, not
   just an unbiased version of it) — the bias was one real bug with a real fix, but member
   textural diversity looks like a separate, still-open problem, possibly requiring an
   architectural change (e.g. per-member-conditioned mixing) rather than a bug fix.
2. ~~FOSI's `stochastic_refine` checkpoints need the same retrain~~ — **LIKELY NOT NEEDED FOR THE
   NUMBERS, checked 2026-09-15.** The el0_dmed_es1 confirmation reruns (item 10 below) happen to
   use identical toggles/domain/hyperparameters to the official recommended config, and one of
   those reruns already had `NOISE_SHARED_BIAS=true` applied on FOSI
   (`results/FOSI_daily_combo_avg_recheck_currentcode_sharedbias/`, jobs 5910301-304): RMSE
   0.1802 (pre-fix) → 0.1805 (both fixes) — essentially flat, unlike MESA's real ~3% win.
   Plausible mechanism: MESA's evaluation is per-member-averaged against a noisy
   single-realization truth, so it's sensitive to the per-member texture artifact the bug
   caused; FOSI's truth is smooth/deterministic, so the artifact barely shows up in its metrics.
   **Not fully closed** — this is evidence from an equivalently-configured proxy run, not a
   dedicated `FOSI_stochastic_refine_sweep_avg_sharedbias` batch at that exact path. Worth doing
   only if something downstream expects a checkpoint at that specific path (provenance/pipeline
   hygiene), not for the numbers themselves.
3. **Coastal RMSE remains genuinely unsolved.** Three independent fixes (land-threshold 0.5,
   refiner+land-threshold, conservative-vs-bilinear regrid) all failed the same way (small
   coastal gain, disproportionate RMSE cost elsewhere) — worth trying a fundamentally different
   angle (e.g. an explicit coastal auxiliary loss term beyond the existing distance-based
   `coastal_boost` weighting) rather than another sweep of the same three ideas, or reporting it
   as an explicit, honest limitation.
4. **The deterministic-vs-stochastic RMSE/CRPS trade-off should be reported explicitly**, framed
   via the reliability/rank-histogram/spread-skill evidence (not CRPS or RMSE alone) — see
   "Distributional evaluation" above. Don't let a reviewer find this tension first.
5. **Gaussian mean + learned-variance baseline** (the other distributional baseline the
   coauthor suggested, alongside the analog method) has not been built. Decide whether it's
   worth the training-compute cost given how decisive the analog-baseline result already is.
6. ~~PIOMAS overlap is 0/2190 for every recommended-config test window~~ — **RESOLVED
   2026-09-15.** Real daily 2021 PIOMAS data was already sitting in scratch
   (`/glade/derecho/scratch/skygale/PIOMAS_daily/PIOMAS_hiday_2021.nc`, same native grid as the
   monthly 1978-2020 campaign file `run_daily_eval_batch.py` already used). `load_piomas()` now
   additively concatenates it (the read-only campaign file itself is untouched) — coverage for
   test=2021 windows is now 2190/2190, not 0/2190. Full PIOMAS comparison tables and spatial
   snapshots were regenerated for the recommended (sharedbias) MESA checkpoint:

   | Method | MAE vs. PIOMAS (m) | RMSE vs. PIOMAS (m) | Bias vs. PIOMAS (m) |
   |---|---|---|---|
   | Truth (MESA) | 0.3832 | 0.6171 | -0.2932 |
   | Bilinear | 0.3648 | 0.5777 | -0.2882 |
   | Deterministic UNet | 0.3846 | 0.6189 | -0.2926 |
   | Stochastic UNet Mean | 0.3799 | 0.6094 | -0.2917 |

   (Single window shown, 2000-2005->2021; all 4 windows now have this table under
   `saved_figs/MESA_stochastic_refine_sweep_avg_sharedbias/`.) Every method shows a substantial,
   similar-magnitude negative bias against PIOMAS (~-0.29 m) including bilinear and even the
   "Truth" (MESA's own single-realization target) — this reads as a systematic MESA-vs-PIOMAS
   offset (different models, different assimilation/forcing) rather than something the
   downscaling method is responsible for, since it's present even in the coarse bilinear
   baseline. Not a strong claim either way on relative skill: RMSE differences between methods
   here (0.578-0.619 m) are small relative to the absolute PIOMAS-vs-MESA gap, so this table is
   better framed as a sanity/consistency check than a decisive skill comparison. **Minor labeling
   bug fixed alongside this**: `run_daily_eval_batch.py` hardcoded the truth-panel label as
   "Truth (FOSI)" regardless of which dataset the batch actually was — fixed to plain "Truth"
   (the table above and its underlying CSV/figure predate this label fix, hence the "Truth
   (FOSI)" wording surviving in already-generated output for a MESA batch; re-running would pick
   up the corrected label, not worth a rerun for a label-only change).
7. ~~The analog baseline's K=20 was chosen to match the model's own ensemble size, not
   independently tuned~~ — **RESOLVED 2026-09-15.** K sensitivity checked directly (K=5, 10, 20,
   50, 4-window MESA avg each):

   | K | RMSE | Coastal RMSE | Pattern Corr | Spread/Error |
   |---|---|---|---|---|
   | 5 | 0.2594 | 0.4223 | 0.6753 | 1.057 |
   | 10 | 0.2641 | 0.4304 | 0.6704 | 1.219 |
   | **20 (reported)** | 0.2700 | 0.4404 | 0.6654 | 1.341 |
   | 50 | 0.2795 | 0.4556 | 0.6571 | 1.453 |

   RMSE and Coastal RMSE get monotonically *worse* as K increases (more, less-similar analogs
   pull the ensemble mean toward climatology), and Spread/Error moves monotonically further
   into over-dispersion (more analogs -> wider, less-precise spread). So K=20 is not the most
   *flattering* choice for the analog baseline -- K=5 is both more accurate and much
   better-calibrated (1.057, close to ideal) than K=20. But the qualitative conclusion is
   robust to this choice: even at K=5's best-case numbers, RMSE (0.2594) is still barely
   different from bilinear's 0.2581 (i.e. still not a real win over the trivial baseline), and
   Spread/Error (1.057) is still worse-calibrated than the stochastic UNet's 0.972. No K value
   makes the analog method competitive with the actual method on any axis -- reporting K=20
   (matching the model's own ensemble size, the natural apples-to-apples choice) is defensible
   as the primary number, with this table available as the robustness check if a reviewer asks.
8. **`submit/evaluation/submit_daily_eval_batch.sh` hardcodes `cd "$PBS_O_WORKDIR/../.."`**,
   which only resolves correctly if it's submitted from `submit/evaluation/` specifically —
   `submit/README.md` currently describes this class of script as "safe to invoke either way,"
   which isn't true for this one. Worth fixing the script (resolve relative to its own file
   location, not `$PBS_O_WORKDIR`) or correcting the README so the next person doesn't get a
   silent wrong-directory failure.
9. ~~The "infrastructure bug... fixed" claim above (PYTHONPATH/`member_metrics` import) is not
   actually confirmed fixed~~ — **RESOLVED 2026-09-12.** The first fix attempt (resolving
   `evaluation/` relative to the script's own `BASH_SOURCE` location) looked correct interactively
   but still failed under real PBS execution, because PBS spools/copies a submitted script at
   `qsub` time — `BASH_SOURCE[0]` at runtime pointed at that spool copy, not the real file under
   `submit/training/`, so the computed path was wrong. Fixed for real by hardcoding the absolute
   path instead (consistent with every other path in these scripts already being absolute). All
   16 jobs (4 windows x 4 experiments: shared-bias, deep-mask-head, noise-mix-none,
   noise-mix-gaussian_fixed) resubmitted after this fix completed successfully end-to-end
   (jobs 5907651-5907666) — see the results table below.
10. **Full-history audit (2026-09-11) against every batch in `results/`** (57 batches, 338 runs,
    every `metrics.csv` aggregated) surfaced one lead not previously documented: `el0_dmed_es1` in
    `results/{FOSI,MESA}_daily_combo_avg/` — identical toggles, domain, and hyperparameters to the
    ★ recommended config (`stochastic_refine=true, avg, medium domain, no extra layer`) — scores
    **MESA RMSE 0.096 / Pattern Corr 0.905** (vs. the currently-recommended checkpoint's 0.111 /
    0.878) and **FOSI RMSE 0.180 / r 0.945** (essentially tied with the recommended checkpoint's
    0.180 / 0.946). On raw numbers this beats the current recommendation on MESA outright. **Not
    adoptable as-is**: job IDs (~5529180-203) place this run *before* both known code fixes
    (smooth-noise, ~job 5573xxx; shared-bias, 2026-08-26) — it should carry the same texture/
    calibration bugs those fixes targeted, which makes its unexpectedly strong RMSE hard to trust
    at face value rather than a reason to prefer it. Worth an explicit rerun of this exact config
    under current (fully-fixed) code once items 1-2 above are resolved, specifically to check
    whether the fixes give up some of this apparent MESA accuracy in exchange for the calibration
    gains they're meant to buy — rather than assuming the fixes are a free win on every axis.
    **Confirmation rerun submitted 2026-09-11**: `submit/training/submit_daily_combo_avg_recheck_currentcode.sh`
    resubmits this exact config (`extra_layer=false`, medium domain, `avg`, `stochastic_refine=true`,
    `noise_shared_bias` left at its default `false` — deliberately isolating just the smooth-noise
    fix, not conflated with the separate shared-bias retrain already queued above), both datasets,
    all 4 windows, under current code, into `results/{FOSI,MESA}_daily_combo_avg_recheck_currentcode/`
    (jobs 5910115-5910122, queued behind the item-9 sweeps as of submission — verify their logs
    import cleanly once they start, same caveat as item 9). **Second confirmation submitted same
    day**: identical sweep with `NOISE_SHARED_BIAS=true` added (both fixes together, not just
    smooth-noise), into `results/{FOSI,MESA}_daily_combo_avg_recheck_currentcode_sharedbias/`
    (jobs 5910301-5910308) — submitted in parallel with the smooth-noise-only rerun rather than
    waiting on it first, since the queue is deep enough that serializing the two would cost a
    full extra queue cycle for a question (does el0_dmed_es1's edge survive both fixes) that's
    worth asking regardless of the smooth-noise-only result.

    **RESOLVED 2026-09-15**: both confirmation reruns completed cleanly (no import errors, all 8
    jobs). The apparent MESA advantage does not survive under current code — it was entirely an
    artifact of the pre-fix bugs, not a missed opportunity:

    | Config | RMSE (MESA) | Pattern Corr (MESA) |
    |---|---|---|
    | Original `el0_dmed_es1` (pre-fix code) | 0.096 | 0.905 |
    | Same config, rerun under current code (smooth-noise fix only) | 0.111 | 0.873 |
    | Same config, rerun with shared-bias fix too | 0.109 | 0.877 |
    | Recommended config (shared-bias fix) | 0.108 | 0.879 |

    Once rerun under current (fixed) code, `el0_dmed_es1` converges to statistically the same
    performance as the actual recommended config (both RMSE and Pattern Corr) — confirming the
    old batch's better-looking number was a bug artifact, and that the current recommended
    config is not leaving free accuracy on the table. FOSI showed no discrepancy to begin with
    and reruns confirm the same (~0.180-0.181 RMSE across original and both rechecks). This item
    is closed.

## Figures for the paper (2026-09-15)

All distributional-evaluation figures for the recommended (shared-bias) MESA checkpoint should
be taken from `saved_figs/MESA_stochastic_refine_sweep_avg_sharedbias/_4window_aggregate/` --
these pool the raw per-cell data from all 4 train windows (via additive sufficient statistics:
bincounts/sums/counts accumulated one window at a time, not naive averaging of already-computed
summary numbers) before recomputing each statistic, since all 4 windows evaluate on the identical
2021 test year and only differ in training years -- built by
`evaluation/build_4window_aggregate_figures.py` (`submit/evaluation/submit_4window_aggregate_figures.sh`
to run it, CPU-only, ~128-192GB, do not run on a login node -- an interactive attempt was
OOM-killed): `14_rank_histogram_4window.png`, `14_reliability_diagram_4window.png`,
`15_spread_skill_by_regime_4window.png`, `16_crps_by_regime_4window.png`,
`13_psd_comparison_4window.png`, `13b_psd_per_member_spread_4window.png`. Pooled numbers match
the earlier simple 4-window-average numbers elsewhere in this doc almost exactly (e.g. CRPS
domain-wide 0.0358 both ways) -- a useful internal consistency check, not a coincidence, since
all 4 windows have equal N.

For spatial/map figures, which don't pool across windows meaningfully (each is a snapshot),
`MESA_refine_avg_sharedbias_2000-2005_2021_5907651.casper-pbs/` is a representative single-window
choice for `03_ensemble_figure.png`, `04_error_figure.png`, `12_piomas_spatial_snapshot.png`, etc.
-- results are consistent across windows per the audit above, so any window works, but stay
consistent about which one is used across figures in the same paper draft.

**Note**: a stray incomplete run directory (`MESA_refine_test_2000-2005_2021_5907648.casper-pbs`,
an early bug-hunting attempt with only `run_config.json`/`description.txt`, no `eval_data/`) sits
in `results/MESA_stochastic_refine_sweep_avg_sharedbias/` -- harmless (the aggregate script
filters for `eval_data/fields.npz` presence, matching `run_daily_eval_batch.py`'s own
convention), but worth a manual cleanup pass if the batch directory is archived/shared as-is.

## Shared-bias / deep-mask-head / noise-mix results (2026-09-12)

4-window MESA/avg averages, all four new experiments compared against the pre-fix recommended
config:

| Batch | RMSE | Coastal RMSE | IIEE (ice-edge err) | Spread/Error (1.0=ideal) |
|---|---|---|---|---|
| Pre-fix (original recommended) | 0.1111 | 0.1938 | 0.3312 | 1.034 |
| **Shared-bias fix** | **0.1080** | **0.1888** | **0.2936** | **0.972** |
| Deep mask head (+shared bias) | 0.1091 | 0.1884 | 0.2431 | 1.129 |
| Noise-mix: none | 0.1109 | 0.1885 | 0.3944 | 1.370 |
| Noise-mix: gaussian_fixed | 0.1112 | 0.1868 | 0.2367 | 1.369 |

**The shared-bias fix is a clean win and should become the new baseline going forward** — better
RMSE, better coastal RMSE, IIEE down 11%, and calibration moved closer to ideal (1.034→0.972).
Not just a cosmetic fix for the patchy-member artifact; it measurably helps accuracy and
ice-edge representation too.

**Deep-mask-head is a genuine trade-off, not a strict win**: IIEE drops further (0.294→0.243,
another 17%, consistent with the intended effect of giving the decoder real depth to use coastal
geometry), but calibration gets worse (0.972→1.129, now over-dispersive).

**Dropping `LocallyConnected2d` entirely (`none`, `gaussian_fixed`) makes calibration
meaningfully *worse* than even the pre-fix buggy version** (~1.37 vs. 1.034) — a real, useful
negative result. The calibration benefit isn't coming from "spatially-correlated noise" in
general; it's specifically tied to having a *per-location, learned* mixing weight (which
`shared_bias` keeps, only removing the buggy degree of freedom), not to smoothing shape or
amount. Between the two removal variants, `gaussian_fixed` clearly beats `none` on every other
metric (best coastal RMSE and IIEE of all five batches) — consistent with the bell-curve-shape
hypothesis being right about spatial texture, just not sufficient to recover calibration.
`--noise-mix-kernel gaussian` (learned width, not yet launched) is deprioritized by this finding
— since calibration seems to require per-location flexibility a single learned scalar-per-channel
can't provide, it's unlikely to close the calibration gap either, though not yet directly tested.

**Recommended config update**: `NOISE_SHARED_BIAS=true` should be added to the recommended
config going forward. `results/MESA_stochastic_refine_sweep_avg_sharedbias/` and
`results/FOSI_stochastic_refine_sweep_avg/` (FOSI retrain still pending, see item 2 above) are
the checkpoints to use for the paper, not the original pre-fix `results/*_stochastic_refine_sweep_avg/`
directories.
