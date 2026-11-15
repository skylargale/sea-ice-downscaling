# Recommended configuration (final model)

The final EngressNet model (FINAL2, adopted 2026-11-14 during the co-author revisions), the model of the submitted paper (FINAL1),
then the record of what did not work. Superseded configurations and the older experiment write-ups are in
`/glade/derecho/scratch/skygale/downscaling/archive/records/config_history.md` (written `records/` below; scratch, so a text copy is
in the project's `archive/records_text_2026-11-13.tar.gz`). All revision-round numbers: `results/revisions_2026-11/summary.md`.

## Final model (FINAL2): one model per dataset, 1,825 random training days

```
DATA_VARIANT=avg
TRAIN_YEARS=2000-2020 TRAIN_RANDOM_DAYS=1825 RANDOM_DAYS_SEED=0   # 1825 dates (5 years' worth) drawn at random from 2000-2020; 2021 never drawn
TEST_YEARS=2021
NOISE_AMP_MAP=true        # per-pixel noise amplitude from each decoder stage (state-dependent spread)
ENERGY_PATCH_SIZE=10      # energy score over 10x10 px tiles (~110 x 45 km)
NOISE_LR_MULT=10          # noise-stage widths/amplitudes + amplitude-map heads train at 10x the base lr (~0.0096)
EMA_DECAY=0.9995 (MESA) / 0.997 (FOSI)   # exponential moving average of the weights (~last 2-3 epochs)
K=20 K_EVAL=20 BETA=0.8 NUM_EPOCHS=20 SEED=0
COARSE_ZERO=true          # evaluation only (--coarse-zero), see below
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
All FINAL2 training runs: `training/submit_final2_training.sh [MESA|FOSI|both] [--submit]` (EngressNet, the matched
deterministic U-Net = det-MSE + EMA with `DETERMINISTIC_MSE=true,K_EVAL=2`, and seeds 1-9 of that U-Net for the 10-model
seed-ensemble baseline, all on the same 1,825 days). MESACLIP uses the same 1,825 dates for all 6 members (10,950 samples), FOSI
1,825 samples. Evaluation and `model_state_dict.pt` use the EMA weights; the last-step weights are `model_state_dict_last_step.pt`.
Submit-script defaults stay off so older runs reproduce.

Inference with a FINAL2 checkpoint on other inputs (`--test-x-path/--test-y-path`, `NUM_EPOCHS=0`, `INIT_CHECKPOINT`) needs
`NOISE_AMP_MAP=true` and the training run's random-day settings plus `RANDOM_DAYS_EXCLUDE_YEARS=2021`, so the normalization
statistics come from the same 1,825 dates (`observations/inference/submit_final2_obs_inference.py`).

Batches (`results/`): `FINAL2_{MESA,FOSI}_random1825` (EngressNet), `FINAL2_{MESA,FOSI}_detmse_random1825` (deterministic),
`FINAL2_{MESA,FOSI}_detmse_seedens` (seeds 1-9), `PIOMAS_2018-2022_FINAL2_FOSI{,_detmse}` (ICESat-2 scoring),
`PIOMAS_2010-2023_cdrv6_FINAL2_FOSI` (application). Evaluation: `evaluation/revisions/submit_final2_eval_chain.sh`,
`submit_final2_metrics.sh`, then all figures with `figures/v2/submit_make_v2_figures.sh` (code: `figures/v2/make_v2_figures.py`); scripts select the model with `FINAL_SET=final2`
(`evaluation/revisions/final_set.py`; default `final2` since 2026-11-15, `final1` = the old model; outputs written with final2 carry a `_final2` suffix). Paper figures:
`figures/v2/` (old-model versions in `figures/v2/superseded_final1/`).

Why random days (revision F, 3 seeds each, same number of training days): sampling the whole 2000-2020 range beats one
consecutive 6-year window on both datasets with non-overlapping seed ranges (RMSE -3.2% MESACLIP / -6.8% FOSI, CRPS -3.5% / -7.6%,
FOSI spread-skill 0.86 -> 1.00); the gain comes from more varied ice states. Why each other piece (details in
`records/config_history.md`): `avg` input beats `interp` on both datasets; 10 px tiles gave the best perfect-model spread-skill
without adding noise texture; NOISE_LR_MULT is needed because the noise parameters never moved from their initial values otherwise;
EMA fixed the noisy last-step checkpoint (MESACLIP 2010-15 window RMSE 0.124 -> 0.106 m).

Final numbers (test year 2021, ocean cells, coarse-zero rule, original truth; `tables23_three_methods_final2.csv` in
`results/revisions_2026-11/`):

| | RMSE | CRPS (det/bil: MAE) | IIEE 0.15 m | pattern r | spread-skill |
|---|---|---|---|---|---|
| MESACLIP EngressNet | 0.1022 | 0.0308 | 0.0209 | 0.879 | 1.13 |
| MESACLIP deterministic | 0.0998 | 0.0424 | 0.0211 | 0.904 | - |
| MESACLIP bilinear | 0.1636 | 0.0686 | 0.0389 | 0.882 | - |
| FOSI EngressNet | 0.1989 | 0.0755 | 0.0305 | 0.939 | 0.98 |
| FOSI deterministic | 0.1920 | 0.1011 | 0.0307 | 0.947 | - |
| FOSI bilinear | 0.3051 | 0.1575 | 0.0530 | 0.885 | - |

- CRPS vs deterministic MAE -27% / -25%; ensemble-mean RMSE +2.4% / +3.6% vs deterministic. Most of that RMSE gap is finite-ensemble
  noise: MSE(K) = MSE(inf) + var/K gives K = inf RMSE 0.0990 / 0.1943 (det 0.0998 / 0.1920). Kept at K = 20 (user decision
  2026-11-15; K = 100 reporting and a mean-branch architecture not pursued).
- Regime spread-skill (thin / moderate / thick; open): MESACLIP 1.19 / 1.08 / 1.08; 1.96, FOSI 1.03 / 0.96 / 0.98; 1.20.
  Truths outside the ensemble 10.5% / 10.7% (ideal 9.5%).
- Seed ensemble (10 det-MSE U-Nets, same days): CRPS 0.0326 / 0.0785 (EngressNet -5.5% / -3.9%), ensemble-mean RMSE 0.0943 / 0.1816
  (better), spread-skill 0.39 / 0.36 (badly under-dispersed). EngressNet costs 1.7x / 1.5x the training of the 10-model ensemble.
- ICESat-2 (FOSI model on PIOMAS, winters Oct 2018-May 2022, 161,376 QC pass-cells): CRPS 0.367 vs deterministic MAE 0.4245
  (-13.5%) vs bilinear 0.4188 (-12.4%); RMSE of the mean ties (0.579 / 0.580 / 0.572).

## Inference rule and evaluation (2026-10-24, both models): coarse-zero rule, original truth, 20 members

**Coarse-zero rule.** At inference, members (and the det-MSE prediction) are set to exactly 0 on ocean cells whose bilinear coarse SIT is
exactly 0 (`coarse_zero_mask` in `training/functions_engressnet.py`). The `avg` coarse input is the area mean of the native field and SIT
>= 0, so the truth is 0 there: in 2021 it never exceeds 4e-5 m in those cells, which are 14% (FOSI) / 29% (MESACLIP) of ocean cells.
FINAL1 outputs got it post hoc with `evaluation/apply_coarse_zero_fields.py` (every other member value unchanged); FINAL2 and all
inference pass `COARSE_ZERO=true`. It also removes rare large spurious coastal values (up to ~7 m in a few summer members).

**Original truth.** `Y_test_phys` is saved after a normalize/de-normalize round trip (float32 error up to 1e-6 m): about a quarter of
true zeros came back as +-2e-9 m (MESACLIP 2015-20 window: 33% of ocean truth negative), which ranked below or above members that are
exactly 0 and made most of the open-water rank-0 spike. `evaluation/restore_raw_truth.py` rewrites it from the Y files; every
perfect-model batch (FINAL1, FINAL2, revisions) has been through it, and any new one needs it before rank histograms are trusted.
Only rank histograms moved (FINAL1 MESACLIP pooled rank 0 3.3x -> 1.17x of uniform, open water 4.7x -> 1.16x).

**Evaluation size.** K_EVAL=20 throughout (the K=100 batches were deleted 2026-10-24).

## Previous final model (FINAL1, the submitted paper)

Same configuration as FINAL2 except the training data: four consecutive 5-year windows (2000-05 / 05-10 / 10-15 / 15-20, all
dates), one model per window, numbers averaged over the windows. Batches `results/{FOSI,MESA}_{final,detmse}_ema`,
`PIOMAS_2018-2022_FOSI_{final,detmse}_ema`, `PIOMAS_2010-2023_cdrv6_emaFOSI_4win`, selected with `FINAL_SET=final1`
(the default is `final2` since 2026-11-15). Paper figures: `figures/final_figures/`.

| (4-window mean) | RMSE | CRPS (det: MAE) | IIEE | pattern corr | spread-skill |
|---|---|---|---|---|---|
| MESACLIP EngressNet | 0.1052 | 0.0319 | 0.075 | 0.884 | 1.02 |
| MESACLIP det-MSE | 0.1060 | 0.0447 | 0.091 | 0.908 | - |
| FOSI EngressNet | 0.2065 | 0.0797 | 0.058 | 0.937 | 0.86 |
| FOSI det-MSE | 0.2056 | 0.1085 | 0.076 | 0.938 | - |

Regime spread-skill (thin / moderate / thick / open): MESACLIP 1.07 / 0.99 / 0.98 / 1.77, FOSI 1.00 / 0.88 / 0.84 / 1.35. Open water
is over-dispersed in the mean-square sense only: far from the edge (coarse < 1 cm) the ratio is 3.5-3.9 but those cells carry 2-9% of
the open-water squared error, and the open-water rank histogram is near uniform. ICESat-2 (PIOMAS input, 2018-22): CRPS 0.366 vs
det 0.388 vs bilinear 0.419, spread/error 0.45. The deterministic U-Net gained more than EngressNet from FINAL2's random days
(RMSE 0.1060 -> 0.0998 / 0.2056 -> 0.1920 vs 0.1052 -> 0.1022 / 0.2065 -> 0.1989), so FINAL1's "ensemble mean ties the
deterministic U-Net" no longer holds.

Tried for the open-water rank-0 spike before the truth fix, not adopted (records in `evaluation/corrected_calibration/`):
- hurdle presence head, 3 variants (`hurdle*_compare.txt`): moved the excess to the top ranks, missed ice x10, less spread in ice
- coarse-zero rule inside the training loss (`coarse_zero_compare.txt`): same open-water gain, spread -2 to -5% in every ice regime,
  CRPS +1%
- 1-deg block-mean conservation rescaling (no-training check): more cells outside the ensemble in every regime
- 25 km concentration input (T3): fixed near-edge open water and cut CRPS 13-15%, dropped because the input is not readily available


### What didn't work (and why)

Record of approaches that were tried and rejected, bugs that were found and fixed, and limitations we kept, all in one place.
Written 2026-10-09, extended with the 2026-11 revisions. The adopted model is FINAL2 above (noise cascade + amplitude maps +
10 px energy-score tiles + noise-parameter learning rate x10 + EMA, 1,825 random training days, 20 members).

Unless stated otherwise, "ICESat-2" scores are against QC'd ICESat-2 pass-cell means (0.1 deg cell, same day, PIOMAS input,
2018-2022; `observations/engressnet_obs_evaluation.ipynb`), and "perfect model" is FOSI or MESACLIP test year 2021. Spread-skill and
spread/error: 1 = calibrated. Lower CRPS is better.

#### Earlier history (documented in `records/config_history.md`)

| Topic | Section in `records/config_history.md` |
|---|---|
| Coastal bias not fixed by land threshold, attention, deep mask head | "What did NOT resolve the coastal-bias problem" |
| Conservative regridding of the target | "Data variant: conservative regridding ... resolved, negative" |
| Season-only vs full-year training | "Season vs. full-year training -- resolved, negative" |
| Toggle sweep (no standout) | "Toggle sweep: no other standout" |
| Shared-bias, deep mask head, noise-mix variants | "Shared-bias / deep-mask-head / noise-mix results" |
| `interp` vs `avg` input | "Data variant: `avg` beats `interp`" |
| Patchy members from `LocallyConnected2d` (EnScale-lite refiner) | "Why the smooth-noise-fixed EnScale-lite refiner ..."; replaced by the Gaussian noise cascade (2026-09-16) |

#### Model changes tested against ICESat-2 (2026-09-29 to 2026-10-08)

| Test | What it was | Result | Why rejected |
|---|---|---|---|
| **T1** obs fine-tuning | Fine-tune on CS2SMOS ~1 deg block CRPS (2011-17) | ICESat-2 bias -0.35 m; CRPS 0.398 vs 0.368 baseline; perfect model slightly worse | Learned CS2SMOS's thin bias relative to ICESat-2 |
| **T3** high-res predictors | + 25 km concentration, log depth, distance to land | Perfect model CRPS -11% (0.0753), RMSE 0.186; ICESat-2 CRPS 0.366 (= baseline), less fine-scale variance (20-40 km 0.07) | Gain did not transfer to observations; needs extra inputs at application |
| **T4** pooled years + time context | One 2000-2020 window, 2 days of past predictors, 4 seeds | Perfect model better (RMSE 0.193); ICESat-2 CRPS 0.378 (worse), within-cell variability 0.33 | Worse against observations |
| **Tile size** 30 / 15 / 5 px | Energy score over N x N px tiles (T2) | Perfect spread-skill 0.75 / 0.75 / 0.81 vs **0.82 at 10 px**; ICESat-2 CRPS 0.365 / 0.366 / 0.366 vs 0.364 | 5 px puts ~4x CICE's 20-40 km variance in (noise texture, worse placement r 0.31); 15 px no better than 30 |
| **1b** wide final-stage noise kernel | 31-tap kernel at the 0.1 deg stage + noise lr x10 | Perfect spread-skill 0.89 (= 1a); ICESat-2 CRPS 0.375 vs 0.371 (1a) | Learned width stayed ~1.6 px, so the extra reach was unused; slightly worse CRPS |
| **Option 2** noise calibrated on ICESat-2 | Mean frozen, only noise parameters trained against ICESat-2 (2 folds by winter, scored on held-out winters) | ICESat-2 CRPS **0.304**, spread/error 1.05, but within-cell variability 2.0x and 20-160 km variance ~6x ICESat-2; perfect model over-dispersed (1.7) and RMSE 0.21 -> 0.29 | Calibrated point-wise only by adding unrealistic fine-scale noise (members no longer realistic fields); close to post-hoc inflation |
| **Option 3** FOSI + MESA-trained members pooled | Pool two training datasets | CRPS 0.371 -> 0.368; spread/error 0.45 -> 0.45; less fine-scale variance | Models agree too closely to add spread; kept only as a robustness result (training data barely matters) |
| **Smooth output floor** | softplus (2 cm scale) instead of clamp at 0, in the training loss and at output | FOSI: domain spread-skill 0.89 -> 0.68; ice regimes 1.05/0.92/0.85 -> 0.79/0.73/0.65; ICESat-2 CRPS 0.363 -> 0.372; within-cell variability 0.63 -> 0.38; open-water rank-0 spike 3.5x -> 8.8x | Model shrank its noise everywhere; softplus never reaches 0 m, so ice-free truth always ranks below all members. MESA runs cancelled |

#### 2026-11 co-author revisions (numbers in `results/revisions_2026-11/summary.md`)

| Test | What it was | Result | Outcome |
|---|---|---|---|
| **H** noise mechanism | Noise multiplying the features (`mult`, SPPT-like; also at the bottleneck) or added to them (`add_feat`), vs the paper's concatenation; MESACLIP 2015-20, 3 seeds | CRPS ties (0.0318-0.0321); mult over-dispersed (spread-skill 1.14) with slightly worse RMSE (+0.7%) and less realistic members; add_feat = concat. Learned mult factor not SPPT-like (median \|r\| ~1e3 at the coarsest stage) | Rejected; concat kept. Code removed 2026-11-15 |
| **H.1** noise ablation | One stage's noise off at inference | Noise is used mainly at the coarsest (d3) and 0.1 deg stages; one middle stage is ignored (FOSI d2, MESACLIP d1) | Diagnostic (`--noise-ablate` kept) |
| **E.2** MC-dropout | Det-MSE U-Net with Dropout2d p = 0.1 in the decoder, 20 dropout passes | Spread-skill 0.19-0.20, about half the truths outside the ensemble; CRPS 18% / 15% worse than EngressNet | Baseline only (FINAL1 tables); code removed 2026-11-15 |
| **E.1** seed ensemble | 10 det-MSE U-Nets (FINAL2: seeds 0-9 on the same days) | Better ensemble-mean RMSE (-8%) but spread ~1/3 of the error; CRPS 4-6% worse | Appendix baseline |
| **F** random days | 2190 random dates from 2000-2020 vs consecutive 2015-20 | RMSE -3% / -7%, CRPS -4% / -8% | **Adopted** (FINAL2, 1,825 days) |
| **G** extrapolation | MESACLIP trained 2000-10, tested 2015-21 | Keeps ~40% of the RMSE and ~56% of the CRPS reduction over bilinear at every gap; spread-skill 0.90-1.00 | Robustness result |

#### Rejected without running

- **Post-hoc, scale-selective variance inflation** (2026-10-06): rejected as not physically grounded. Spread must come from training on real
  targets or from real data.
- **Synthetic input perturbation during training** (2026-10-06): rejected for the same reason.
- **ICESat-1 (GLAS) and the 2003-2008 period**: not used at all (2026-09-29); the related data were removed on purpose.

#### Data and evaluation choices that did not hold up

- **Petty GPSat** as an independent input or validation product: it is built from ICESat-2 freeboards, so it is flagged
  "not independent" and kept out of the pooled product.
- **IS2SITMOGR4** (ICESat-2 monthly) agrees best with the reference (RMSE 0.36 m) only because it is the same instrument.
- **PIOMAS as the reference** (until 2026-10-06): replaced by ICESat-2 along-track pass-cells, because PIOMAS is a model and is
  EngressNet's own input.
- **10 km ICESat-2 sections** (`_sm` files): replaced by full-resolution segments grouped per 0.1 deg cell and pass.
- **Median of per-cell ratios** for the CICE ceiling: replaced by pooled within-cell variance. The median made CICE look like ~1/3
  of ICESat-2; pooled, FOSI CICE is 66% and MESACLIP 43% (sub-grid bar cell in `figures/v2/make_v2_figures.py`).
- **Results emails** from Casper (mailx): looked undelivered on 2026-10-09; confirmed working 2026-10-17 (`evaluation/submit_email_summary.sh`).

#### Bugs found and fixed

- **Noise parameters never trained** (found 2026-10-07): widths and amplitudes stayed at their initial values in every run before
  the final model, so T2's state-dependent spread was effectively off. Fixed by the noise learning rate x10 (adopted).
- **Bilinear IIEE = 0.0**: asymmetric epsilon between normalising and de-normalising; fixed in `functions_engressnet.py`.
- **Energy-score tiling** (2026-10-06): tile sizes that don't divide the field (15 px) dropped the remainder columns (the last 16 of
  310 never entered the loss; RMSE 0.64 there). Fixed with zero-weight padding; 30 / 10 / 5 px were unaffected.
- **T2 out of memory**: the full-resolution amplitude head doubled the backward peak; the final-stage amplitude is now computed from
  the 1 deg (d1) features and upsampled.
- **T3 inputs** missing 2000-01-01: the auxiliary file needed the 1999 file too.
- **`/glade/work` quota full** (2026-10-09): three MESA smooth-floor checkpoints were truncated on save (16 KB instead of 54 MB).
  Check `gladequota` before long runs.

#### Limitations we kept (state in the paper)

- **Ensemble-mean RMSE (FINAL2)**: +2.4% (MESACLIP) / +3.6% (FOSI) vs the deterministic U-Net trained on the same days, mostly the
  var/K cost of a calibrated 20-member mean (K = inf estimate within -0.8% / +1.2% of deterministic). The FINAL1 analysis below
  explains the mechanism.
- **Accuracy cost on MESACLIP (FINAL1, 4 windows)**: the final model's ensemble mean is less accurate than the deterministic model there (RMSE 0.113
  vs 0.108 m, pattern correlation 0.850 vs 0.888; the earlier under-dispersed cascade matched it at 0.108 / 0.905). FOSI is
  unaffected (0.211 vs 0.209 m, r 0.932 vs 0.936). Bias stays ~0 (+0.004 m).
  **Cause (diagnosed 2026-10-17)**, from saved MESACLIP 2021 fields, ocean cells. "K=inf" is the unbiased estimate
  mean[(ens mean - truth)^2 - ens var / K]; RMSE over K = 1, 2, 5, 10, 20 member subsets follows MSE(K) = MSE(inf) + var/K exactly:

  | Window | spread-skill | ens-mean RMSE K=20 | K=inf | det-MSE | gap mainly from |
  |---|---|---|---|---|---|
  | 2000-05 | 1.11 | 0.1101 | 0.1068 | 0.1076 | finite ensemble only |
  | 2005-10 | 1.07 | **0.1056** | **0.1027** | 0.1075 | none (ensemble mean wins) |
  | 2010-15 | 0.98 | 0.1238 | 0.1209 | 0.1086 | thick ice (>1 m): 0.232 vs 0.196, 88% of the gap, every month |
  | 2015-20 | 1.18 | 0.1123 | 0.1085 | 0.1067 | positive mean bias: open water +0.012, thin +0.026, moderate +0.028 m; open-water spread-skill 2.2; Sep-Nov about 2x det-MSE |

  So there are three separate causes:
  1. **Finite ensemble.** A calibrated ensemble's 20-member mean keeps var/K of member noise. The earlier cascade was
     under-dispersed (0.70-0.75), so it paid less of this cost, which is why it matched det-MSE (K=inf 0.106-0.108 per window).
  2. **Window-to-window training variability.** The 2010-15 window's mean is worse at any K.
  3. **Positive mean bias**, mostly in the 2015-20 window. This is the network's own mean, not the clamp: the reported ensemble
     mean (`Y_pred_phys`) already averages unclamped members and clamps afterwards, and the mean of clamped members gives the
     same RMSE (0.1101 vs 0.1107 in 2000-05). So "average before clamping" is already in place and is not a remedy.

  - Not the cause: member texture. A 3x3-smoothed ensemble mean is worse (0.123 in 2000-05), and the ensemble median equals the mean.
  - In expectation an energy-score ensemble mean and an MSE model estimate the same conditional mean, so an RMSE tie is the
    expected outcome. The probabilistic gain is CRPS vs det-MSE MAE.
  - Diagnostic: `evaluation/diag_ensmean_vs_det.py` (retired 2026-11-13, in the code snapshot); per-regime/per-month numbers in `records/corrected_calibration/old_experiments/MESA_final_avg_ensmean_diag.csv` (cascade: `MESA_noise_cascade_avg_ensmean_diag.csv`).
  - Remedies being tested (2026-10-17): K_EVAL=100 on the existing checkpoints (`results/MESA_final_avg_K100/`,
    `results/CAL1a_noiselr_FOSI_K100/`), and retraining with EMA weights (`--ema-decay`, 0.9995 MESA / 0.997 FOSI) alone and
    with a debiased ensemble-mean MSE term (`--mean-mse-weight` 1 and 5): `results/{MESA,FOSI}_final_{ema,emamm1,emamm5}/`.
    Second sweep, all with EMA + cosine lr decay (`--lr-schedule cosine`): noise-lr x1 for 3 epochs first
    (`--noise-lr-warmup-epochs 3`) `final_ema_cos_nw3`, + 40 epochs `_e40`, + beta 1.2 `_b12`, one 2000-2020 window (6 epochs)
    `_pooled`; det-MSE controls with the same changes `detmse_ema_cos{,_e40,_pooled}`.

- **Open-water over-dispersion**: with the clamp, noise over ice-free cells can only add ice, so members carry a few cm of one-sided
  spurious ice. Open-water spread-skill 1.44 (FOSI) / 2.24 (MESACLIP); rank-0 spike 3.5x / 6.5x. It averages out in the ensemble
  mean (open-water RMSE 0.026-0.041 m). Ice regimes are near-calibrated (MESACLIP 1.17 / 1.03 / 0.99 thin / moderate / thick).
- **CICE ceiling**: EngressNet learns sub-grid variability from CICE. Pooled within-1 deg variability vs ICESat-2: FOSI CICE 66%,
  MESACLIP 43%, final EngressNet 59% (on PIOMAS), det-MSE 31%, PIOMAS 16-22%. In the perfect model EngressNet matches CICE (pooled
  ratio 1.05; more in smooth cells, 0.89 in CICE's most variable 1%).
- **Input error dominates** against observations: input choice changes CRPS ~10x more than model choice. Pooling independent inputs
  (PIOMAS + CS2SMOS + TOPAZ4-ML) is the remaining lever (CRPS 0.31, spread/error 0.64).
- **No coastal validation**: only 17 QC'd ICESat-2 pass-cells lie within 10 km of land (908 within 10-25 km).
- **Winter only**: ICESat-2 thickness is Oct-Apr.

#### Cleanup record (2026-10-09)

- `/glade/work` hit its 2 TiB quota. Version1-5 moved to `/glade/derecho/scratch/skygale/downscaling/archive/versions/` (absolute paths
  inside them rewritten). Superseded Version6 results were deleted: T1-T4, 5 / 15 / 30 px tile runs, CAL1b, CAL2, older-model
  input swaps, PIOMAS 2021-only and 2010-2023 old-baseline runs, era / older-window tests, appendix sensitivity raw runs, smooth-floor
  runs. Their summary figures and tables remain in `evaluation/saved_figs/`, `evaluation/corrected_calibration/` and
  `observations/data/alongtrack_eval/`, and their ICESat-2 samples remain on scratch
  (`/glade/derecho/scratch/skygale/observations/ICESat2_IS2SITDAT4_cells/eval/`).
- Code for those experiments was moved to `_to_delete_py/`; the opt-in training flags (`--obs-ft-*`, `--aux-path`,
  `--time-context`, `--noise-kernel-size-final`, `--train-only-noise`, `--output-floor`) stay in the code, default off.
#### Cleanup record (2026-11-13)

- Code snapshot of everything before the cleanup: `records/code_snapshot_2026-11-13.tar.gz` (all scripts, notebooks, docs, small
  records; no `.nc`/`.npz`/`.parquet`, results or logs). Restore one file with
  `tar -xzf records/code_snapshot_2026-11-13.tar.gz ./path/to/file`. A text-only copy (no figures, notebook outputs stripped) is
  `archive/records_text_2026-11-13.tar.gz` in the project, in case scratch is purged.
- Deleted (only pointed at deleted batches or rejected experiments): `evaluation/{diag_ensmean_vs_det.py,compare_all_batches.ipynb,
  submit_compare_all_batches.sh,spread_inflation/}`, `training/submit_{noise_cascade_sweep_{mesa,fosi},pretrain_mesa_then_fosi}.sh`,
  seven old `figures/final_chain/` job chains, `figures/{submit_fig7_final.sh,build_engressnet_on_products.py}`,
  `sensitivity/obs_eval_tests/{submit_calibration_options.py,build_is2_calib_targets.py}`, the stale
  `processing/build_X_Y_from_{FOSI,MESA}-HR_daily.ipynb` (the `.py` scripts are the production versions), six old along-track
  `.npz` files in `observations/data/alongtrack_eval/` (T1-T4, scratch FOSI, det-MSE without EMA; ~260 MB), two byte-identical
  weight files (`weighted_grids/tx0.1v2_to_latlon_{0.1deg,1deg}.nc` = `ice_hr_to_{0p1deg,1deg}.nc`, 771 MB), PBS logs from before
  2026-11-01.
- Moved: one-off analyses to `evaluation/diagnostics/`; `compare_output_floor.py` renamed `evaluation/regime_table.py`
  (+ `submit_regime_table.sh`); `build_input_swap.py` (+ submit) to `processing/`; metric records for deleted batches to
  `evaluation/corrected_calibration/old_experiments/` (then to `records/`, see below).
- Flag pruning, deferred until the 2026-11 revision runs were finished: done 2026-11-15, see the next record. Plan as written
  2026-11-13: Checked 2026-11-13: every flag below has its default value in every kept and revision run config and is not set
  by any revision sweep. **Prune:** `--obs-ft-x-path/-target-path/-weight/-only`, `--train-only-noise`, `--crps-weight`
  (obs fine-tuning, option 2), `--output-floor`, `--output-floor-scale` (smooth floor), `--time-context` (T4),
  `--aux-path`, `--test-aux-path`, `--static-path` (T3), `--lr-schedule`, `--noise-lr-warmup-epochs`, `--mean-mse-weight`
  (2026-10-17 sweep), `--noise-kernel-size-final` (1b), `--attention-end`, `--attn-window-size`, `--attn-num-heads`
  (coastal-bias tests), `--patches` and its `--context-size/--target-size/--stride` (patch mode; the final model is
  sub-domain only), plus the matching env vars in both submit templates. Optional: `--collapse-wind-vector` (MESA->FOSI
  pretraining, dropped). **Keep:** `--late-mask-fusion`, `--noise-channels`, `--noise-kernel-size` (sensitivity appendix),
  the revision flags, `--months`, `--train-frac`, `--init-checkpoint`. Careful: `ENERGY_PATCH_SIZE` / `energy_loss_patched` are
  the final model's 10 px energy tiles, not patch mode. Verify by training 1 epoch with defaults before and after (same seed)
  and comparing the checkpoint and `fields.npz` bit for bit, plus the final-config smoke test.
  Constraints from the revisions session (2026-11-14): keep `--test-x-path`/`--test-y-path` (U10 check),
  `--init-checkpoint` with `NUM_EPOCHS=0` inference, and `--grad-accum-steps` (new train_model branch; the bit-identity check must
  cover `train_model`, not just `UNet`). `evaluation/revisions/mult_clip_check.py` sets `context_size`/`target_size` from
  run_config.json; remove that line if those fields go. Don't edit `functions_engressnet.py` while revision jobs that import it
  are queued or running.

#### Training-code pruning (2026-11-15, after the revisions round)

- Removed from `training/functions_engressnet.py`, `train_engressnet.py` and both submit templates: patch mode (`--patches`,
  `--context-size/--target-size/--stride`, `extract_patches`; `--no-patches` is still accepted and ignored, the `--lat/--lon`
  bounds are now required), observation fine-tuning (`--obs-ft-*`, `--train-only-noise`), `--crps-weight`, the softplus output
  floor (`--output-floor*`), `--time-context`, auxiliary inputs (`--aux-path/--test-aux-path/--static-path`), `--lr-schedule`,
  `--noise-lr-warmup-epochs`, `--mean-mse-weight`, `--noise-kernel-size-final`, windowed attention (`--attention-end`,
  `--attn-*`, `WindowedSelfAttention2d`), and the rejected revision options `--noise-mode mult/add_feat`,
  `--noise-at-bottleneck`, `--noise-feat-init`, `--dropout-p`, `--mc-dropout-eval`. The library went from 3110 to ~2450 lines.
- Kept: `--late-mask-fusion`, `--noise-channels`, `--noise-kernel-size` (appendix), `--deterministic-mse`, the final-model flags,
  `--train-random-days/--random-days-seed/--random-days-exclude-years`, `--grad-accum-steps`, `--noise-ablate`, `--months`,
  `--train-frac`, `--test-x-path/--test-y-path`, `--init-checkpoint`, `--collapse-wind-vector`.
- Verified bit-identical against the pre-pruning code (CPU, same seed, FOSI): checkpoints, EMA and last-step weights, loss
  history, metrics and every `fields.npz` array, for four cases: paper EngressNet flags; det-MSE + random days; inference from
  the FINAL2 FOSI checkpoint through the cross-dataset path with random-day exclusion and noise ablation; grad accumulation +
  late mask fusion + 2 noise channels + kernel 3.
- Old checkpoints of removed options (REV_H mult/add_feat, REV_E2 MC-dropout) no longer load; their outputs and metrics stay.
- Retired to scratch `records/revisions_2026-11/` (text copy in the project's `archive/records_text_2026-11-15.tar.gz`):
  `training/{submit_revisions_sweeps.sh,revisions_smoke_test.sh}` (drivers of the H/F/E/G runs, partly using removed flags) and
  `evaluation/revisions/{mult_clip_check.py,submit_mult_clip_check.sh}` (needs the mult code; result kept in
  `results/revisions_2026-11/mult_clip_check.csv`). New: `training/submit_final2_training.sh`.
- Pre-pruning training code: in `records/revisions_2026-11/training_preprune_2026-11-15/`.

#### Records and unused data moved to scratch (2026-11-13, second pass)

- Project -> `/glade/derecho/scratch/skygale/downscaling/archive/records/`: `code_snapshot_2026-11-13.tar.gz`, `config_history.md`,
  `corrected_calibration/old_experiments/` (150 metric records of deleted batches), `sensitivity/submit/` (appendix sweep
  drivers; the raw runs are gone), `paper_snippets/` (`methods_reconstruction_addition.tex`, `computational_cost_appendix.tex`),
  `observations/` (tile-sweep notebook, calibration-option / output-floor / tile-sweep CSVs, 26 figures of the T1-T4, tile-sweep,
  calibration-option and output-floor tests), `figures/` (superseded Fig. 7b, FOSI wind check). Text copy on work:
  `archive/records_text_2026-11-13.tar.gz` (1.1 MB).
- Scratch `downscaling/` reorganized: the per-year PIOMAS application inputs (2010-2023, cdrv6) are in `piomas_obs/` (the three
  PIOMAS builders write there via `PIOMAS_DIR`; `submit_piomas_2010-2023_inference.py` and the PIOMAS 2010-2023 run configs
  were repointed). Data no live code reads moved to `retired/` (~1.3 TB, nothing deleted; every move listed in
  `retired/MANIFEST_moves_2026-11-13.txt`): the CESM3-alpha JRA-prescribed run, monthly perfect-model sets (Cambridge Bay,
  Kivalina, FOSI monthly), MESA RCP8.5-only and cross-eval files, the 1960-80 / 1980-2000 era windows, conservative 2015-2021
  targets, PIOMAS 2020 and CDR-v4 2021 inputs, rejected obs-test inputs (T1 CS2SMOS targets, option-2 ICESat-2 folds, T3
  PIOMAS aux), the T2 / T2p10 pooled products. Only the empty `.ipynb_checkpoints/` was deleted.

#### Scratch folder layout (2026-11-13, third pass)

`/glade/derecho/scratch/skygale/downscaling` is now grouped by purpose (see its `README.md`): `training_data/` (perfect-model
pairs; `mesa_hist_1920-2005/`, `conservative_targets/`), `application_inputs/` (`piomas_2010-2023/` = the former `piomas_obs/`,
`obs_inputs_2017-2022/` = the former `obs_eval_tests/`), `ancillary/` (`maps/`, `observed_products/` = the former
`Observations/`), `outputs/` (`pooled_obs_product/` = the former `obs_product/`, `revisions_2026-11/`, `backups/`) and
`archive/` (`records/`, `retired/`, `versions/`). All project code and all 304 scratch paths in `results/**/run_config.json` were
rewritten. The old root paths remain as symlinks (`archive/legacy_links_2026-11-13.txt`) for jobs queued before the move;
`archive/remove_legacy_links.sh` removes them once the queue has drained.
