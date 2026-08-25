# Paragraph 7: Observational Application

**Compiled 2026-08-24/25, updated 2026-08-25.** Covers the full investigation from initial setup
through four fix attempts, following the FOSI-trained (JRA55-forced) recommended EngressNet
checkpoint's application to real PIOMAS input. All figures referenced below are in
`results/PIOMAS_obs_2020/` (`01_`-`14_`), all raw data/CSVs in the individual run directories
cited throughout.

## 1. Setup

The recommended checkpoint (`STOCHASTIC_REFINE=true, NOISE_SIGMA=1.0, DATA_VARIANT=avg`,
trained 2015-2019 on FOSI simulation only — see `recommended_config.md`) was applied via pure
inference (`--calibrate-from <checkpoint> --num-epochs 0`, no gradient updates) to real-world
input for all of 2020, the last year of PIOMAS data on glade:

- **hi_d (SIT)**: PIOMAS v2.1 daily thickness, downloaded directly from PSC
  (`pscfiles.apl.washington.edu/zhang/PIOMAS/data/v2.1/hiday/`), converted and validated against
  the official glade mirror (correlation 0.99999 against `PIOMAS.hi.1978-2020.nc`).
- **aice_d (concentration)**: NSIDC/NOAA CDR daily sea ice concentration (no PIOMAS-native
  concentration product covering 2020 was found on glade).
- **u_10/v_10 (winds)**: the same JRA55 forcing files FOSI's own training X was built from,
  deliberately unchanged, to keep the network's non-ice-state inputs apples-to-apples with
  training.

This is the first point in the project where predictor and target no longer share an
underlying simulation — "closing the loop" from the perfect-model experiments.

## 2. A real bug, found and fixed: PIOMAS land-bleed contamination

PIOMAS's raw file has **no land flag** — land and zero-ice ocean are both encoded as the
literal value `0.0`, with no `NaN` anywhere. Every regrid in this project used `skipna=True`,
which only skips `NaN`; since PIOMAS has none, land cells were never excluded from bilinear
interpolation, and every coastal destination cell was pulled toward zero by real land values in
its stencil.

**Confirmed** against PSC's own official land/ocean mask (`io.dat_360_120.output`,
downloaded from the same PSC utilities directory): at Kivalina specifically, every `hi==0`
native cell in the contaminated stencil is confirmed land (perfect match); globally, ~77% of
`hi==0` cells are land and ~23% are genuine ice-free ocean.

**Fixed** by masking confirmed-land cells to `NaN` before regridding
(`processing/build_X_Y_PIOMAS_obs_2020_daily_v2_landfixed.py`), so `skipna=True` actually
excludes them. Every downstream file/result below uses this land-fixed data
(`X_PIOMAS_obs_2020_daily_v2_interp.nc`, `Y_PIOMAS_obs_2020_daily_v2.nc`) unless noted.
**All qualitative conclusions in this document were re-verified against the fixed data and
survived the fix** — absolute error numbers shrank (real signal was being contaminated), but
no conclusion changed.

## 3. The self-consistency trap

The first metrics table produced (scoring the network against PIOMAS regridded to the target
grid, since no independent truth existed yet) showed bilinear beating the network on every
metric. **This is not a skill comparison** — both the network's input and that "truth" trace
back to the same PIOMAS field, so naive bilinear upsampling trivially resembles its own target.
Labeled explicitly as a self-consistency diagnostic, not an accuracy score, in every run
directory's `SELF_CONSISTENCY_NOTE.txt`.

## 4. Independent validation: CryoSat-2

Found and integrated **CryoSat-2 RDEFT4** (NASA Goddard), a genuine satellite altimetry
retrieval, as real independent truth — `/glade/derecho/scratch/skygale/CryoSat2_RDEFT4/`.
Coverage caveats, quantified (Figure 09):

- **May-September: zero data**, every year — altimetry can't retrieve thickness through melt
  ponds/wet snow. Half the year is simply unobservable by this product.
- **Even in-season, coastal coverage lags interior** by a consistent, large margin (e.g.
  Jan-Apr: 50% of coastal-band cells covered vs. 72% of interior cells), every available month.
- Practical result: **Jan-Apr 2020 is the only reliable, full-coverage window**; Oct-Dec are
  small, partial, non-overlapping samples (Figure 10 uses Jan-Apr only for this reason).

**Result (Figure 02, 03)**: within-month spatial correlation between the network's output and
CryoSat-2 is weak-to-negative everywhere. Pooling across months gives a misleadingly positive
number — that's just both products agreeing on the seasonal thickening trend, not on where
thick/thin ice actually sits at a given time.

**Ruled out "just a resolution mismatch" (Figure 04)**: coarsening the comparison from native
0.1° up to ~1° (CryoSat's own footprint scale and beyond) did **not** improve correlation — it
got slightly worse. The disagreement is a real coarse-scale pattern mismatch, not hidden fine
detail being unfairly penalized.

## 5. Root cause: PIOMAS's own coastal bias

Compared PIOMAS's raw 1° input directly against CryoSat-2, bypassing the network and bilinear
entirely (Figure 05, 10). Found a **spatially coherent coastal band where PIOMAS runs
1.0-1.8 m thinner than CryoSat-2**, with much better agreement in the open-ocean interior — not
noise: month-to-month bias-pattern correlation was 0.48 (land-fixed) to 0.55 (pre-fix), still
well above zero, and Figure 10's four individual monthly maps show the same band in the same
place every month. Consistent with PIOMAS's coarse ~22 km grid failing to resolve landfast/
coastal-compressed ice, and with literature flagging the Beaufort/Chukchi coast as a documented
hot spot of SIT-product disagreement.

**The network amplifies this bias rather than correcting it** (Figure 06): worse than bilinear
in 6 of 7 months in the coastal band, ~17% worse RMSE on average (land-fixed data) — plausibly
because `coastal_boost=2.0` (loss upweighting near the coast during FOSI training) taught the
network to be assertive about coastal ice patterns learned from FOSI, which doesn't transfer to
PIOMAS's specific bias direction.

## 6. How much does accuracy degrade, fairly measured? (Figure 08)

Bilinear was never a fair comparison partner (zero learned parameters, zero training-distribution
exposure). The more relevant number: how much does the network's own accuracy degrade from its
best case (FOSI-trained, FOSI-tested) to the real-world case, vs. how much bilinear degrades
over the same transition.

| Scope | Network degradation | Bilinear degradation |
|---|---|---|
| Domain-wide | 3.33x | 1.24x |
| Coastal | 3.74x | 2.01x |
| Interior | 2.87x | 1.10x |

The network degrades 1.5-1.9x more than bilinear in every scope going out-of-domain — a
specific, quantified overfitting signature.

## 7. Calibration is separately, more severely broken (Figure 07)

Everything above graded the ensemble **mean**. Checking whether CryoSat-2's true value falls
inside the network's own K=20 ensemble's 5-95% band (the fair standard for a *probabilistic*
model) gives a much sharper result: coverage of **1.5-2.8%** against an ideal ~90%. The network
isn't just wrong on real input — it's *confidently* wrong. This is a distinct failure mode from
the accuracy gap above (the Spread/Error≈0.8 calibration number that looked fine throughout
this project measured the model against PIOMAS-as-its-own-truth; against real independent
truth it collapses).

## 8. Five fix attempts (Figures 11, 14)

| Fix | Target | Coastal ratio (network/bilinear) | Verdict |
|---|---|---|---|
| Bias-correct PIOMAS input | Accuracy | 1.54x (worse than baseline 1.17x) | **Failed** — widened the relative gap even though both methods improved in absolute terms |
| `coastal_boost=1.0` retrain | Accuracy | 1.14x | Marginal improvement |
| Domain-randomization retrain | Accuracy | 1.13x (interior: ~1.02x) | Best of the three training-based fixes, still doesn't fully close the gap |
| Conservative regrid of PIOMAS input | Accuracy (regridding fidelity, not the model) | 1.11x (0.898m network vs. 0.809m bilinear) | Best single fix; also best bias comparison (4/7 months) of anything tested |
| **Domain-randomization + conservative regrid (combined)** | Accuracy | **1.095x** (0.886m network vs. 0.809m bilinear) | **Best RMSE ratio overall**, but worse bias comparison (2/7 months) than the conservative regrid alone — see below |

Domain-randomization (`--sit-bias-aug-prob 0.5 --sit-bias-aug-mag 1.5`, new
`functions_engressnet.apply_sit_bias_augmentation`): each training sample has a 50% chance of
having its SIT input channel shifted by a random bias up to 1.5 m during training, teaching the
network not to over-trust a systematically-biased low-res input. Best of the three
training-based levers tried for the accuracy problem.

**Conservative regrid — a data-pipeline fix, not a model fix.** Every PIOMAS input used
everywhere above (including Sections 2-7) was regridded onto the training grid with plain
bilinear interpolation — but FOSI/MESACLIP's own training data uses the "avg" (conservative,
area-weighted) method, built from exact native grid-cell corners. PIOMAS's public distribution
doesn't document its native cell corners, so no conservative regridder existed for it until
`grid.dat.pop` (PSC's standard POP-grid corner file, U-point `ULAT/ULONG`) was located and used
to build one the same way FOSI's own t13 grid is built (`processing/
build_X_Y_PIOMAS_obs_2020_daily_v3_conservative.py`). Rebuilding X with this regridder
(`X_PIOMAS_obs_2020_daily_v3_avg.nc`) and rerunning pure inference against the **original,
unmodified checkpoint** (no retraining) gave the best coastal result of any test in this
document: RMSE ratio 1.11x (vs. 1.17x baseline), and the network's bias is smaller than
bilinear's in 4 of 7 months — every other version tested (baseline and all three training
fixes) showed the network's bias worse than bilinear's in 5-7 of 7 months. PIOMAS's own native
grid has real coastal texture that plain bilinear regridding visibly smoothed away before the
network ever saw it — a genuine information loss in data prep, not a property of PIOMAS's ice
state itself, and not something any amount of retraining on bilinear-regridded input could fix.
(Figure 14 was updated 2026-08-25 to show the *fixed*, conservative-regridded PIOMAS row
alongside FOSI/MESACLIP, once all three shared the same regridding method. Even after this fix,
a smaller residual sharpness gap remains between PIOMAS and FOSI/MESACLIP — a follow-up pilot
[not yet written up here; see Figure 15] found this residual gap is consistent with PIOMAS's own
coarser native resolution rather than a further regridding-method artifact, and separately found
it isn't something MESACLIP training exposure would be expected to fix.)

**Combining conservative regrid with domain-randomization.** Running the domain-randomization
checkpoint (trained/validated only against the bilinear-regridded input) as pure inference
against the conservative-regridded input instead gives coastal RMSE ratio **1.095x** (0.886m vs.
0.809m) — the best of every configuration tested, edging out the conservative regrid alone
(1.11x). The two fixes do stack, though the improvement over the conservative regrid alone is
modest. There's a real tradeoff, though: the bias-comparison metric gets *worse* in this
combined configuration — the network's bias is smaller than bilinear's in only 2 of 7 coastal
months, vs. 4 of 7 for the conservative regrid alone with the original checkpoint. Plausible
explanation: domain-randomization learned to compensate for the *larger* bias present in the
bilinear-regridded input it was trained/tested against; applied to the conservative-regridded
input's smaller bias, it may be mildly overcorrecting. Net read: best headline RMSE number, but
not a clean win — the conservative-regridded input with the **original** (non-domain-randomized)
checkpoint remains the more balanced choice if bias direction matters as much as RMSE magnitude.

## 9. Fixing calibration (Figure 12)

| Fix | Result |
|---|---|
| **Post-hoc spread rescaling** (fit a single multiplier on ensemble spread) | **Success** — 89.3% (coastal) / 90.5% (interior) coverage on **held-out** Mar-Apr, fit only on Jan-Feb. Free, no retraining. |
| Noise-pathway-only fine-tune (`--freeze-backbone`, real CryoSat-2 truth, 8 epochs) | **Failed** — coverage moved from 6.1% to 6.6% on held-out data; gradient descent on ~50 real samples is far too weak to fix a ~100x spread deficit that an analytic scale factor fixes directly. |

The calibration problem turned out to be a simple, large, fixable scale error — not a subtle
shape problem requiring retraining.

## 10. Bottom line

> The FOSI-trained network, applied unmodified to real PIOMAS input, reproduces its in-domain
> pattern correlation and rough magnitude on a self-consistency check, but independent CryoSat-2
> validation reveals (a) a coastal thickness bias inherited from PIOMAS itself and amplified
> rather than corrected by the network, and (b) badly overconfident uncertainty estimates. Of
> four fixes tested, the single best result came not from retraining but from a data-pipeline
> correction — regridding PIOMAS's input with a true conservative regridder matching FOSI's own
> training convention, instead of the plain bilinear regrid used everywhere else in this
> project (1.11x coastal RMSE ratio, best bias comparison of any test) — ahead of the best
> training-based fix (domain-randomization, 1.13x). A simple post-hoc variance rescaling fully
> resolves the calibration problem, entirely separately.
> **Practical recommendation**: use the conservative-regridded PIOMAS input as the default going
> forward (it strictly dominates the bilinear input it replaces, at no retraining cost), combined
> with post-hoc spread rescaling for honest uncertainty. Stacking the domain-randomization
> checkpoint on top of the conservative-regridded input pushes the RMSE ratio slightly further
> (1.095x, the best tested) but at the cost of a worse bias-direction comparison — use the
> **original checkpoint** with the conservative-regridded input if bias direction matters as much
> as RMSE magnitude; use the domain-randomization checkpoint if RMSE alone is the target metric.

## 11. Caveats

- CryoSat-2 itself is an imperfect independent truth at this domain scale: ~25 km footprint,
  known thin-ice retrieval biases (worst in autumn), and systematically sparser coastal
  coverage than interior coverage (Section 4) — some of the "disagreement" measured throughout
  is CryoSat-2's own uncertainty, not solely PIOMAS/network error.
- Test 3's calibrate/validate split (Jan-Feb vs. Mar-Apr) is within-2020, not a genuine
  cross-year holdout — a real limitation, though the held-out-month split is still a legitimate
  train/test separation with no leakage.
- The domain-randomization augmentation perturbs the SIT channel uniformly across the whole
  low-res domain, not specifically the coastal band (a scope simplification, documented in
  `functions_engressnet.apply_sit_bias_augmentation`'s docstring) — a coastal-targeted version
  might do better and hasn't been tried.
- 2021/2022 daily PIOMAS + CryoSat-2 data are downloaded and available on scratch but unused
  beyond Test 3's small within-2020 slice — a natural next step for a genuine cross-year
  calibration/validation split.
- The conservative regrid and the domain-randomization retrain were combined (Section 8) — the
  domain-randomization checkpoint's inference run against the conservative-regridded input gives
  the best RMSE ratio of anything tested (1.095x), but a worse bias-comparison count (2/7 vs. 4/7
  months) than the conservative regrid alone. The two fixes stack on RMSE, but not cleanly —
  which of the two is "better" now depends on whether RMSE magnitude or bias-direction accuracy
  matters more for the intended use.

## 12. The deliverable itself (Figure 13)

Every figure above is a diagnostic or a comparison. Figure 13 is just the product: ensemble-mean
SIT maps for four representative months (including July, which has no CryoSat-2 coverage at
all, to show the product covers months no independent check can touch), plus full-year daily
time series with ±1 ensemble-std bands for Kivalina and Point Hope. Both show a physically
coherent seasonal cycle (spring thickness peak, near-total melt-out by June, autumn freeze-up
visible by December) and the ensemble spread correctly narrows toward zero during the ice-free
summer — a sanity check the network wasn't given directly but reproduces on its own.

## 13. Where everything lives

- **Data**: `/glade/derecho/scratch/skygale/Downscaling_Data/X_PIOMAS_obs_2020_daily_v2_interp.nc`,
  `Y_PIOMAS_obs_2020_daily_v2.nc` (land-fixed, bilinear-regridded, primary through Section 7);
  `X_PIOMAS_obs_2020_daily_v3_avg.nc` (land-fixed, **conservative**-regridded, Section 8's best
  result — same Y); `/glade/derecho/scratch/skygale/PIOMAS_daily/` (raw PIOMAS binaries +
  `utilities/io.dat_360_120.output` land mask + `utilities/grid.dat.pop`, PSC's POP-format
  U-point corner file used to build the conservative regridder);
  `/glade/derecho/scratch/skygale/CryoSat2_RDEFT4/` (96 months, 2010-2024).
- **Build scripts**: `processing/build_X_Y_PIOMAS_obs_2020_daily_v2_landfixed.py` (primary,
  bilinear), `processing/build_X_Y_PIOMAS_obs_2020_daily_v3_conservative.py` (conservative
  regrid), `processing/build_X_PIOMAS_obs_2020_v2_biascorrected.py` (Test 1 input),
  `processing/build_Y_CRYOSAT_2020_daily.py` (Test 3 truth), `processing/convert_PIOMAS_hiday_raw.py`.
- **Code changes**: `functions_engressnet.py` (`apply_sit_bias_augmentation`, `test_months`
  override for cross-dataset eval); `train_engressnet.py` (`--sit-bias-aug-prob/mag`, `--test-months`).
- **Checkpoints**: original — `results/FOSI_stochastic_refine_bilinear_2000_2020_avg/
  FOSI_refine_bilin_avg_2015-2019_2020_5631173.casper-pbs/` (used, unmodified, for the
  conservative-regrid result too); coastal_boost=1.0 —
  `results/FOSI_stochastic_refine_bilinear_2000_2020_avg_coastalboost1/`; domain-randomization —
  `results/FOSI_stochastic_refine_bilinear_2000_2020_avg_sitbiasaug/`.
- **Primary inference run** (land-fixed, CDR-based, bilinear input): `results/PIOMAS_obs_2020/
  FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs/`, with all CryoSat-2
  comparison CSVs/figures in its `cryosat2_validation/` subfolder.
- **Conservative-regrid inference run**: `results/PIOMAS_obs_2020/
  FOSI_recommended_infer_2020_daily_v3_conservative_5717425.casper-pbs/`, comparison script
  `application/observing_cryosat2_check_v3_conservative.py`.
- **Combined (domain-randomization + conservative regrid) inference run**: `results/PIOMAS_obs_2020/
  FOSI_sitbiasaug_infer_2020_daily_v3_conservative_5717788.casper-pbs/`, submit script
  `submit/evaluation/submit_infer_piomas_obs_2020_v3conservative_sitbiasaug.sh`, comparison script
  `application/observing_cryosat2_check_v3conservative_sitbiasaug.py`.
- **Root-cause bias analysis**: `results/PIOMAS_obs_2020/piomas_vs_cryosat_raw_v2_landfixed/`.
- **Figure 14** (native vs. regridded PIOMAS vs. FOSI vs. MESACLIP snapshots):
  `application/build_fig_piomas_fosi_mesa_snapshots.py`.
