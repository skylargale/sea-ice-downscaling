# sea-ice-downscaling

Deep learning stochastic downscaling of Arctic sea ice thickness (SIT) for regional,
coastal sea ice. A coarse CESM/reanalysis-forced "low-res" field is downscaled onto a
high-res target grid with a stochastic UNet trained via an energy-score ("EngressNet")
loss, targeting the Kivalina/Shishmaref/Kotzebue/Nome/Point Hope coastal region of
Alaska.

## Repository Contents

The pipeline is organized into five stages, one per top-level directory:

### `processing/` -- build training data

- **`build_X_Y_from_FOSI-HR_daily.py`** / **`build_X_Y_from_MESA-HR_daily.py`** --
  auto-generated (from the corresponding `.ipynb`; edit the notebook and regenerate,
  don't hand-edit the `.py`) scripts that build the perfect-model low-res/high-res
  training pair at daily frequency: `build_X_Y_from_FOSI-HR_daily.py` from the single
  JRA55-forced FOSI_BGC HR hindcast (t13), `build_X_Y_from_MESA-HR_daily.py` from the
  MESACLIP/CESM-LE HR HIST ensemble (d651007/d651030). Both regrid the native ice
  history onto the 1-degree low-res grid (`interp` and `avg`, saved separately) and the
  0.1-degree high-res target grid.
- **`build_X_Y_from_FOSI-HR_daily.ipynb`** / **`build_X_Y_from_MESA-HR_daily.ipynb`** --
  the editable notebook source for the two `.py` scripts above (previously only the
  generated `.py` output was tracked; the source notebooks are now included too).
- **`build_X_Y_from_FOSI-HR.ipynb`** -- the earlier **monthly**-cadence counterpart to
  `build_X_Y_from_FOSI-HR_daily.ipynb` (no `_daily` in the name). Kept for reference: it
  builds the monthly `Y_FOSI_HR_JRA55.nc` still used by `evaluation/evaluation_plots.ipynb`
  (the non-daily notebook) via its `DEFAULT_Y_PATH`, alongside the newer daily pipeline
  used everywhere else.
- **`build_X_Y_from_MESA-HR_daily_rcp85.py`** -- standalone RCP8.5 continuation of the
  MESA-HR build above (reads d651009/BRCP85C5 instead of d651007/BHISTC5, 2006 onward),
  so training windows can extend past the HIST record's hard stop. Only 6 of the
  9 HIST members have usable daily ice+wind output under RCP8.5 (see the script's
  docstring for which members and why); the combined record uses that same 6-member
  subset throughout, not 9 pre-2006 and 6 after.
- **`stitch_MESA_HR_daily_hist_rcp85.py`** -- stitches the HIST and RCP8.5 daily
  MESA-HR X/Y files into one continuous per-member record spanning the 2006 boundary
  (this pipeline line has no separate `--x-path-future`/`--y-path-future` splicing
  support), remapping each RCP8.5 ensemble index to its matching HIST physical member.
- **`build_Y_FOSI-HR_daily_conservative.py`** / **`build_Y_FOSI-HR_daily_conservative_2000_2020.py`**
  and their MESA counterparts **`build_Y_MESA-HR_daily_conservative.py`** /
  **`build_Y_MESA-HR_daily_conservative_2000_2020.py`** -- rebuild the high-res `Y` truth
  with **conservative** (exact source-cell-area-weighted) regridding instead of the
  production pipeline's bilinear regrid, to test whether truth-side coastal regridding
  noise (a destination cell with a tiny true ocean footprint still getting a full
  thickness value under bilinear averaging) is a real contributor to the model's coastal
  bias. The non-`2000_2020` scripts are scoped to a single baseline split (2015-2020
  train / 2021 test); the `_2000_2020` scripts extend the same conservative regrid across
  the full standard 4-window grid (2000-2004/2005-2009/2010-2014/2015-2019, test 2020),
  matching production splits, and the MESA `_2000_2020` variant replicates the full
  HIST+RCP8.5 stitching pipeline (not just the RCP8.5-only reuse the earlier MESA script
  could get away with, since the 2000-2004 window starts before the 2006 boundary).
  **Result (see `recommended_config.md`): conservative regridding is a net loss on both
  datasets and both scopes** -- a small coastal-RMSE gain bought at a larger overall-RMSE
  cost -- so this is a resolved, closed experiment, not an adopted change to the
  production `Y` files.
- **`submit_build_X_Y_from_FOSI-HR_daily.sh`** / **`submit_build_X_Y_from_MESA-HR_daily.sh`**
  / **`submit_build_X_Y_from_MESA-HR_daily_rcp85.sh`** / **`submit_stitch_MESA_HR_daily.sh`**
  / **`submit_build_Y_FOSI-HR_daily_conservative.sh`** /
  **`submit_build_Y_FOSI-HR_daily_conservative_2000_2020.sh`** /
  **`submit_build_Y_MESA-HR_daily_conservative.sh`** /
  **`submit_build_Y_MESA-HR_daily_conservative_2000_2020.sh`** -- PBS batch wrappers for
  the build/stitch scripts above (`qsub submit_build_...sh`). CPU/memory-only regridding
  jobs (xESMF/pop_tools), no GPU needed.

### `training/` -- train the model

- **`functions_engressnet.py`** -- all pipeline logic: data loading, land-sea masking
  (regridded POP `KMT` ocean mask), sliding-window patch extraction (or single
  sub-domain cropping when patches are disabled), the stochastic residual UNet model,
  the energy-score training loss, evaluation metrics (MAE/RMSE, coastal MAE/RMSE,
  IIEE, spread/error), figure generation, and candidate-coastal-point / domain-mean SIT
  time series output. `run_pipeline()` is the single entry point everything else calls.
  The UNet supports several opt-in, default-off architecture toggles layered on the
  baseline model (see `train_engressnet.py --help` for the matching CLI flags): an
  extra downsample/upsample stage, EnScale-style (Schillinger et al. 2025)
  locally-connected stochastic refinement (either as a single added stage,
  `--stochastic-refine`, or replacing the whole decoder, `--enscale-net`), an explicit
  coastal-fraction input channel (`--coastal-channel`), an auxiliary land/ocean/ice
  classification head (`--classification-head`), windowed self-attention at the end of
  the decoder (`--attention-end`), and a `--calibrate-from`/`--freeze-backbone` path for
  loading a trained checkpoint and recalibrating only its noise-injection pathway.
  Metrics computation (`compute_metrics_table`) now takes `preds_all_phys=None,
  per_member=False`: when `per_member=True` (auto-set by `run_pipeline` whenever the
  output directory is a MESACLIP run), the "Stochastic UNet Mean" row is computed as
  each ensemble member scored against truth, then averaged, rather than the smoother
  ensemble-mean prediction scored against truth, MESACLIP's saved truth is a single
  CESM1 realization, not a deterministic target, so the naive ensemble-mean comparison
  understates real error (Jensen's inequality). `Spread/Error` is deliberately unchanged
  by this, since it's already single-truth-aware by design. Also gained
  `--test-x-path`/`--test-y-path` (train on one dataset's `train_years`, evaluate on a
  *different* dataset's `test_years` -- used for the FOSI&harr;MESA cross-dataset
  generalization tests in `recommended_config.md`) and a wind-channel harmonization step
  (`collapse_wind_vector_channel`) needed because FOSI's `X` carries full vector wind
  while MESA-HR's carries only wind speed.
- **`train_engressnet.py`** -- CLI entry point. Parses training/test years, patches vs.
  single sub-domain, sub-domain bounds, and model/training/architecture hyperparameters,
  then hands off to `run_pipeline()`. Gained `--seed` (reproducibility/variance testing)
  and the `--test-x-path`/`--test-y-path` cross-dataset options above. Run `python
  train_engressnet.py --help` for the full set of options.
- **`submit_engressnet.sh`** -- PBS batch submission wrapper for Casper (`qsub
  submit_engressnet.sh`). Hyperparameters and the main architecture toggles are
  overridable at submit time via environment variables, e.g.:
  ```bash
  qsub -v TRAIN_YEARS="1980-2005",TEST_YEARS="2006-2014",BETA=0.8,K=20,K_EVAL=20 submit_engressnet.sh
  ```
  Setting `BATCH_NAME` nests output under `results/<BATCH_NAME>/<run_tag>` instead of
  the default flat `results/<run_tag>`, for grouping a batch of related runs (e.g. a
  sensitivity sweep) together.
- **Daily-cadence and sweep driver scripts** (all new) -- `submit_engressnet_daily.sh`
  / `submit_engressnet_daily_mesa.sh` are the daily-data-pipeline counterparts to
  `submit_engressnet.sh`. The remaining `submit_*.sh` scripts are the "self-relocating"
  sweep drivers that `cd` to their own directory and `qsub` a sibling template once per
  train/test-window or architecture-toggle combination (run directly as
  `./submit_whatever_sweep.sh [--submit]`, not via `qsub` -- see each script's header
  comment for what it sweeps): `submit_daily_sweep.sh`, `submit_daily_length_sweep.sh`(`_mesa`),
  `submit_daily_combo_sweep.sh`(`_mesa`), `submit_new_features_sweep.sh`(`_avg`, `_mesa`,
  `_avg_mesa`), `submit_season_sweep.sh`(`_mesa`), `submit_enscalenet_sweep.sh`(`_mesa`),
  `submit_enscalenet_landthresh_combo.sh`(`_mesa`), `submit_stochastic_refine_sweep.sh`(`_mesa`),
  `submit_stochastic_refine_landthresh_combo.sh`(`_avg`, `_mesa`, `_avg_mesa`),
  `submit_stochastic_refine_attn_combo.sh`(`_mesa`), `submit_stochastic_refine_seed_variance.sh`(`_mesa`),
  `submit_stochastic_refine_bilinear_2000_2020.sh`(`_avg`, `_mesa`, `_avg_mesa`),
  `submit_stochastic_refine_consregrid_2000_2020.sh`(`_avg`, `_mesa`, `_avg_mesa`),
  `submit_coastal_bias_experiments.sh`(`_mesa`), `submit_cross_fosi_train_mesa_test.sh`,
  `submit_cross_mesa_train_fosi_test.sh`, `submit_calibrate_daily.sh`. Together these
  produced the evidence behind `recommended_config.md`'s recommendation. Several
  reference `../../recommended_config.md` / `../../process_data/...` in header comments
  from their original nested location under `Version5/submit/training/` -- those paths
  are relative to the working project layout, not this flattened repo copy.
  `submit_sensitivity_tests.sh` is a legacy script from the earlier *monthly*-cadence
  sensitivity sweep; its output batches no longer exist under `results/` -- kept for
  reference/history, not expected to be rerun (this mirrors how the working copy's own
  `submit/README.md` describes it).

### `optimization/` -- hyperparameter search

- **`objective_engressnet.py`** -- [ECHO](https://github.com/NCAR/echo-opt) objective
  for distributed-PBS hyperparameter search over `run_pipeline()` (lr, k, batch_size,
  latent_channels), optimizing test-set RMSE of the stochastic UNet mean.
- **`hyperparameters.yml`** -- ECHO config: the search space for the four
  hyperparameters above, plus PBS/Optuna settings (job count, sampler, storage
  backend).
- **`model_config.yml`** -- fixed settings for every trial (everything the search
  doesn't vary), mirroring `functions_engressnet.py`'s `DEFAULT_*` constants.
- **`launch_pbs.sh`** -- PBS batch wrapper that runs `echo-run hyperparameters.yml
  model_config.yml` (`qsub launch_pbs.sh`).
- **`functions_engressnet.py`** -- symlink to `training/functions_engressnet.py`, so
  `objective_engressnet.py` can import it directly. **Currently broken** (points at
  `../functions_engressnet.py`, i.e. the repo root, where the file no longer lives
  after the `training/` reorg) -- repoint it at `../training/functions_engressnet.py`
  before running a search. Still unresolved as of this update -- out of scope for this
  notebook/script sync, flagged again here.
- **`pbs_job_ids.txt`**, **`study_journal.log`**, **`trial_results.csv`**,
  **`trial_results/trial_results_<jobid>.casper-pbs.csv`** -- committed output from a
  completed ECHO study (`engressnet_hpo_coastal_v2`): the PBS job IDs launched, the
  raw Optuna journal-storage log, and the per-trial (lr, k, batch_size,
  latent_channels) results, both merged and per-worker. Kept as a record of that
  search, not regenerated automatically -- a fresh `echo-run` will produce new files
  alongside these rather than overwrite them. No new ECHO study was run since the last
  sync; these files are unchanged.
- **`mesaclip_fosi_validate_sic_spatial.ipynb`** -- a stray, **now-diverged** duplicate
  of `observations/mesaclip_fosi_validate_sic_spatial.ipynb` (its logical home). It was
  a byte-identical accidental double-commit as of the last README update; this sync
  updated the `observations/` copy with the Version5 crash-fix/presentation work below
  but deliberately left this stray copy untouched, so the two files no longer match.
  Still flagged as cleanup to do (delete this copy), not fixed here.

### `evaluation/` -- compare and visualize results

- **`compare_runs.ipynb`** -- side-by-side comparison of finished `results/` runs:
  full `metrics.csv` table plus a combined Taylor diagram.
- **`compare_all_batches.ipynb`** -- loads `metrics.csv` from every
  `results/<batch>/<run>/` folder (e.g. a sensitivity-sweep grouped via
  `submit_engressnet.sh`'s `BATCH_NAME`) and renders heatmaps comparing batches
  across metrics and splits, including a member-wise-vs-ensemble-mean RMSE
  comparison and a % change vs. a chosen baseline batch. The member-metrics cell now
  computes the **full** per-member-averaged metrics table (not just RMSE) for every
  MESACLIP run via the new `member_metrics.py`, replacing (not just appending to) that
  run's "Stochastic UNet Mean" row so every downstream heatmap reflects the
  per-member-averaged fix automatically; cached in `.member_metrics_cache.json`
  (gitignored, regenerated locally) with incremental checkpointing since a full
  cold-cache sweep is compute-heavy (dominated by per-member SSIM).
- **`evaluation_plots.ipynb`** -- notebook-side figure regeneration from a saved
  `eval_data/` dump, without re-running the model: the standard quick-look figures,
  candidate-coastal-point time series, PIOMAS-referenced comparisons, ensemble
  reliability diagnostics (rank histogram, reliability diagram, spread-skill by SIT
  regime), and a batch-mode section that regenerates every figure/table across a
  whole `results/<batch>/` folder in one pass. Its metrics cell and batch-mode
  metrics block now apply the same MESACLIP per-member-averaging fix as
  `compare_all_batches.ipynb` above, recomputed retroactively from each run's already
  -saved `eval_data/fields.npz`.
- **`evaluation_plots_daily.ipynb`** (new) -- the daily-cadence sibling of
  `evaluation_plots.ipynb`, sharing the same section layout and the per-member-averaging
  fix, plus two additional fixes specific to daily-cadence (mostly 2021-only) runs: (1)
  PIOMAS comparisons are now gated to only use samples within `PIOMAS_MAX_GAP_DAYS` (20
  days) of an actual PIOMAS month, since PIOMAS only covers 1978-2020 and every daily
  run tests on 2021 -- previously every PIOMAS-referenced plot/metric silently compared
  against whatever PIOMAS month was nearest, however far away; (2) a
  `rolling_window_samples()`-based centered rolling-mean smoothing (5-10 day window) on
  the domain-mean SIT, candidate-point, and domain-mean-bias timeseries plots, for
  readability at daily cadence. Its batch-mode section (`run_all_sections`) is
  functionally identical to `run_daily_eval_batch.py` below -- keep the two in sync
  manually if one changes.
- **`run_daily_eval_batch.py`** (new) -- headless CLI mirror of
  `evaluation_plots_daily.ipynb`'s batch-mode cell, for regenerating every
  `saved_figs/<batch>/<run>/` figure set via PBS rather than interactively:
  `python run_daily_eval_batch.py --batch-dir results/<batch>`.
- **`member_metrics.py`** (new) -- shared module backing the per-member-averaged-metrics
  fix used by both notebooks above and `run_daily_eval_batch.py`: torch-based
  MAE/RMSE/Bias/Grad MAE/Pattern Corr/SSIM/IIEE/coastal-MAE/RMSE (copied
  formula-for-formula from `functions_engressnet.py`, not re-derived, so the two
  implementations can't drift apart), `is_mesaclip_run()`, and
  `compute_member_avg_metrics()` (each ensemble member scored against truth, then
  averaged). Deliberately does not import `functions_engressnet.py` itself, to avoid
  pulling in `xesmf`/`pop_tools`/`cartopy` at import time in these lightweight
  `eval_data`-only notebooks.
- **`noise_floor_evaluation.ipynb`** -- compares a run's bias against an estimated
  natural-internal-variability "noise floor" (Cachay et al. 2024, §5.3), using the
  inter-member spread of MESACLIP's 9-member CESM1 ensemble as a same-model-class
  proxy for FOSI's single-trajectory bias, both domain-mean and at named coastal
  points (with a local-footprint average to avoid coastal single-cell noise). Carried
  over unchanged from the prior sync -- this notebook does not currently exist in the
  Version5 working copy.
- **`submit_compare_all_batches.sh`** / **`submit_daily_eval_batch.sh`** /
  **`submit_daily_eval_sweep.sh`** (all new) -- PBS wrappers for the two notebooks/script
  above: `submit_compare_all_batches.sh` executes `compare_all_batches.ipynb`
  end-to-end via `qsub`; `submit_daily_eval_batch.sh` runs `run_daily_eval_batch.py`
  for one `BATCH_DIR` (`qsub -v BATCH_DIR=results/<batch> submit_daily_eval_batch.sh`);
  `submit_daily_eval_sweep.sh` loops over a list of batch directories and submits one
  `submit_daily_eval_batch.sh` job per batch (`./submit_daily_eval_sweep.sh [--submit]`).
  `submit_daily_eval_batch.sh`'s working-directory handling was adjusted from the
  working copy's `cd "$PBS_O_WORKDIR/../.."` (needed there because that script lives two
  directories below the project root) to plain `cd "$PBS_O_WORKDIR"`, matching this
  repo's flat layout where `run_daily_eval_batch.py` sits directly alongside it --
  submit these from `evaluation/`.
- **`fullyear_on_season_results.csv`** (new) -- one-off result table (8 rows) backing
  `recommended_config.md`'s "Season vs. full-year training" re-check: each already
  -trained `FOSI_fullyear_avg`/`MESA_fullyear_avg` checkpoint re-evaluated (inference
  only) against a Mar-Jul-only slice of its own test set, for a genuinely matched
  comparison against the season-restricted-training runs. The one-off analysis script
  that produced it was a scratch script in a since-cleaned-up job tmp directory, not a
  tracked, rerunnable part of this pipeline -- this CSV is kept as the record of that
  result, similar to `optimization/trial_results.csv`.

### `observations/` -- model-vs-observations validation

- **`mesaclip_fosi_validate_sic_spatial.ipynb`** -- validates MESACLIP and FOSI sea
  ice concentration spatially against NOAA CDR SIC, over the ML regional domain.
  Updated this sync: fixed a crash (both raw MESACLIP/FOSI opens were cropped to the
  small regional bbox only *after* materializing the full global 2400x3600 grid via
  `xr.open_mfdataset`, pushing memory past 20+GB for even one ensemble member; now
  cropped at open time via `preprocess=` so `combine="by_coords"` never builds a graph
  over the full grid) and a real, previously-latent bug found once the crash was fixed
  (`open_mfdataset`'s default `data_vars="all"` treated the static `tarea` grid-cell
  -area field as time-varying, stacking duplicate copies along a spurious new axis --
  fixed via `data_vars="minimal", compat="override", coords="minimal"`). Also
  restyled for presentation: a consistent colorblind-safe palette across every figure
  (MESACLIP/FOSI/CDR share the same color in every panel), larger fonts, and
  `savefig.dpi=220`.
- **`mesaclip_fosi_validate_sic_timeseries.ipynb`** -- same comparison, integrated to
  a regional sea ice area time series over the shared MESACLIP/FOSI/CDR period. Same
  crash fix, `tarea`-shape fix, per-member incremental compute (loop over ensemble
  members with `gc.collect()` between, rather than one `.compute()` over the full
  9-member array), and presentation restyling as the spatial notebook above; also adds
  PIOMAS to the shared palette (SIV panel only) and replaces three unlabeled red
  reference lines at 1950/2000/2050 with a single subtle line at 2006 (the
  historical→RCP8.5 forcing splice).
- **`sea_ice_volume_evaluation.ipynb`** -- domain-integrated sea ice volume (not
  thickness) vs. PIOMAS, time series and seasonal-cycle comparison. Reuses
  `evaluation/evaluation_plots.ipynb`'s config/load/PIOMAS-regrid cells and saves into
  the same `saved_figs/<batch>/<run>/` folder, prefixed `v1_`/`v2_`. Scoped to the
  model's regional sub-domain, not pan-Arctic volume. Minor sync since the last
  update (kept in step with `evaluation_plots.ipynb`'s shared cells).

---

Large run outputs are not tracked in this repository (see `.gitignore`):
`results/` (per-run model checkpoints, `eval_data/` dumps, figures, `metrics.csv`),
`hpo_echo/` (ECHO hyperparameter-search trial logs/results), `saved_figs/`, `logs/`
(PBS stdout/stderr), `__pycache__/`, `optimization/log.txt` (a multi-MB ECHO run
log, ignored by explicit path since it isn't covered by the `logs/` directory rule),
and `.member_metrics_cache.json`/`.member_rmse_cache.json` (locally regenerated
per-run metrics caches used by `evaluation/compare_all_batches.ipynb`). These live on
NCAR GLADE scratch/work space alongside the code. Likewise, training/intermediate
data (the FOSI/MESA HR X/Y pairs) and xESMF regridding weight caches are not tracked
-- they're large binaries regenerated or reused locally on first use.

## Setup

Assumes a conda environment (`downscaling_env`) with `torch`, `pop_tools`, `xesmf`,
`numpy`, `xarray`, `matplotlib`, and `cartopy` available. Regridding depends on
`pop_tools`/`xesmf`, which in turn expect NCAR-internal grid/data paths (`/glade/...`)
-- this code is not expected to run outside NCAR HPC without adapting those paths (see
the `DEFAULT_*` constants near the top of `training/functions_engressnet.py`).
Hyperparameter search additionally needs `echo-opt` (`pip install echo-opt
--break-system-packages`).

Build training data (from `processing/`):

```bash
python build_X_Y_from_FOSI-HR_daily.py    # or build_X_Y_from_MESA-HR_daily.py
```

or submit as a PBS batch job via `submit_build_X_Y_from_FOSI-HR_daily.sh` /
`submit_build_X_Y_from_MESA-HR_daily.sh`. For MESA-HR training windows that extend
past 2006, also run `build_X_Y_from_MESA-HR_daily_rcp85.py` (or its `submit_*.sh`)
and then `stitch_MESA_HR_daily_hist_rcp85.py` (`submit_stitch_MESA_HR_daily.sh`) to
splice the HIST and RCP8.5 records into one continuous file.

Train directly (from `training/`):

```bash
python train_engressnet.py --train-years 1958-2000 --test-years 2001-2022 --patches
```

or on a single lat/lon sub-domain (also produces a domain-mean SIT time series and the
candidate-coastal-point time series, which are skipped under `--patches`):

```bash
python train_engressnet.py --train-years 1980-2005 --test-years 2006-2014 \
    --no-patches --lat-min 60 --lat-max 75 --lon-min -182 --lon-max -151
```

or submit as a PBS batch job via `submit_engressnet.sh`, `submit_engressnet_daily.sh`
(daily-cadence data), or one of the sweep drivers documented above.

Hyperparameter search via ECHO (from `optimization/`, distributed across PBS jobs):

```bash
echo-run hyperparameters.yml model_config.yml
```

or submit as a PBS batch job via `launch_pbs.sh`.

See `recommended_config.md` for the current best-known architecture/data-variant
configuration (`STOCHASTIC_REFINE=true, NOISE_SIGMA=1.0, DATA_VARIANT=avg`), the full
evidence behind it, and a record of the alternatives that were tried and ruled out.

## Data Availability

Source data (FOSI_BGC HR, JRA55-forced; MESACLIP/CESM-LE HR HIST and RCP8.5) is
accessed from NCAR's GLADE/campaign storage and is not distributed with this
repository. `processing/build_X_Y_from_FOSI-HR_daily.py`,
`processing/build_X_Y_from_MESA-HR_daily.py`, and
`processing/build_X_Y_from_MESA-HR_daily_rcp85.py` document how the perfect-model
training pairs are built from raw CICE history files.

## Notes

- The model predicts entirely in normalized (z-scored) space; de-normalization
  (`* Y_std + Y_mean`) and land-masking of predictions both happen downstream, in
  `run_pipeline`, on the physical-space tensors -- not inside the model's `forward()`.
  Hard-zeroing land inside the model instead is a real (previously hit) regression:
  normalized zero is not physical zero.
- IIEE (Integrated Ice Edge Error, Goessling et al. 2016) is computed ocean-only.
  Including land inflates it spuriously, since de-normalizing truth doesn't round-trip
  land's normalized ~0 back to an exact `0.0`.
- MESACLIP metrics (any `MESA_`-prefixed run) should always be read as **per-member
  -averaged** values, not the raw ensemble-mean-vs-single-realization number -- see
  `training/functions_engressnet.py`'s `per_member` auto-detection and
  `evaluation/member_metrics.py`. `Spread/Error` is the one metric intentionally
  unaffected by this distinction.

## Citation

If you use this code, please cite this repository. A formal citation (paper/DOI) will
be added here once available.
