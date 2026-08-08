# sea-ice-downscaling

Deep learning stochastic downscaling of Arctic sea ice thickness (SIT) for regional,
coastal sea ice. A coarse CESM/reanalysis-forced "low-res" field is downscaled onto a
high-res target grid with a stochastic UNet trained via an energy-score ("EngressNet")
loss, targeting the Kivalina/Shishmaref/Kotzebue/Nome/Point Hope coastal region of
Alaska. This is research code developed and run on NCAR HPC (Casper).

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
- **`submit_build_X_Y_from_FOSI-HR_daily.sh`** / **`submit_build_X_Y_from_MESA-HR_daily.sh`**
  / **`submit_build_X_Y_from_MESA-HR_daily_rcp85.sh`** / **`submit_stitch_MESA_HR_daily.sh`**
  -- PBS batch wrappers for the four build/stitch scripts above (`qsub
  submit_build_X_Y_from_<...>.sh`). CPU/memory-only regridding jobs (xESMF/pop_tools),
  no GPU needed.

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
  locally-connected stochastic refinement (either as a single added stage or replacing
  the whole decoder), an explicit coastal-fraction input channel, an auxiliary
  land/ocean/ice classification head, and windowed self-attention at the end of the
  decoder -- plus a `--calibrate-from`/`--freeze-backbone` path for loading a trained
  checkpoint and recalibrating only its noise-injection pathway.
- **`train_engressnet.py`** -- CLI entry point. Parses training/test years, patches vs.
  single sub-domain, sub-domain bounds, and model/training/architecture hyperparameters,
  then hands off to `run_pipeline()`. Run `python train_engressnet.py --help` for the
  full set of options.
- **`submit_engressnet.sh`** -- PBS batch submission wrapper for Casper (`qsub
  submit_engressnet.sh`). Hyperparameters and the main architecture toggles are
  overridable at submit time via environment variables, e.g.:
  ```bash
  qsub -v TRAIN_YEARS="1980-2005",TEST_YEARS="2006-2014",BETA=0.8,K=20,K_EVAL=20 submit_engressnet.sh
  ```
  Setting `BATCH_NAME` nests output under `results/<BATCH_NAME>/<run_tag>` instead of
  the default flat `results/<run_tag>`, for grouping a batch of related runs (e.g. a
  sensitivity sweep) together.

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
  before running a search. (This has regressed once already: it was fixed this way in
  an earlier commit, then reverted back to the broken target in the latest push.)
- **`pbs_job_ids.txt`**, **`study_journal.log`**, **`trial_results.csv`**,
  **`trial_results/trial_results_<jobid>.casper-pbs.csv`** -- committed output from a
  completed ECHO study (`engressnet_hpo_coastal_v2`): the PBS job IDs launched, the
  raw Optuna journal-storage log, and the per-trial (lr, k, batch_size,
  latent_channels) results, both merged and per-worker. Kept as a record of that
  search, not regenerated automatically -- a fresh `echo-run` will produce new files
  alongside these rather than overwrite them.
- **`mesaclip_fosi_validate_sic_spatial.ipynb`** -- byte-identical duplicate of
  `observations/mesaclip_fosi_validate_sic_spatial.ipynb` (see that entry below); its
  logical home is `observations/`, this copy looks like an accidental double-commit.

### `evaluation/` -- compare and visualize results

- **`compare_runs.ipynb`** -- side-by-side comparison of finished `results/` runs:
  full `metrics.csv` table plus a combined Taylor diagram.
- **`compare_all_batches.ipynb`** -- loads `metrics.csv` from every
  `results/<batch>/<run>/` folder (e.g. a sensitivity-sweep grouped via
  `submit_engressnet.sh`'s `BATCH_NAME`) and renders heatmaps comparing batches
  across metrics and splits, including a member-wise-vs-ensemble-mean RMSE
  comparison and a % change vs. a chosen baseline batch.
- **`evaluation_plots.ipynb`** -- notebook-side figure regeneration from a saved
  `eval_data/` dump, without re-running the model: the standard quick-look figures,
  candidate-coastal-point time series, PIOMAS-referenced comparisons, ensemble
  reliability diagnostics (rank histogram, reliability diagram, spread-skill by SIT
  regime), and a batch-mode section that regenerates every figure/table across a
  whole `results/<batch>/` folder in one pass.
- **`noise_floor_evaluation.ipynb`** -- compares a run's bias against an estimated
  natural-internal-variability "noise floor" (Cachay et al. 2024, §5.3), using the
  inter-member spread of MESACLIP's 9-member CESM1 ensemble as a same-model-class
  proxy for FOSI's single-trajectory bias, both domain-mean and at named coastal
  points (with a local-footprint average to avoid coastal single-cell noise).

### `observations/` -- model-vs-observations validation

- **`mesaclip_fosi_validate_sic_spatial.ipynb`** -- validates MESACLIP and FOSI sea
  ice concentration spatially against NOAA CDR SIC, over the ML regional domain.
- **`mesaclip_fosi_validate_sic_timeseries.ipynb`** -- same comparison, integrated to
  a regional sea ice area time series over the shared MESACLIP/FOSI/CDR period.
- **`sea_ice_volume_evaluation.ipynb`** -- domain-integrated sea ice volume (not
  thickness) vs. PIOMAS, time series and seasonal-cycle comparison. Reuses
  `evaluation_plots.ipynb`'s config/load/PIOMAS-regrid cells and saves into the same
  `saved_figs/<batch>/<run>/` folder, prefixed `v1_`/`v2_`. Scoped to the model's
  regional sub-domain, not pan-Arctic volume.

---

Large run outputs are not tracked in this repository (see `.gitignore`):
`results/` (per-run model checkpoints, `eval_data/` dumps, figures, `metrics.csv`),
`hpo_echo/` (ECHO hyperparameter-search trial logs/results), `saved_figs/`, `logs/`
(PBS stdout/stderr), `__pycache__/`, and `optimization/log.txt` (a multi-MB ECHO run
log, ignored by explicit path since it isn't covered by the `logs/` directory rule).
These live on NCAR GLADE scratch/work space alongside the code. Likewise,
training/intermediate data (the FOSI/MESA HR X/Y pairs) and xESMF regridding weight
caches are not tracked -- they're large binaries regenerated or reused locally on
first use.

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

or submit as a PBS batch job via `submit_engressnet.sh`.

Hyperparameter search via ECHO (from `optimization/`, distributed across PBS jobs):

```bash
echo-run hyperparameters.yml model_config.yml
```

or submit as a PBS batch job via `launch_pbs.sh`.

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

## Citation

If you use this code, please cite this repository. A formal citation (paper/DOI) will
be added here once available.
