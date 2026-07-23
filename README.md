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
  MESACLIP/CESM-LE HR ensemble (d651007/d651030). Both regrid the native ice history
  onto the 1-degree low-res grid (`interp` and `avg`, saved separately) and the
  0.1-degree high-res target grid.
- **`submit_build_X_Y_from_FOSI-HR_daily.sh`** / **`submit_build_X_Y_from_MESA-HR_daily.sh`**
  -- PBS batch wrappers for the two build scripts above (`qsub
  submit_build_X_Y_from_<...>.sh`). CPU/memory-only regridding jobs (xESMF/pop_tools),
  no GPU needed.

### `training/` -- train the model

- **`functions_engressnet.py`** -- all pipeline logic: data loading, land-sea masking
  (regridded POP `KMT` ocean mask), sliding-window patch extraction (or single
  sub-domain cropping when patches are disabled), the stochastic residual UNet model,
  the energy-score training loss, evaluation metrics (MAE/RMSE, coastal MAE/RMSE,
  IIEE, spread/error), figure generation, and candidate-coastal-point / domain-mean SIT
  time series output. `run_pipeline()` is the single entry point everything else calls.
- **`train_engressnet.py`** -- CLI entry point. Parses training/test years, patches vs.
  single sub-domain, sub-domain bounds, and model/training hyperparameters, then hands
  off to `run_pipeline()`. Run `python train_engressnet.py --help` for the full set of
  options.
- **`submit_engressnet.sh`** -- PBS batch submission wrapper for Casper (`qsub
  submit_engressnet.sh`). All hyperparameters are overridable at submit time via
  environment variables, e.g.:
  ```bash
  qsub -v TRAIN_YEARS="1980-2005",TEST_YEARS="2006-2014",BETA=0.8,K=20,K_EVAL=20 submit_engressnet.sh
  ```
- **`submit_sensitivity_tests.sh`** -- standalone driver (not part of the core
  pipeline) that qsub-submits a battery of sensitivity-test batches through
  `submit_engressnet.sh`, sweeping one axis at a time (epochs, beta, an extra UNet
  layer, stochastic refinement, domain size) across the standard 5-way train/test
  split, relative to a fixed baseline. Dry-run by default; pass `--submit` to actually
  qsub.

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
  before running a search.

### `evaluation/` -- compare and visualize results

- **`compare_runs.ipynb`** -- side-by-side comparison of finished `results/` runs:
  full `metrics.csv` table plus a combined Taylor diagram.
- **`compare_all_batches.ipynb`** -- loads `metrics.csv` from every
  `results/<batch>/<run>/` folder (e.g. the sensitivity-test batches from
  `submit_sensitivity_tests.sh`) and renders heatmaps comparing batches across
  metrics, splits, and vs. a baseline.
- **`evaluation_plots.ipynb`** -- notebook-side figure regeneration from a saved
  `eval_data/` dump, without re-running the model: the standard quick-look figures,
  candidate-coastal-point time series, PIOMAS-referenced comparisons, and a
  batch-mode section that regenerates every figure/table across a whole
  `results/<batch>/` folder in one pass.

### `observations/` -- model-vs-observations validation

- **`mesaclip_fosi_validate_sic_spatial.ipynb`** -- validates MESACLIP and FOSI sea
  ice concentration spatially against NOAA CDR SIC, over the ML regional domain.
- **`mesaclip_fosi_validate_sic_timeseries.ipynb`** -- same comparison, integrated to
  a regional sea ice area time series over the shared MESACLIP/FOSI/CDR period.

---

Large run outputs are not tracked in this repository (see `.gitignore`):
`results/` (per-run model checkpoints, `eval_data/` dumps, figures, `metrics.csv`),
`hpo_echo/` (ECHO hyperparameter-search trial logs/results), `saved_figs/`, `logs/`
(PBS stdout/stderr), and `__pycache__/`. These live on NCAR GLADE scratch/work space
alongside the code. Likewise, training/intermediate data (the FOSI/MESA HR X/Y pairs)
and xESMF regridding weight caches are not tracked -- they're large binaries
regenerated or reused locally on first use.

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
`submit_build_X_Y_from_MESA-HR_daily.sh`.

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

Source data (FOSI_BGC HR, JRA55-forced; MESACLIP/CESM-LE HR) is accessed from NCAR's
GLADE/campaign storage and is not distributed with this repository.
`processing/build_X_Y_from_FOSI-HR_daily.py` and
`processing/build_X_Y_from_MESA-HR_daily.py` document how the perfect-model training
pairs are built from raw CICE history files.

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
