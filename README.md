# sea-ice-downscaling

Deep learning stochastic downscaling of Arctic sea ice thickness (SIT) for regional,
coastal sea ice. A coarse CESM/reanalysis-forced "low-res" field is downscaled onto a
high-res target grid with a stochastic UNet trained via an energy-score ("EngressNet")
loss, targeting the Kivalina/Shishmaref/Kotzebue/Nome/Point Hope coastal region of
Alaska. This is research code developed and run on NCAR HPC (Casper).

## Repository Contents

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
- **`objective_engressnet.py`** -- [ECHO](https://github.com/NCAR/echo-opt) objective
  for distributed-PBS hyperparameter search over `run_pipeline()` (lr, k, batch_size,
  latent_channels), optimizing test-set RMSE of the stochastic UNet mean.
- **`compare_runs.ipynb`** -- notebook for loading `metrics.csv` across multiple
  `results/` runs side by side (e.g. comparing a hyperparameter sweep).
- **`evaluation_plots.ipynb`** -- notebook-side figure regeneration from a saved
  `eval_data/` dump, without re-running the model.
- **`comp_plots/`** -- simulated and observational data comparison notebooks:
  `mesaclip_fosi_validate_sic_spatial.ipynb` evaluates both MESACLIP and FOSI against
  NOAA CDR SIC spatially; `mesaclip_fosi_validate_sic_timeseries.ipynb` does the same for the
  shared period time series.
- **`process_data/`** -- data-preparation notebooks: `build_X_Y_from_XXXX-HR_daily.ipynb` builds
  the perfect-model low-res/high-res training pair from raw FOSI_BGC CICE history
  files (JRA55-forced, single realization) or MESACLIP output for daily frequency.

Large run outputs are not tracked in this repository (see `.gitignore`):
`results/` (per-run model checkpoints, `eval_data/` dumps, figures, `metrics.csv`),
`hpo_echo/` (ECHO hyperparameter-search trial logs/results), `saved_figs/`, `logs/`
(PBS stdout/stderr), and `__pycache__/`. These live on NCAR GLADE scratch/work space
alongside the code. Likewise, training/intermediate data (the FOSI HR X/Y pair) and
xESMF regridding weight caches are not tracked -- they're large binaries regenerated
or reused locally on first use.

## Setup

Assumes a conda environment with `torch`, `pop_tools`, `xesmf`, `numpy`, `xarray`,
`matplotlib`, and `cartopy` available. Regridding depends on `pop_tools`/`xesmf`, which
in turn expect NCAR-internal grid/data paths (`/glade/...`) -- this code is not
expected to run outside NCAR HPC without adapting those paths (see the `DEFAULT_*`
constants near the top of `functions_engressnet.py`).

Train directly:

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

Hyperparameter search via ECHO (distributed across PBS jobs):

```bash
echo-run objective_engressnet.py  # see objective_engressnet.py / ECHO docs for the config file it expects
```

## Data Availability

Source data (FOSI_BGC HR, JRA55-forced) is accessed from NCAR's GLADE/campaign storage
and is not distributed with this repository. `process_data/build_X_Y_from_HR.ipynb`
documents how the perfect-model training pair is built from raw CICE history files.

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
