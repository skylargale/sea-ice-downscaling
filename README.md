# sea-ice-downscaling

Deep learning stochastic downscaling of Arctic sea ice thickness (SIT) for regional,
coastal sea ice. A coarse CESM/reanalysis-forced "low-res" field is downscaled onto a
high-res target grid with a stochastic UNet trained via an energy-score ("EngressNet")
loss, targeting the Kivalina/Shishmaref/Kotzebue/Nome coastal region of Alaska.

This is research code developed on NCAR HPC (Casper/Derecho), organized as a sequence of
dated experiment directories rather than a single packaged library. There is no single
"the code" -- check which directory below is relevant before reading further.

## Repository Contents

- **`data_access/`** -- notebooks for pulling raw source data (CORDEX, MESACLIP).
- **`2d_models/`, `3d_models/`** -- early exploratory notebooks (2D single-timestep and 3D
  spatiotemporal CNN/UNet experiments) that predate the `Version2+` refactor.
  `3d_models/build_X_Y_from_HR.ipynb` builds the perfect-model X/Y training pair
  currently used by `Version4` (FOSI_BGC HR, JRA55-forced, single realization).
- **`make_plots/`** -- standalone validation/plotting notebooks (SIC spatial/timeseries
  validation against MESACLIP, CESM tutorial, prescribed-atmosphere checks). Not wired
  into the training pipeline; run independently against saved output.
- **`Version2/`** -- a from-scratch, importable-package refactor of the original one-off
  notebook pipeline (see `Version2/README.md` for the bugs it fixes and why). Kept for
  reference; not the currently-trained-against line.
- **`Version3/`** -- the first single-file EngressNet line (`functions_engressnet.py` +
  `train_engressnet.py` + `hpo_engressnet.py` + a PBS submit script). Kept for
  reference/reproducibility.
- **`Version4/`** -- **the current, actively-trained version.** Adds candidate-coastal-point
  time series, an `eval_data/` dump for re-plotting without re-running the model, and
  `engressnet_evaluation_plots.ipynb` for notebook-side figure regeneration.

Weight caches produced by xESMF (regridding weight matrices, land-sea masks) are not
tracked in this repository -- they're large binaries reused across runs and regenerated
locally on first use (see `.gitignore`). Likewise, training/intermediate data (the FOSI
HR X/Y pair, MESACLIP perfect-model files, etc.) live on NCAR GLADE scratch space, not
in this repo.

## Setup

All code assumes a conda environment with `torch`, `pop_tools`, `xesmf`, `numpy`,
`xarray`, `matplotlib`, `zarr`, and `pyproj` available (see `Version2/requirements.txt`).
Regridding depends on `pop_tools`/`xesmf`, which in turn expect NCAR-internal grid/data
paths (`/glade/...`) -- this code is not expected to run outside NCAR HPC without
adapting those paths.

Train the current pipeline:

```bash
cd Version4
python train_engressnet.py --train-years 1958-2000 --test-years 2001-2022 --patches
```

or submit as a PBS batch job via `Version4/submit_engressnet.sh`. See the docstrings in
`Version4/functions_engressnet.py` and `Version4/train_engressnet.py --help` for the
full set of options (sub-domain vs. patch extraction, hyperparameters, splicing in
SSP/RCP8.5 continuation data for `Version3`, etc.).

Hyperparameter search (Optuna, reuses the same pipeline):

```bash
cd Version4
python hpo_engressnet.py --train-years 1980-2005 --test-years 2006-2014 --patches --n-trials 30
```

`Version2` has an actual test suite (`Version2/functions/test_pipeline_logic.py`, run via
`pytest`) covering pipeline logic that doesn't need GLADE/`pop_tools`/`xesmf`/GPU access.

## Data Availability

Source data (CESM-LE, MESACLIP, FOSI_BGC HR) is accessed from NCAR's GLADE/campaign
storage and is not distributed with this repository. `data_access/` documents how the
raw CORDEX/MESACLIP inputs are pulled; `3d_models/build_X_Y_from_HR.ipynb` documents how
the current FOSI-based perfect-model training pair is built from raw CICE history files.

## Notes

- When working on "the" training pipeline with no version specified, assume `Version4`.
  `Version3` and `Version2` are kept for reference/reproducibility, not active
  development.
- `Version2/README.md` documents several real bugs found and fixed during that refactor
  (undefined regridders, hardcoded positional channel indices, longitude-seam bugs,
  silent time-misalignment, `DataParallel`-only `.module` access) -- worth reading before
  assuming any given behavior in the original notebook pipeline was intentional.

## Citation

If you use this code, please cite this repository. A formal citation (paper/DOI) will be
added here once available.
