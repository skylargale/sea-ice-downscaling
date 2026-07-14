# sea_ice_downscaling data pipeline (refactored)

This is a function-based, importable replacement for the original
perfect-model-experiment notebook cell you pasted. It's organized as a
small package (`sea_ice_downscaling/`) so it can be unit tested, imported
from a notebook, run as a script, or eventually wired into NCAR MILES's
CREDIT platform as a preprocessing step.

## Layout

```
sea_ice_downscaling/
    config.py            # PipelineConfig, RegionBBox, region registry, var->component map
    file_discovery.py     # glob + year-filter CESM history files (robust year parsing)
    grids.py               # native source grids (POP, SCRIP), destination grids
                            #   (1deg/0p1deg rectilinear + EASE-Grid 2.0 N25km), regridders
    regrid.py               # per-file open/subset/regrid/region-select workers
    dataset_builder.py       # assemble per-variable file lists into aligned X/Y datasets
    io_utils.py                # chunked Zarr/NetCDF writing + normalization-stats sidecar
    build_dataset.py             # top-level run_pipeline() / CLI entry point
```

## Quick start

```python
from sea_ice_downscaling.config import PipelineConfig, REGIONS
from sea_ice_downscaling.build_dataset import run_pipeline

# Legacy behavior: rectilinear 1deg/0.1deg destinations, Cambridge Bay
config = PipelineConfig()
run_pipeline(config)

# New: regrid the coarsened ice predictor channels onto EASE-Grid 2.0 N25km
config = PipelineConfig(region=REGIONS["cambridge_bay"], dest_grid="ease2_n25km")
run_pipeline(config)
```

or from the command line:

```bash
python -m sea_ice_downscaling.build_dataset --region cambridge_bay --dest-grid ease2_n25km
```

## What was actually broken in the original script, and what changed

1. **Undefined regridders (`NameError` at runtime).** `process_file` /
   `process_file_hi` referenced `regridder_coarse_ice`, `regridder_low_atm`,
   and `regridder_high` -- none of these were ever defined. Only
   `grid["regrid"]["ice_hr_to_lr"]` etc. existed, under different names, in
   a different namespace. Fixed by passing the actual regridder objects
   into worker functions as explicit arguments (`regrid.py`,
   `dataset_builder.py`), keyed consistently in `grids.GridBundle`.

2. **`jmin_high_ice` / `jmin_high` / `atm_mask` undefined**, and even once
   defined, the implied approach (`np.any(TLAT >= 40, axis=1)` converted to
   a single `slice(jmin, None)`) silently assumes the high-latitude rows
   of a tripole POP grid are one contiguous block, which isn't guaranteed.
   Fixed by keeping the boolean mask itself (`grids.IceSourceGrid.lat_mask`,
   `grids.AtmSourceGrid.ncol_mask`) and applying the *same* mask object both
   when building the grid descriptor and when subsetting each opened data
   file, so the regridder's stored source shape always matches what it's
   fed at apply time.

3. **`ProcessPoolExecutor` workers depended on module-level globals**
   (`ice_vars`, `bbox`, `lon_min`, regridders, masks) that may or may not
   exist in a child process depending on the multiprocessing start method.
   Works under fork-on-Linux by accident; not guaranteed elsewhere. Fixed
   by making every worker function (`regrid.process_ice_file`,
   `regrid.process_atm_file`) take a single explicit, picklable job object
   with all required data as fields.

4. **Longitude seam handling.** `lon_min % 360` / `lon_max % 360` followed
   by a plain `.sel(lon=slice(...))` silently breaks for any bbox that
   straddles the 0/360 seam (e.g. a box defined as `lon_min=-10,
   lon_max=10` in -180/180 terms wraps to `(350, 10)`, and a slice from 350
   to 10 selects nothing). Your two actual regions (Cambridge Bay,
   Kivalina) do NOT cross the seam, so this wasn't biting you yet -- but
   it's a landmine for any future Alaska coastal community near the
   dateline. `config.RegionBBox.crosses_seam()` checks this, and
   `PipelineConfig` now refuses to construct with a seam-crossing region
   rather than silently mis-selecting.

5. **Time alignment by positional truncation**
   (`isel(time=slice(0, min_t))`) assumed index 0 in every channel/member
   corresponds to the same calendar date. If any file list starts at a
   different month, this silently misaligns predictor/target pairs with no
   error. Fixed by aligning on actual time-coordinate values
   (`dataset_builder._align_on_time`, using `xr.align(..., join="inner")`),
   which raises if the intersection is empty and warns if it's
   suspiciously small.

6. **Unchunked, uncompressed `to_netcdf` on the full array.** Fixed via
   `io_utils.save_dataset`, which chunks before writing and defaults to
   Zarr (CREDIT's preferred format) with NetCDF still available via
   `fmt="netcdf"`.

## EASE-Grid 2.0 support

`grids.build_ease2_n25km_grid()` builds the NSIDC EASE-Grid 2.0 Northern
Hemisphere 25 km grid (`EASE2_N25km`: Lambert azimuthal equal-area, WGS84,
720x720 cells) as a 2D lat/lon xarray Dataset on `(y, x)` dims -- the same
curvilinear-grid shape xESMF already expects for your POP source grids, so
it plugs into the existing `xe.Regridder` machinery directly. Set
`PipelineConfig(dest_grid="ease2_n25km")` to regrid the coarsened `hi`/`aice`
predictor channels onto it instead of the rectilinear "1deg" grid. Requires
`pyproj` (not otherwise a dependency of this pipeline).

**Worth deciding deliberately, not defaulting silently:** the regridder
for `aice` (sea ice concentration, bounded [0,1]) is built with
`method="bilinear"` everywhere in this refactor, matching your original
script. Bilinear can produce small negative values or values >1 at the ice
edge and doesn't conserve total ice area. If area conservation matters for
your validation metrics, switch the `aice` regridder to
`method="conservative"` in `grids.build_grid_bundle` -- this needs cell
corner coordinates (POP grids expose these as `ULAT`/`ULONG` via
`pop_tools`), which I haven't wired up since your original used bilinear
throughout and I didn't want to silently change your science without you
deciding that's wanted.

## Toward CREDIT

I could not find a documented, stable schema for CREDIT's *input* data
class (variable/dim naming conventions, exact config keys) in what's
publicly available right now -- what is confirmed is that CREDIT consumes
chunked Zarr stores and expects normalization statistics as a separate
sidecar rather than baked into the data
(https://github.com/NCAR/miles-credit, https://miles-credit.readthedocs.io/).
This refactor moves you toward that shape:

- `io_utils.save_dataset` writes chunked Zarr by default.
- `io_utils.compute_and_save_scaling` computes and saves per-channel
  mean/std as a sidecar file, mirroring CREDIT's separately-distributed
  ERA5 scaling file.

Before wiring this directly into a CREDIT training config, check the
current docs for the expected Zarr variable/dimension naming and config
schema -- I'd rather you verify against the live docs than have me guess
at field names that may be wrong or may have changed.

## Model training (EngressNet)

Your `3d_unet_model.ipynb` notebook is functionized the same way as the data
pipeline, into these additional modules:

```
sea_ice_downscaling/
    model.py            # EngressNet (UNet + Engression decoder) architecture + build_model()
    channels.py           # name-based channel units/clipping/subsetting (see below)
    patches.py              # patch extraction, train/test split, normalization
    losses.py                  # energy_loss, land_loss
    training.py                  # TrainConfig, train_one_epoch, train
    evaluation.py                   # ensemble eval, bilinear baseline, metrics
    land_mask.py                      # land-sea mask construction (reuses grids.py/regrid.py)
    train_pipeline.py                   # TrainingPipelineConfig, prepare_training_data,
                                           # run_training_pipeline -- the top-level entry point
```

with `engressnet_training.ipynb` as the notebook front-end (same step-by-step structure as
`engressnet_data_prep.ipynb`).

### Bugs fixed vs. the original notebook

1. **Dead code referencing undefined names.** The "Training loop (normal)" cell referenced
   `masked_loss`, `base_loss`, and `USE_MASKED_LOSS` -- none of which are defined anywhere in
   the notebook. It also doesn't pass `z` to the model at all, unlike every other cell, which
   suggests it's an earlier draft superseded by the "Training loop (Engression)" cell. Only the
   Engression loop (which IS fully defined and consistent with the rest of the notebook) is
   reproduced in `training.py`. If you want the masked-loss variant, `masked_loss`/`base_loss`
   need to be written from scratch -- they don't exist to recover.

2. **`model.module.latent_channels`** assumed `model` is always wrapped in `nn.DataParallel`
   (`.module` is how DataParallel exposes the underlying model). Breaks on a single GPU, on CPU,
   or under `DistributedDataParallel`. Fixed by having callers pass `latent_channels` explicitly
   (`build_model(..., data_parallel=False)` by default now -- only wrap in DataParallel if you're
   actually running multi-GPU).

3. **Hardcoded positional channel indices -- the most important one for your new data.** The
   original did:
   ```python
   X[:, :, 3:5, :, :] /= 100.0          # assumes uvel,vvel are channels 3,4 (cm/s -> m/s)
   X[:, :, 0, :, :] = np.clip(...)       # assumes SIT is channel 0
   X = X[:, :, [0, 3, 4], :, :]           # keep SIT, uvel, vvel by POSITION
   ```
   This only works if your saved `X`'s 5 channels are in *exactly* `[hi, Tsfc, SST, uvel, vvel]`
   order. It silently breaks against this package's `dataset_builder.py` output, whose default
   `low_vars` is `("hi", "aice", "U", "V")` -- 4 channels, no Tsfc/SST, and `U`/`V` are
   **atmosphere** winds (already m/s) rather than POP ocean velocities (cm/s). Running the old
   `/= 100.0` against the new data would silently shrink wind data by 100x with no error.

   `channels.py` replaces all of this with name-based lookups (`apply_channel_processing`,
   `select_channels`, `find_channel_index`) driven by a `ChannelSpec` per variable (units,
   conversion factor, clip bounds). `NEW_PIPELINE_CHANNELS` matches this package's data
   pipeline; `LEGACY_POP_CHANNELS` matches your original 5-channel notebook data, for
   reprocessing old saved files. Get the channel order wrong and it raises a `KeyError`
   immediately instead of silently mis-converting or mis-selecting a channel.

4. **`sit_idx = 0`** (hardcoded literal for the bilinear-baseline channel) replaced by
   `channels.find_channel_index(channel_order, "hi")`.

### Verified vs. not verified

I confirmed the following directly, in this sandbox (no GPU, no `torch` installed, no network):

- Every line of `EngressNet.forward()`'s actual tensor operations is **character-for-character
  identical** to your original notebook's `UNet.forward()` (diffed programmatically) -- the
  only differences are the function signature gaining type hints and the removal of comments
  for a dead, commented-out additive-noise branch you'd already chosen not to use.
- Every encoder/decoder/bottleneck layer's declared in/out channel counts chain together
  correctly end-to-end (traced statically against the source, since I can't run a forward
  pass without `torch`).
- `channels.py`'s name-based processing exactly reproduces the old positional behavior on
  synthetic 5-channel data, AND correctly avoids the unit-conversion bug on synthetic
  4-channel new-pipeline data (confirmed `U`/`V` are NOT divided by 100, since they're
  spec'd as already-m/s).
- The channel-mismatch guards (`KeyError` on an unrecognized or missing channel name) actually
  fire on bad input rather than silently passing through.
- `extract_patches`'s scale-factor warning correctly fires when LR/HR grids have meaningfully
  different y/x scale ratios (the EASE-Grid-vs-rectilinear mismatch scenario) and stays silent
  for a clean matched-ratio case.
- Found and fixed one real bug **introduced while writing the notebook**: the original draft of
  the data-prep preview cell tried to `contourf` a small patch (`X_train[0,0]`, shape ~16x24)
  against the full-domain lat/lon coordinate arrays (shape matching the whole region) -- a shape
  mismatch that would have crashed on first run. Fixed by plotting patches with `imshow` against
  their own pixel grid and reserving `contourf`-against-real-lat/lon for the full-domain land
  mask plot, where the shapes actually match.

**Not verified** (no `torch`, `pop_tools`, `xesmf`, or GPU access in this sandbox): the actual
forward pass numerics, training convergence, `build_land_mask`'s regridding against real POP
grid files, and anything that needs your actual saved `X`/`Y` arrays. Smoke-test on a small
subset on Casper/Derecho (a GPU node, since `EngressNet` is sized for one) before trusting a
full training run.

### Before you run `engressnet_training.ipynb`

- **Set `channel_order` / `channel_specs` to match what's ACTUALLY in your saved X file.** If
  you preprocessed with this package's `build_dataset.py`, the defaults in the notebook's config
  cell are already correct. If you're loading an older 5-channel file from the original
  notebook's data pipeline, switch to the commented-out `LEGACY_POP_CHANNELS` line instead.
- **Set `region` to match what you used during preprocessing** -- the land mask needs to be
  cropped to the same bbox as your X/Y data, or it won't be spatially aligned.
- The energy-score loss (`energy_loss`) divides by `K - 1` where `K` is the ensemble size
  (`train_config.k_ensemble`); this is unchanged from your notebook, but note `K` must be `>= 2`
  or you'll hit a division by zero -- the default of 6 is fine, just don't set it to 1 expecting
  a deterministic-only run (use `z=torch.zeros(...)` for that instead, as the eval code does).


- `file_discovery.parse_file_start_year` assumes a CESM-style trailing
  date stamp (`YYYYMM` or `YYYY-MM`) immediately before `.nc`. It's been
  tested against a few synthetic CESM-like filenames but not against your
  actual GLADE file listing -- run `collect_files` on real paths and
  spot-check a few before trusting it at scale.
- `grids.build_ease2_n25km_grid`'s cell grid uses the official NSIDC
  EASE2_N25km parameters (720x720 cells, 25,025.26 m resolution,
  9,000,000 m half-extent) but is not fetched from NSIDC dynamically --
  if NSIDC revises the grid definition, this needs a manual update.
- `dataset_builder.build_target_dataset` keeps Y on the HR rectilinear
  destination (`config.hr_dest_grid`) by default; it does not
  automatically follow `config.dest_grid`. If you want Y on EASE-Grid 2.0
  too, set `hr_dest_grid="ease2_n25km"` explicitly.
- None of this has been run against the real MESACLIP/GLADE files (no
  network/filesystem access to `/glade/...` from this environment) --
  only the pure-Python logic (year parsing, seam detection, config
  validation, the curvilinear masking math, and the EASE-Grid 2.0
  projection geometry) has been independently tested here. Run a
  small-scale smoke test (one ensemble member, a couple of files) on
  Casper/Derecho before launching the full historical-period build.
