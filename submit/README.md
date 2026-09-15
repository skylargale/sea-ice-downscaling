# Job submission scripts

All `qsub`-able entry points for this project, grouped by what they do. Everything else (core
library code in `training/`, notebooks/scripts in `evaluation/`, `processing/`, `results/`,
`logs/`) lives in its own stage folder at the `Version6/` project root — see `Version6/README.md`.

Trimmed 2026-09-15 to just the scripts this fork's one job (retrain the recommended config under
the increment/residual formulation, then evaluate it) actually needs. Version5's much larger
script history (sensitivity sweeps, ablations, one-off PIOMAS/CryoSat2 inference runs) still
exists unchanged in `Version5/submit/` if you need to reference or rerun any of it.

## Layout

- `training/`
  - `submit_engressnet_daily_mesa.sh`, `submit_engressnet_daily.sh` — the general-purpose,
    per-run PBS templates (MESA and FOSI respectively). Every env var (`BATCH_NAME`,
    `TRAIN_YEARS`, `TEST_YEARS`, `DATA_VARIANT`, `STOCHASTIC_REFINE`, `NOISE_SIGMA`,
    `NOISE_SHARED_BIAS`, etc.) can be overridden per-submission without editing the file, e.g.
    `qsub -v TRAIN_YEARS="2000-2005",TEST_YEARS="2021" submit_engressnet_daily_mesa.sh`.
  - `submit_stochastic_refine_sweep_mesa_sharedbias.sh` — the actual sweep driver for this
    fork's primary comparison: loops the recommended config over the 4 train-window splits
    (test=2021) and `qsub`s `submit_engressnet_daily_mesa.sh` once per split. Run as
    `./submit_stochastic_refine_sweep_mesa_sharedbias.sh` (dry run) or `--submit`.
  - `functions_engressnet.py`, `train_engressnet.py`, `processing`, `results`, `logs` — symlinks
    back to the real copies, so the sweep driver's own `cd "$(dirname "$0")"` still lands
    somewhere with everything it needs.
- `evaluation/`
  - `submit_daily_eval_batch.sh` — regenerates evaluation figures/tables (including CRPS by
    regime, per-member PSD spread) from an already-trained run; no training, no GPU.
  - `submit_4window_aggregate.sh` / `submit_4window_aggregate_figures.sh` — pools the 4
    train-window runs (same test=2021) into one set of aggregate figures via additive
    sufficient statistics.
  - `submit_analog_baseline.sh` — runs the nearest-neighbor analog distributional baseline
    (CPU-only) against a template run's config.
  - `submit_compare_all_batches.sh` — regenerates `compare_all_batches.ipynb`'s cross-batch
    comparison tables/plots.
  - `compare_all_batches.ipynb`, `run_daily_eval_batch.py`, `.member_metrics_cache.json`,
    `results`, `logs` — symlinks back to the real copies.

## Two conventions (inherited unchanged from Version5/Version4)

1. **Plain PBS templates** (`submit_engressnet_daily*.sh`, `submit_daily_eval_batch.sh`,
   `submit_4window_aggregate*.sh`, `submit_analog_baseline.sh`, `submit_compare_all_batches.sh`):
   hardcode an absolute `cd` to the `Version6/` root rather than relying on `$PBS_O_WORKDIR` or
   `BASH_SOURCE` (both were found to resolve incorrectly once PBS spools/copies the script at
   submission time) — safe to `qsub` from anywhere.
2. **Self-relocating "sweep" drivers** (`submit_stochastic_refine_sweep_mesa_sharedbias.sh`): run
   directly as `./submit_....sh [--submit]`, not via `qsub`. `cd`s to its own directory first,
   then `qsub`s a sibling template by bare filename — that's why the symlinks listed above exist
   in `training/`/`evaluation/`. Don't delete them or move the driver out of this folder without
   also moving what it points to.
