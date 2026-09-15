# SeaIceDownscaling (Version 6)

## Version Update

Version6 is a deliberately narrow fork of `Version5/`'s code: same recommended-config toggles
(`STOCHASTIC_REFINE=true`, `NOISE_SIGMA=1.0`, `NOISE_SHARED_BIAS=true`, `DATA_VARIANT=avg`), but
`functions_engressnet.py` unconditionally trains against a standardized *physical increment*
(`Y_phys - baseline_phys`, normalized by the increment's own train-set mean/std, no internal
bilinear-base residual add inside the model) instead of the standardized raw value — matching
Brajard et al. 2026 (EGUsphere, Arctic SIT super-resolution via conditional diffusion)'s Eq. 4 /
NVIDIA CorrDiff's residual construction. There is no toggle for this; it's simply how the model
trains now. Compare its output directly against `Version5/results/MESA_stochastic_refine_sweep_avg_sharedbias/`
for a clean before/after of the residual/normalization change alone.

Only the code and scripts needed for that one comparison were kept — Version6 was cloned from
Version5's full script history (data-prep variants, a finished HPO study, CryoSat2/PIOMAS
observing-system-design research, validation notebooks), and everything not load-bearing for
this specific retrain-and-evaluate task was deleted (2026-09-15) to keep the fork legible. If you
need any of that other material, it still exists unchanged in `Version5/`.

## Layout

- `training/` — core library/CLI: `functions_engressnet.py` (all pipeline logic, including the
  increment/residual formulation above), `train_engressnet.py` (CLI entry point).
- `processing/` — the two production data-build pipelines this project trains against:
  `build_X_Y_from_FOSI-HR_daily.{py,ipynb}` and `build_X_Y_from_MESA-HR_daily.{py,ipynb}`, each
  with its own `submit_build_*.sh`.
- `evaluation/` — evaluation notebooks/scripts: `run_daily_eval_batch.py` (sections 00-16,
  including CRPS by SIT regime and per-member PSD spread), `member_metrics.py`,
  `build_4window_aggregate_figures.py` (pools the 4 train-window runs into one set of
  rank-histogram/reliability/spread-skill/CRPS/PSD figures), `build_analog_baseline.py`
  (nearest-neighbor distributional baseline), `compare_all_batches.ipynb`.
- `results/` — every training run's output (`metrics.csv`, `eval_data/`, checkpoints), one
  subfolder per batch, one run-folder per split inside each. Empty until a job is submitted.
- `saved_figs/` — evaluation figures/tables generated from `results/`, mirroring its
  `<batch>/<run>/` structure.
- `logs/` — PBS stdout/stderr for every submitted job.
- `submit/` — job-submission scripts, grouped by purpose. See `submit/README.md`.
- `recommended_config.md` — currently an unmodified copy of `Version5/recommended_config.md`
  (the findings doc that established the recommended config this fork trains with); not yet
  updated with Version6-specific results, since no training run has been submitted here yet.
