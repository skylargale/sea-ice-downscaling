# SeaIceDownscaling (Version 6)

## Purpose

Version6 started as a deliberately narrow fork of `Version5/`'s code: same recommended-config
toggles as Version5 initially (`STOCHASTIC_REFINE=true`, `NOISE_SIGMA=1.0`, `NOISE_SHARED_BIAS=true`,
`DATA_VARIANT=avg`), but `functions_engressnet.py` unconditionally trains against a standardized
*physical increment* (`Y_phys - baseline_phys`, normalized by the increment's own train-set
mean/std, no internal bilinear-base residual add inside the model) instead of the standardized raw
value — matching Brajard et al. 2026 (EGUsphere, Arctic SIT super-resolution via conditional
diffusion)'s Eq. 4 / NVIDIA CorrDiff's residual construction. There is no toggle for this; it's
simply how the model trains now. This part is unchanged and still true.

Only the code and scripts needed for that one comparison were kept — Version6 was cloned from
Version5's full script history (data-prep variants, a finished HPO study, CryoSat2/PIOMAS
observing-system-design research, validation notebooks), and everything not load-bearing for
this specific retrain-and-evaluate task was deleted (2026-09-15) to keep the fork legible. If you
need any of that other material, it still exists unchanged in `Version5/`.

**`LocallyConnected2d`/`stochastic_refine`/`enscale_net` permanently removed (2026-09-16).** The
initial `STOCHASTIC_REFINE=true` retrain (`results/MESA_stochastic_refine_sweep_avg_sharedbias/`)
showed the same patchy, unrealistic per-pixel texture in individual ensemble members that
`Version5/recommended_config.md` had already root-caused to `LocallyConnected2d`'s per-location
noise-mixing weights. Rather than keep sweeping around that known-bad mechanism, it — along with
`enscale_net`, `noise_sigma`, `noise_shared_bias`, `noise_mix_kernel`, `noise_bias_smooth_weight`,
and the `--freeze-backbone`/`set_noise_only_trainable` calibration path built around it — was
deleted outright. `Version5/recommended_config.md`'s `STOCHASTIC_REFINE=true` recommendation no
longer applies to this fork and should not be resurrected here.

**The global latent `z` pathway that replaced it was itself replaced again (2026-09-16), same
day.** Two comparison batches (`results/MESA_no_stochastic_refine_avg/`, z at 3 of 4 decoder
stages; `results/MESA_z_inject_final_avg/`, z at all 4) confirmed a single shared/resized `z`
was patchiness-free but over-dispersed (Spread/Error ~1.87 vs. the removed refiner's ~0.97).
Root cause: `z` was drawn once at the bottleneck and merely bilinearly *resized* into every
stage — resizing a coarse field can't add genuinely new high-frequency content, and reusing the
identical field at every stage let its effects reinforce rather than average down, plausibly
inflating the spread. **Current architecture**: each of the 4 decoder stages
(`d3`/`d2`/`d1`/`final`) now has its own independent `GaussianNoiseStage` — a fresh noise draw at
that stage's *own* resolution, correlated by a shared (translation-invariant, not per-location)
5x5 Gaussian kernel with a learnable width, renormalized to undo the blur's own variance
shrinkage, then scaled by a learnable per-stage amplitude. No `z`, no `z_proj_*`, no
`--noise-smooth-kernel-size`, no `--latent-channels` (nothing left to size). Trained end-to-end
via the plain (coastal-weighted) energy loss — no post-hoc calibration step (won't be available
against real observations with no matching high-res truth) and no separate calibration loss term
(deliberate choice). The high-res land mask and the former `--deep-mask-head` toggle's extra
depth are now permanently folded into the final stage's noise/feature fusion, *before* that
fusion runs (and before `attention_end`, if ever enabled) — not concatenated afterward as a
separate late step the way every earlier version of this model did. See
`training/functions_engressnet.py`'s `GaussianNoiseStage`/`UNet` docstrings for the full
mechanism and reasoning. **Not yet retrained or evaluated under this exact architecture** — the
two comparison batches above still reflect the single-global-`z` design, not this one.

## Layout

**Reorganized 2026-09-18 (two passes, same day)**: the old `submit/` folder (a separate tree of
`qsub`-able scripts with symlinks back to the real code) was retired — every submit script now
lives directly alongside the code/data it drives, and `results/`-style sensitivity-sweep output
was split out into its own `sensitivity/` stage. PBS log files are kept in a dotted (hidden)
`.logs/` folder so they don't clutter a plain `ls` of any stage. In a second pass the same day,
`saved_figs/` moved under `evaluation/` (it's evaluation output, not its own top-level stage), and
`paper_appendix/` was retired entirely — its figure-generating code moved into a new `figures/`
stage, and everything else in it moved to whichever stage actually produces/uses it
(`evaluation/` or `training/`). No content was deleted in either pass, only relocated.

- `training/` — core library/CLI *and* its job-submission scripts, together:
  `functions_engressnet.py` (all pipeline logic, including the increment/residual formulation
  above), `train_engressnet.py` (CLI entry point), `submit_engressnet_daily.sh`/
  `submit_engressnet_daily_mesa.sh` (the general-purpose, per-run PBS templates for FOSI/MESA
  respectively — override any env var per-submission, e.g.
  `qsub -v TRAIN_YEARS="2000-2005",TEST_YEARS="2021" submit_engressnet_daily_mesa.sh`, run from
  inside `training/`), `submit_noise_cascade_sweep_{mesa,fosi}.sh` and
  `submit_pretrain_mesa_then_fosi.sh` (self-relocating "sweep driver" scripts — run directly as
  `./submit_....sh [--submit]`, not via `qsub`; they `cd` to their own directory, then `qsub` a
  sibling template above by bare filename), `methods_reconstruction_addition.tex` (the paper's
  reconstruction-equation snippet, describing `functions_engressnet.py`'s de-normalize +
  bilinear-baseline-add step). `training/.logs` and `training/results` are symlinks back to the
  real `.logs/`/`results/` one level up, needed because the templates above `cd` to wherever
  they were `qsub`'d from and reference `results/`/`.logs/` by bare relative name.
- `processing/` — the two production data-build pipelines this project trains against:
  `build_X_Y_from_FOSI-HR_daily.{py,ipynb}` and `build_X_Y_from_MESA-HR_daily.{py,ipynb}`, each
  with its own `submit_build_*.sh` and its own separate `.logs/` (distinct from the main
  `Version6/.logs/` used by `training/`/`evaluation/`).
- `evaluation/` — evaluation notebooks/scripts, their job-submission scripts, and their output:
  `run_daily_eval_batch.py` (sections 00-16, including CRPS by SIT regime and per-member PSD
  spread) with `submit_daily_eval_batch.sh`, `member_metrics.py`,
  `build_4window_aggregate_figures.py` (pools the 4 train-window runs into one set of
  rank-histogram/reliability/spread-skill/CRPS/PSD figures) with
  `submit_4window_aggregate_figures.sh`, `build_analog_baseline.py` (nearest-neighbor
  distributional baseline) with `submit_analog_baseline.sh`, `compare_all_batches.ipynb` with
  `submit_compare_all_batches.sh`, `recompute_iiee.py` (see
  [[project-seaicedownscaling-v6-iiee-bugfix]]) with its `watch_and_fix_iiee.sh` watcher/driver
  and `iiee_recompute_log.txt` journal, `time_inference.py` (one-off GPU inference-cost timing for
  the paper's computational-cost appendix, `computational_cost_appendix.tex`) with
  `submit_time_inference.sh`. **`saved_figs/`** — evaluation figures/tables generated from
  `results/`, mirroring its `<batch>/<run>/` structure, lives here too (moved from the `Version6/`
  root 2026-09-18: it's this stage's output, not its own stage). The plain-template `submit_*.sh`
  scripts here hardcode an absolute `cd` to `Version6/` (not `$PBS_O_WORKDIR`/`BASH_SOURCE`, both
  of which were found to resolve incorrectly once PBS spools/copies the script at submission
  time) — safe to `qsub` from anywhere; `evaluation/.logs` is still a symlink back to the real
  `.logs/` since PBS's own `-o .logs/` output-file directive is resolved relative to wherever
  `qsub` was invoked from, not the script's `cd` target. Anything that reads/writes `saved_figs`
  by a bare relative path (e.g. `run_daily_eval_batch.py --save-root`, default now
  `"evaluation/saved_figs"`) is still relative to the `Version6/` root, not to `evaluation/`
  itself, since every submit script's `cd` target didn't change.
- `figures/` — everything whose main job is producing a plot, pulled out of the old
  `paper_appendix/` (retired 2026-09-18) plus the per-run figure notebook (moved from
  `evaluation/`, since it's a figure tool, not an eval-pipeline one):
  `all_figures_by_run.ipynb` (the "generate every figure for one run" notebook — set `RUN`/
  `WINDOW` and run top to bottom), `build_fig7_piomas_application.py` /
  `fig7_piomas_application_2020.png` (the observational-application figure),
  `build_piomas_fosi_mesa_seasonal_cycle.py` / its `.csv`/`.png` output (the seasonal-cycle
  consistency figure), and `paper_figures_export/` (the already-exported, submission-ready final
  figure set). All three scripts use hardcoded absolute paths, so they don't care what directory
  they're run from.
- `sensitivity/` — everything for the per-stage noise-cascade sensitivity/ablation battery (seed
  variance, mask-fusion timing, noise-channel count, kernel size 3/9), kept separate from the
  flagship default-config runs in `results/`/`evaluation/saved_figs/`:
  - `submit/` — the 5 sweep-driver scripts (`submit_noise_cascade_{k3,k9,latemask,nc2}_sweep_mesa.sh`,
    `submit_noise_cascade_seedvar_mesa.sh`); run directly (`./submit_....sh --submit`), same
    self-relocating convention as `training/`'s drivers, but `qsub`ing
    `../../training/submit_engressnet_daily_mesa.sh` explicitly since the plain template they
    drive lives in a different stage folder now.
  - `results/`, `saved_figs/` — the 5 sensitivity batches' training output and generated figures.
  - `analysis/` — `make_sensitivity_figure.py`/`make_sensitivity_heatmap.py` (regenerate
    `sensitivity_tests.{png,pdf}`/`sensitivity_heatmap*.{png,pdf}` from hand-verified numbers, not
    read live from `results/`) plus their `.tex`/`.csv` outputs, used by the paper's appendix.
- `results/` — the flagship/default-config and retired-comparison runs' output (`metrics.csv`,
  `eval_data/`, checkpoints), one subfolder per batch, one run-folder per split inside each.
- `.logs/` — PBS stdout/stderr for every `training/`/`evaluation/` submitted job (hidden so it
  doesn't show up in a plain `ls`; still just a normal directory — `ls -a .logs/` or `cat
  .logs/<jobid>.casper-pbs.OU` work as usual).
- `recommended_config.md` — currently an unmodified copy of `Version5/recommended_config.md`
  (the findings doc that established the recommended config this fork trains with); not yet
  updated with Version6-specific results, since no training run has been submitted here yet.
