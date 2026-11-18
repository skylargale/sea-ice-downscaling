# SeaIceDownscaling

Stochastic downscaling of Arctic sea ice thickness (SIT) from a 1° CESM-like input to a 0.1° regional grid (lat 60-75,
lon -182 to -151, 150 x 310 cells) with **EngressNet**: a U-Net whose four decoder stages each add their own spatially
correlated Gaussian noise, trained with an energy score so that each forward pass is one realistic ensemble member. The network
predicts the standardized increment over the bilinear upsampling of the coarse SIT.

- Training data: two perfect-model pairs (coarse X and 0.1° truth Y from the same simulation): **FOSI** (CESM high-res,
  JRA55-forced, 1 member) and **MESACLIP** (CESM high-res, 6 members). Daily, test year 2021. The final model (FINAL2) is one
  model per dataset trained on 1,825 days drawn at random from 2000-2020; the submitted paper's model (FINAL1) used four
  consecutive 5-year windows (2000-05, 05-10, 10-15, 15-20).
- Observational application: PIOMAS as the coarse input (2010-2023), scored against ICESat-2 pass-cells (2018-2022).
- **Final model and everything that was tried:** `recommended_config.md`. Superseded configurations and older write-ups:
  `/glade/derecho/scratch/skygale/downscaling/archive/records/config_history.md`.

## Layout

| Folder | What |
|---|---|
| `training/` | `functions_engressnet.py` (all pipeline logic: data, model, loss, training, evaluation, metrics), `train_engressnet.py` (CLI), the PBS templates `submit_engressnet_daily.sh` (FOSI) and `submit_engressnet_daily_mesa.sh` (MESACLIP), `submit_final2_training.sh` (every FINAL2 training run). |
| `processing/` | Builders for the training pairs (`build_X_Y_from_{FOSI,MESA}-HR_daily.py`), the PIOMAS application inputs (3-step chain: `build_X_Y_PIOMAS_obs_2021_daily.py` -> `..._v2_landfixed.py` -> `build_X_PIOMAS_obs_2021_daily_v3_avg.py`; the "2021" names are historical, `FILE_TAG`/`CDR_*` env vars select years), `convert_PIOMAS_hiday_raw.py`, `build_input_swap.py` (other products as the coarse input). Each has a `submit_*.sh`. |
| `evaluation/` | Core evaluation of saved runs: `run_daily_eval_batch.py` (per-run figures/tables), `build_4window_aggregate_figures.py` (pools the 4 windows), `recompute_{ocean_metrics,spread_skill}.py` (ocean-only corrected metrics -> `corrected_calibration/`), `regime_table.py` and `rank_hist_breakdown.py` (per-regime tables), `apply_coarse_zero_fields.py` and `restore_raw_truth.py` (the 2026-10-24 output rule and truth fix), `summarize_ensmean_sweep.py`, `member_metrics.py`. `submit_cz_metrics.sh` reruns the whole final metrics chain. |
| `evaluation/diagnostics/` | One-off analyses whose results are quoted in the paper or in `recommended_config.md`: open-water/far-field breakdowns, open-water spread-skill split, analog baseline, inference timing. |
| `evaluation/revisions/` | 2026-11 co-author revision analyses and the FINAL2 evaluation chain (`submit_final2_{eval_chain,metrics}.sh`). `final_set.py` picks the model: `FINAL_SET=final2` (default since 2026-11-15) or `final1` (submitted paper); outputs written with final2 carry a `_final2` suffix. Numbers in `results/revisions_2026-11/summary.md`. |
| `evaluation/corrected_calibration/` | Metric records for the final batches. Records for deleted/rejected batches are in scratch `downscaling/archive/records/corrected_calibration/old_experiments/`. |
| `evaluation/saved_figs/` | Figures/tables generated from `results/`, mirroring its `<batch>/<run>/` layout. |
| `observations/` | Observed-SIT intercomparison and ICESat-2 scoring (own `README.md`). Notebooks are built from `notebook_src/*_src.py` (the source of truth) with `py2nb.py`. `inference/` holds the PIOMAS-input inference drivers. |
| `figures/` | Paper figures: `figures_v2/` (revised paper figures for FINAL2: all their code is `figures_v2/make_v2_figures.py`, `# %%` cells, with the generated notebook `make_figures.ipynb` and `submit_make_v2_figures.sh` to run it on a compute node; `figures_v2/superseded_final1/` = the same figures for FINAL1), `figures_v1/` (the submitted set), `all_figures_by_run.ipynb` (set `RUN`; run it on a PBS node, not the login node), `build_*.py` for the other figures, `final_chain/` (the two job chains that regenerate the FINAL1 figures, plus `final_results_summary.py`). |
| `sensitivity/` | Appendix ablations (seed variance, late mask fusion, 2 noise channels, kernel 3/9): `analysis/` (figure/heatmap scripts + the CSVs and `.tex` they produce). The raw runs were deleted; the CSVs are the record; the sweep drivers are in scratch `downscaling/archive/records/sensitivity/submit/`. |
| `results/` | FINAL2: `FINAL2_{MESA,FOSI}_{random1825,detmse_random1825,detmse_seedens}`, `PIOMAS_2018-2022_FINAL2_FOSI{,_detmse}`, `PIOMAS_2010-2023_cdrv6_FINAL2_FOSI`. FINAL1 (`FINAL_SET=final1`): `{FOSI,MESA}_{final,detmse}_ema`, `PIOMAS_2018-2022_FOSI_{final,detmse}_ema`, `PIOMAS_2010-2023_cdrv6_emaFOSI_4win`. Revision sweeps: `REV_{E1,E2,F,G,H}_*`. Revision numbers and tables: `revisions_2026-11/`. |
| `weighted_grids/` | Cached xESMF regridding weights (a cache; names can be built at run time, e.g. the land-mask file). |
| `logs/` | PBS output. Every stage's `.logs` is a symlink into it (`processing/`, `observations/`, `sensitivity/` use subfolders). |
| `archive/` | Text-only copies of what was moved to scratch `downscaling/archive/records/` (insurance against scratch purges): `records_text_2026-11-13.tar.gz` (pre-cleanup code and records), `records_text_2026-11-15.tar.gz` (pre-pruning training code, retired revision drivers). |

## Running

Environment: `module load conda && conda activate seaice-downscaling` (Jupyter kernel of the same name). Account `P93300065`, Casper.

```bash
# train the final model (FINAL2): EngressNet, deterministic U-Net and seed ensemble; prints the qsub commands, --submit sends them
training/submit_final2_training.sh FOSI --submit

# evaluate (qsub from evaluation/ so the relative .logs/ resolves)
cd evaluation
qsub -v DS=FOSI revisions/submit_final2_eval_chain.sh     # truth fix, per-run tables, corrected metrics
qsub -v BATCH_DIR=results/FOSI_final_ema submit_daily_eval_batch.sh      # any single batch
qsub submit_cz_metrics.sh          # FINAL1: corrected metrics + regime / rank tables for the 4 final batches
```

Inference on new inputs is a training run with `NUM_EPOCHS=0`, `INIT_CHECKPOINT=...`, `TEST_X_PATH`/`TEST_Y_PATH`; see
`observations/inference/submit_ema_obs_inference.py` and `submit_piomas_2010-2023_inference.py`.

Data live on scratch (purged when left unused; don't keep the only copy of anything there).
`/glade/derecho/scratch/skygale/downscaling` (details in its own `README.md`):

| Path | What |
|---|---|
| `training_data/` | Perfect-model pairs: `{X,Y}_FOSI_HR_JRA55_daily*.nc`, `{X,Y}_MESA_HR_daily*.nc`, monthly FOSI (library defaults), the 2021 U10-conservative revision input; `mesa_hist_1920-2005/` (MESACLIP build intermediates), `conservative_targets/`. |
| `application_inputs/` | `piomas_2010-2023/` (per-year PIOMAS inputs, CDR v6), `obs_inputs_2017-2022/` (PIOMAS / CS2SMOS / TOPAZ4-ML / Petty GPSat inputs for the ICESat-2 scoring, FOSI 25 km aux, static fields). |
| `ancillary/` | `maps/` (etopo2), `observed_products/` (Landy, CDR, ERA5, NSIDC regions). |
| `outputs/` | `pooled_obs_product/`, `revisions_2026-11/`, `backups/`. |
| `archive/` | `records/` (moved out of this project: config history, old metric records, code snapshot, retired drivers, old test figures), `retired/` (~1.3 TB nobody reads, with a move manifest), `versions/` (Version1-5). |

`/glade/derecho/scratch/skygale/observations` holds raw products and ICESat-2 pass-cells. The git mirror is `/glade/work/skygale/repos/sea-ice-downscaling`.
