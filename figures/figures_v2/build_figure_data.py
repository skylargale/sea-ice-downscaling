#!/usr/bin/env python3
"""
build_figure_data.py: the summary data behind the MESACLIP line/curve figures of make_v2_figures (domain-mean and community
time series, Taylor, power spectra, rank histograms, spread-skill and CRPS by regime), one CSV per figure input, written to
figures/figures_v2/data_<FINAL_SET>/. Run it once per model (and again after a model's evaluation changes); the notebook only
reads these files.

    python build_figure_data.py                 # FINAL_SET=final2 (the revised final model, the paper)
    FINAL_SET=final1 python build_figure_data.py

Sources (MESACLIP, test year 2021; batches from evaluation/revisions/final_set.py):
  EngressNet: the run's summary files in evaluation/saved_figs/<EngressNet batch>/_4window_aggregate/, computed from the model
    fields by evaluation/build_4window_aggregate_figures.py. For final2 they summarize the single FINAL2 run, pooling only over
    the 6 MESACLIP members ("4window" in those names is left over from the 4-window final1 model, which it pools over).
  Deterministic U-Net: the det-MSE U-Net's own per-run files (evaluation/saved_figs/<det batch>/<run>/), averaged per date over
    the 6 MESACLIP members (final1: and over its 4 windows), the same pooling as the EngressNet summaries. They replace the
    summaries' "deterministic" column, which is EngressNet's zero-noise pass, not the det-MSE U-Net. The truth is checked to
    match between the two.
Outputs: domain_mean_sit_timeseries.csv, domain_mean_pattern_corr_timeseries.csv, community_point_timeseries.csv,
taylor_stats.csv, psd.csv, psd_member_spread.csv, rank_histogram.csv, rank_histogram_by_regime.csv, spread_skill_by_regime.csv,
crps_by_regime.csv, mesaclip_seasonal_cycle_members.csv (Fig. 8 band), seasonal_cycle_2021.csv (both datasets), engressnet_members_seasonal_cycle_2021.csv, and sources.json (what each file came from).
"""
import glob
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd

BASE = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, f"{BASE}/evaluation/revisions")
from final_set import SETS  # noqa: E402

FINAL_SET = os.environ.get("FINAL_SET", "final2")
FS = SETS[FINAL_SET]
ENG, DET = FS["MESA"]["eng"], FS["MESA"]["det"]
OUT = os.path.join(BASE, "figures", "figures_v2", f"data_{FINAL_SET}")
ENG_SUMMARY = os.path.join(BASE, "evaluation", "saved_figs", ENG, "_4window_aggregate")
DET_RUNS = sorted(d for d in glob.glob(os.path.join(BASE, "evaluation", "saved_figs", DET, "*"))
                  if os.path.isdir(d) and not d.endswith("_4window_aggregate"))


def eng(name, **kw):
    """EngressNet summary file <name>_4window.csv."""
    return pd.read_csv(os.path.join(ENG_SUMMARY, f"{name}_4window.csv"), index_col=0, **kw)


def det(name, **kw):
    """The det-MSE U-Net's per-run file <name>.csv, every run stacked."""
    return pd.concat([pd.read_csv(os.path.join(d, f"{name}.csv"), index_col=0, **kw) for d in DET_RUNS], ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"FINAL_SET {FINAL_SET}\n  EngressNet: {ENG_SUMMARY}\n  det U-Net : {[os.path.basename(d) for d in DET_RUNS]}\n  -> {OUT}")
    written = {}

    def save(df, name, how):
        df.to_csv(os.path.join(OUT, f"{name}.csv"))
        written[f"{name}.csv"] = how
        print(f"  {name}.csv  ({how})")

    # EngressNet-only quantities: unchanged
    # (rank histogram / spread-skill / CRPS by SIT type: evaluation/revisions/sit_type_metrics.py, five types, 2026-11-19)
    for src, name in (("13b_psd_per_member_spread_data", "psd_member_spread"), ("14_rank_histogram_data", "rank_histogram")):
        shutil.copy2(os.path.join(ENG_SUMMARY, f"{src}_4window.csv"), os.path.join(OUT, f"{name}.csv"))  # exact copy
        written[f"{name}.csv"] = f"EngressNet summary {src}_4window.csv (copied unchanged)"
        print(f"  {name}.csv  ({written[f'{name}.csv']})")

    # domain-mean SIT and pattern-correlation time series: det column = det-MSE U-Net, mean per date
    for src, name, has_truth in (("01_domain_mean_sit_timeseries_data", "domain_mean_sit_timeseries", True),
                                 ("01b_domain_mean_pattern_corr_timeseries_data", "domain_mean_pattern_corr_timeseries", False)):
        a = eng(src, parse_dates=["time"])
        g = det(src, parse_dates=["time"]).groupby("time").mean(numeric_only=True).reindex(a["time"])
        if has_truth:
            assert np.allclose(g["truth"].values, a["truth"].values, atol=1e-5), "truth differs between runs"
        a["deterministic_unet"] = g["deterministic_unet"].values
        save(a, name, f"EngressNet summary {src}_4window.csv; deterministic_unet = det-MSE per-date mean")

    # community nearest-cell series
    a = eng("02_candidate_point_timeseries_data", parse_dates=["time"])
    g = det("02_candidate_point_timeseries_data", parse_dates=["time"]).groupby(["point", "method", "time"])["value"].mean()
    key = pd.MultiIndex.from_frame(a[["point", "method", "time"]])
    t = a["method"] == "truth"
    assert np.allclose(g.reindex(key)[t.values].values, a.loc[t, "value"].values, atol=1e-5), "truth differs between runs"
    d = a["method"] == "deterministic_unet"
    a.loc[d, "value"] = g.reindex(key)[d.values].values
    save(a, "community_point_timeseries", "EngressNet summary 02_..._4window.csv; deterministic_unet = det-MSE per-date mean")

    # Taylor statistics and power spectra (det: mean over the det runs, i.e. the single run for final2)
    a = eng("11_taylor_stats")
    dt = det("11_taylor_stats").groupby("Method").mean(numeric_only=True)
    for c in dt.columns:
        a.loc[a["Method"] == "Deterministic UNet", c] = dt.loc["Deterministic UNet", c]
    save(a, "taylor_stats", "EngressNet summary 11_taylor_stats_4window.csv; Deterministic UNet row = det-MSE")
    a = eng("13_psd_data")
    dp = det("13_psd_data").groupby("wavenumber_cycles_per_km", sort=False).mean(numeric_only=True)
    assert np.allclose(dp["Truth"].values, a["Truth"].values, rtol=1e-4), "truth PSD differs between runs"
    a["Deterministic UNet"] = dp["Deterministic UNet"].values
    save(a, "psd", "EngressNet summary 13_psd_data_4window.csv; Deterministic UNet column = det-MSE")

    # MESACLIP seasonal cycle per member (Fig. 8 band): monthly means of each member's domain-mean truth SIT, 2021, ocean cells
    rd = sorted(d for d in glob.glob(os.path.join(BASE, "results", ENG, f"*{FS['map_window']}*")) if os.path.isdir(d))[0]
    z = np.load(os.path.join(rd, "eval_data", "fields.npz"))
    truth, w = z["Y_test_phys"][:, 0], (1.0 - z["mask_test"][0, 0]).clip(0, 1)   # same ocean weights as the domain means
    st = pd.read_csv(os.path.join(rd, "eval_data", "sample_times.csv"), parse_dates=["time"])
    member = np.arange(len(st)) // st["time"].nunique()                      # samples are stored member-block by member-block
    dm = pd.DataFrame({"month": st["time"].dt.month, "member": member,
                       "sit": (truth * w).sum(axis=(1, 2)) / w.sum()})
    sc = dm.groupby(["month", "member"])["sit"].mean().unstack().add_prefix("member_")
    sc["mean"], sc["std"] = sc.mean(axis=1), sc.std(axis=1, ddof=1)
    save(sc, "mesaclip_seasonal_cycle_members",
         f"truth of {os.path.basename(rd)}: monthly mean domain-mean SIT per MESACLIP member, their mean and std (ddof=1)")

    # 2021 seasonal cycle of the domain-mean SIT, both datasets (perfect model): truth, EngressNet ensemble mean and bilinear from
    # the EngressNet runs, deterministic = the det-MSE U-Net's runs; per-date mean over runs and MESACLIP members, then monthly mean
    rows = []
    for ds, label in (("MESA", "MESACLIP"), ("FOSI", "FOSI")):
        def runs(batch):
            return sorted(d for d in glob.glob(os.path.join(BASE, "evaluation", "saved_figs", batch, "*"))
                          if os.path.isdir(d) and not d.endswith("_4window_aggregate"))
        f = "01_domain_mean_sit_timeseries_data.csv"
        e = pd.concat([pd.read_csv(os.path.join(d, f), parse_dates=["time"]) for d in runs(FS[ds]["eng"])])
        dd = pd.concat([pd.read_csv(os.path.join(d, f), parse_dates=["time"]) for d in runs(FS[ds]["det"])])
        e = e.groupby("time")[["truth", "stochastic_unet_mean", "bilinear"]].mean()
        dd = dd.groupby("time")[["truth", "deterministic_unet"]].mean().reindex(e.index)
        assert np.allclose(dd["truth"].values, e["truth"].values, atol=1e-5), f"{ds}: truth differs between runs"
        m = e.join(dd[["deterministic_unet"]]).groupby(lambda t: t.month).mean()
        m.index.name = "month"
        rows.append(m.rename(columns={"stochastic_unet_mean": "engressnet_mean"}).assign(dataset=label).reset_index())
    sc2 = pd.concat(rows, ignore_index=True)[["dataset", "month", "truth", "engressnet_mean", "deterministic_unet", "bilinear"]]
    save(sc2, "seasonal_cycle_2021", "monthly mean domain-mean SIT (2021, ocean cells), MESACLIP and FOSI: truth, EngressNet ensemble "
         "mean, bilinear (EngressNet runs) and the det-MSE U-Net (its own runs)")

    # EngressNet members' seasonal cycle, both datasets: monthly mean of each member's domain-mean SIT (2021, ocean cells; MESACLIP
    # pooled over its 6 members per date), plus two spreads: member_std = std of the members' domain means (what the member lines
    # show), cell_std = domain mean of the per-cell ensemble std (the local spread, as in Fig. 7 / the supplementary seasonal cycle)
    rows = []
    for ds, label in (("MESA", "MESACLIP"), ("FOSI", "FOSI")):
        for rd in sorted(d for d in glob.glob(os.path.join(BASE, "results", FS[ds]["eng"], "*")) if os.path.isdir(d)):
            z = np.load(os.path.join(rd, "eval_data", "fields.npz"))
            P, w = z["preds_all_phys"][:, :, 0], (1.0 - z["mask_test"][0, 0]).clip(0, 1)
            dmean = lambda a: (a * w).sum(axis=(-2, -1)) / w.sum()
            st = pd.read_csv(os.path.join(rd, "eval_data", "sample_times.csv"), parse_dates=["time"])
            mem = np.stack([dmean(P[:, k]) for k in range(P.shape[1])], axis=1)                       # (samples, K)
            cs = np.concatenate([dmean(P[i:i + 100].std(axis=1, ddof=1)) for i in range(0, P.shape[0], 100)])
            df = pd.DataFrame(mem, columns=[f"member_{k:02d}" for k in range(P.shape[1])])
            df["cell_std"], df["time"] = cs, st["time"].values
            daily = df.groupby("time").mean()                                                          # mean over MESACLIP members per date
            m = daily.groupby(daily.index.month).mean()
            m.index.name = "month"
            mc = [c for c in m.columns if c.startswith("member_")]
            m["mean"], m["member_std"] = m[mc].mean(axis=1), m[mc].std(axis=1, ddof=1)
            rows.append(m.assign(dataset=label).reset_index())
            del P, z
    em = pd.concat(rows, ignore_index=True)
    save(em, "engressnet_members_seasonal_cycle_2021", "EngressNet runs' members: monthly mean domain-mean SIT per member, their "
         "mean, member_std (spread of the members' domain means) and cell_std (domain mean of the per-cell ensemble std)")

    json.dump({"final_set": FINAL_SET, "written": datetime.now().isoformat(timespec="seconds"),
               "engressnet_batch": ENG, "engressnet_summary_dir": ENG_SUMMARY, "det_batch": DET,
               "det_runs": [os.path.basename(d) for d in DET_RUNS], "files": written},
              open(os.path.join(OUT, "sources.json"), "w"), indent=1)
    print("done:", OUT)


if __name__ == "__main__":
    main()
