"""
Ported from Version5/application/build_piomas_fosi_mesa_seasonal_cycle.py for the Section 4.5
Observational Consistency figure/seasonal-amplitude numbers, pointed at the Version6 noise-cascade
checkpoints (test year 2021, not Version5's 2020) instead of rewriting from scratch. Both training
windows evaluate on the same fixed 2021 test period, so any one window's truth is representative --
uses the 2015-2020 window for each dataset.

Unlike the Version5 original, this reports amplitude from the RAW (unshifted) PIOMAS monthly
series as the primary number -- the +1-month shift in the original was a display choice for that
analysis, not independently re-justified here.
"""
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/glade/work/skygale/projects/SeaIceDownscaling/Version6/evaluation")
from run_daily_eval_batch import build_piomas_regridded  # noqa: E402

BASE = "/glade/work/skygale/projects/SeaIceDownscaling/Version6/results"
RUNS = {
    "FOSI": f"{BASE}/FOSI_noise_cascade_avg/FOSI_noisecasc_avg_2015-2020_2021_5962017.casper-pbs",
    "MESA": f"{BASE}/MESA_noise_cascade_avg/MESA_noisecasc_avg_2015-2020_2021_5956149.casper-pbs",
}

OUT_DIR = f"{BASE}/../figures"

monthly = {}
piomas_monthly_ref = None
for label, run_dir in RUNS.items():
    edir = f"{run_dir}/eval_data"
    fields = np.load(f"{edir}/fields.npz")
    mask_test = fields["mask_test"]
    truth = fields["Y_test_phys"][:, 0]  # (N,H,W)
    ocean_w = (1.0 - mask_test[0, 0]).clip(0, 1)
    weight_sum = ocean_w.sum()

    with open(f"{edir}/tile_geometry.pkl", "rb") as f:
        tg = pickle.load(f)
    target_lat, target_lon = np.asarray(tg[0]["target_lat"]), np.asarray(tg[0]["target_lon"])

    st = pd.read_csv(f"{edir}/sample_times.csv")
    st["time"] = pd.to_datetime(st["time"])

    domain_mean_truth = (truth * ocean_w[None]).sum(axis=(1, 2)) / weight_sum
    df = pd.DataFrame({"time": st["time"], "truth": domain_mean_truth})
    df["month"] = df["time"].dt.month
    monthly[label] = df.groupby("month")["truth"].mean()

    piomas_regridded = build_piomas_regridded(target_lat, target_lon, st["time"].values)
    valid = np.isfinite(piomas_regridded[:, 0])
    w3 = np.broadcast_to(ocean_w[None], valid.shape)
    num = np.where(valid, piomas_regridded[:, 0] * w3, 0.0).sum(axis=(1, 2))
    den = np.where(valid, w3, 0.0).sum(axis=(1, 2))
    piomas_dm = np.where(den > 0, num / den, np.nan)
    pdf = pd.DataFrame({"time": st["time"], "piomas": piomas_dm})
    pdf["month"] = pdf["time"].dt.month
    piomas_month = pdf.groupby("month")["piomas"].mean()
    if piomas_monthly_ref is None:
        piomas_monthly_ref = piomas_month
    else:
        diff = (piomas_month - piomas_monthly_ref).abs().max()
        print(f"[check] PIOMAS monthly series FOSI-run vs MESA-run max diff: {diff:.5f} m")

out = pd.DataFrame({
    "FOSI truth": monthly["FOSI"],
    "MESA truth": monthly["MESA"],
    "PIOMAS (obs)": piomas_monthly_ref,
}).sort_index()
out.index.name = "month"
print(out.round(4))
out.to_csv(f"{OUT_DIR}/piomas_fosi_mesa_seasonal_cycle_2021.csv")

amplitudes = (out.max() - out.min()).round(4)
print("\nSeasonal amplitudes (max-min of monthly climatology, m):")
print(amplitudes)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(out.index, out["PIOMAS (obs)"], color="black", linewidth=2.2, marker="o", markersize=4, label="PIOMAS (obs)")
ax.plot(out.index, out["FOSI truth"], color="tab:blue", linewidth=1.8, marker="o", markersize=3, label="FOSI (perfect-model truth)")
ax.plot(out.index, out["MESA truth"], color="tab:orange", linewidth=1.8, marker="o", markersize=3, label="MESACLIP (perfect-model truth)")
ax.set_xticks(range(1, 13))
ax.set_xlabel("Month")
ax.set_ylabel("Domain-mean sea ice thickness (m)")
ax.set_title("Seasonal cycle: PIOMAS vs. FOSI and MESACLIP, 2021")
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
out_png = f"{OUT_DIR}/piomas_fosi_mesa_seasonal_cycle_2021.png"
fig.savefig(out_png, dpi=150)
print(f"\nSaved: {out_png}")
