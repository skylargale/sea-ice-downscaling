"""
observing_perfect_model_ceiling_comparison.py

Test C: a fairer baseline than bilinear. Bilinear has zero learned
parameters and zero training-distribution exposure -- "beats/loses to
bilinear" isn't really asking "is the network good," it's asking "is a
trivial linear operator more robust to distribution shift than a trained
one" (usually yes, for any trained model, and not very informative on its
own). The more relevant question for a domain-transfer result: how much
does the network's OWN accuracy degrade going from its best case
(FOSI-trained, FOSI-tested -- same distribution) to the real-world case
(FOSI-trained, PIOMAS-tested, scored against independent CryoSat-2 truth)?

Pulls the perfect-model ceiling directly from the exact checkpoint's own
original training-time evaluation (same run, same weights, no re-run
needed) and compares against the already-computed PIOMAS-vs-CryoSat
numbers.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEILING_METRICS = ("/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/"
                    "FOSI_stochastic_refine_bilinear_2000_2020_avg/"
                    "FOSI_refine_bilin_avg_2015-2019_2020_5631173.casper-pbs/metrics.csv")
REALWORLD_CSV = ("/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/"
                  "FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs/cryosat2_validation/"
                  "network_vs_bilinear_coastal_bias.csv")
OUT_DIR = ("/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020/"
           "FOSI_recommended_infer_2020_daily_v2_landfixed_5714638.casper-pbs/cryosat2_validation")

ceiling = pd.read_csv(CEILING_METRICS)
ceiling_row = ceiling[ceiling["Method"] == "Stochastic UNet Mean"].iloc[0]
ceiling_rmse_domain = float(ceiling_row["RMSE"])
ceiling_rmse_coastal = float(ceiling_row["Coastal RMSE"])
bilin_ceiling_row = ceiling[ceiling["Method"] == "Bilinear"].iloc[0]
bilin_ceiling_rmse_domain = float(bilin_ceiling_row["RMSE"])
bilin_ceiling_rmse_coastal = float(bilin_ceiling_row["Coastal RMSE"])

real = pd.read_csv(REALWORLD_CSV)
real_coastal = real[real["region"] == "coastal"]
real_interior = real[real["region"] == "interior"]

def weighted_rmse(df, col):
    return float(np.average(df[col], weights=df["n_valid"]))

net_rw_coastal = weighted_rmse(real_coastal, "network_rmse_vs_cryosat")
bilin_rw_coastal = weighted_rmse(real_coastal, "bilinear_rmse_vs_cryosat")
net_rw_interior = weighted_rmse(real_interior, "network_rmse_vs_cryosat")
bilin_rw_interior = weighted_rmse(real_interior, "bilinear_rmse_vs_cryosat")
# Domain-wide (coastal + interior pooled by cell count)
all_n = real_coastal["n_valid"].sum() + real_interior["n_valid"].sum()
net_rw_domain = (net_rw_coastal * real_coastal["n_valid"].sum() + net_rw_interior * real_interior["n_valid"].sum()) / all_n
bilin_rw_domain = (bilin_rw_coastal * real_coastal["n_valid"].sum() + bilin_rw_interior * real_interior["n_valid"].sum()) / all_n

rows = [
    {"scope": "Domain-wide", "method": "Network",
     "perfect_model_ceiling_rmse": ceiling_rmse_domain, "real_world_rmse_vs_cryosat": net_rw_domain,
     "degradation_factor": net_rw_domain / ceiling_rmse_domain},
    {"scope": "Domain-wide", "method": "Bilinear",
     "perfect_model_ceiling_rmse": bilin_ceiling_rmse_domain, "real_world_rmse_vs_cryosat": bilin_rw_domain,
     "degradation_factor": bilin_rw_domain / bilin_ceiling_rmse_domain},
    {"scope": "Coastal", "method": "Network",
     "perfect_model_ceiling_rmse": ceiling_rmse_coastal, "real_world_rmse_vs_cryosat": net_rw_coastal,
     "degradation_factor": net_rw_coastal / ceiling_rmse_coastal},
    {"scope": "Coastal", "method": "Bilinear",
     "perfect_model_ceiling_rmse": bilin_ceiling_rmse_coastal, "real_world_rmse_vs_cryosat": bilin_rw_coastal,
     "degradation_factor": bilin_rw_coastal / bilin_ceiling_rmse_coastal},
    {"scope": "Interior", "method": "Network",
     "perfect_model_ceiling_rmse": ceiling_rmse_domain, "real_world_rmse_vs_cryosat": net_rw_interior,
     "degradation_factor": net_rw_interior / ceiling_rmse_domain},
    {"scope": "Interior", "method": "Bilinear",
     "perfect_model_ceiling_rmse": bilin_ceiling_rmse_domain, "real_world_rmse_vs_cryosat": bilin_rw_interior,
     "degradation_factor": bilin_rw_interior / bilin_ceiling_rmse_domain},
]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "perfect_model_ceiling_vs_realworld.csv"), index=False)
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Figure: perfect-model ceiling vs real-world RMSE, network and bilinear
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
for ax, scope in zip(axes, ["Domain-wide", "Coastal", "Interior"]):
    sub = df[df["scope"] == scope]
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w / 2, sub["perfect_model_ceiling_rmse"], width=w, color="#8fd694", label="Perfect-model ceiling\n(FOSI-in/FOSI-out)")
    ax.bar(x + w / 2, sub["real_world_rmse_vs_cryosat"], width=w, color="#eb6834", label="Real-world\n(PIOMAS-in/CryoSat-truth)")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["method"])
    ax.set_title(scope)
    ax.set_ylabel("RMSE (m)")
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.annotate(f"{r['degradation_factor']:.1f}x", xy=(i, r["real_world_rmse_vs_cryosat"]),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle("How much does accuracy degrade from perfect-model to real-world application?\n"
             "(label = real-world RMSE / perfect-model ceiling RMSE)", y=1.05)
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "perfect_model_ceiling_vs_realworld.png")
fig.savefig(fig_path, dpi=180, bbox_inches="tight")
print("\nSaved figure:", fig_path)
print("Done.")
