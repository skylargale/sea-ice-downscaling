"""
Summarize the 2026-10-17 ensemble-mean RMSE sweep (EMA / mean-MSE term / coastal boost / pooled years / seeds) against the
det-MSE controls, from the CSVs that recompute_ocean_metrics.py and recompute_spread_skill.py write to
evaluation/corrected_calibration/. Batches that are missing or not scored yet are listed as pending.

Writes evaluation/corrected_calibration/ensmean_sweep_summary.txt and prints the same text.

Usage:
    python evaluation/summarize_ensmean_sweep.py
"""
import os

import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corrected_calibration")
METRICS = ["RMSE", "MAE", "Bias", "Pattern Corr", "Grad MAE", "SSIM", "IIEE", "Coastal RMSE"]
LOWER_BETTER = {"RMSE", "MAE", "Grad MAE", "IIEE", "Coastal RMSE", "CRPS"}
ENS = "EngressNet ensemble mean"  # the row recompute_ocean_metrics writes for every batch (det-MSE: its only prediction)

BATCHES = {
    "MESA": [("det-MSE (original)", "MESA_deterministic_mse_avg"), ("det-MSE + EMA", "MESA_detmse_ema"),
             ("det-MSE + EMA, pooled 2000-20", "MESA_detmse_ema_pooled"),
             ("EngressNet final (2026-10-07)", "MESA_final_avg"), ("EngressNet final, K=100", "MESA_final_avg_K100"),
             ("EngressNet + EMA", "MESA_final_ema"), ("EngressNet + EMA, K=100", "MESA_final_ema_K100"), ("EngressNet + EMA + mean term x1", "MESA_final_emamm1"),
             ("EngressNet + EMA + mean term x5", "MESA_final_emamm5"), ("EngressNet + EMA, coastal x3", "MESA_final_ema_cb3"),
             ("EngressNet + EMA, pooled 2000-20", "MESA_final_ema_pooled")],
    "FOSI": [("det-MSE (original)", "FOSI_deterministic_mse_avg"), ("det-MSE + EMA", "FOSI_detmse_ema"),
             ("det-MSE + EMA + cosine", "FOSI_detmse_ema_cos"), ("det-MSE + EMA + cosine, 40 ep", "FOSI_detmse_ema_cos_e40"),
             ("det-MSE + EMA, pooled 2000-20", "FOSI_detmse_ema_pooled"),
             ("EngressNet final (2026-10-07)", "CAL1a_noiselr_FOSI"), ("EngressNet final, K=100", "CAL1a_noiselr_FOSI_K100"),
             ("EngressNet + EMA", "FOSI_final_ema"), ("EngressNet + EMA, K=100", "FOSI_final_ema_K100"), ("EngressNet + EMA + mean term x1", "FOSI_final_emamm1"),
             ("EngressNet + EMA + mean term x5", "FOSI_final_emamm5"),
             ("EngressNet + EMA + cosine + noise warmup", "FOSI_final_ema_cos_nw3"),
             ("  ... + 40 epochs", "FOSI_final_ema_cos_nw3_e40"), ("  ... + beta 1.2", "FOSI_final_ema_cos_nw3_b12"),
             ("EngressNet + EMA, coastal x3", "FOSI_final_ema_cb3"), ("EngressNet + EMA, pooled 2000-20", "FOSI_final_ema_pooled")],
}
SEEDS = {"EngressNet + EMA": "{ds}_final_ema", "det-MSE + EMA": "{ds}_detmse_ema"}


def load(batch):
    """4-window mean metrics + pooled CRPS/spread-skill + per-window RMSE, or None if not scored."""
    om, ss = os.path.join(D, f"{batch}_ocean_metrics.csv"), os.path.join(D, f"{batch}.csv")
    if not os.path.exists(om):
        return None
    d = pd.read_csv(om)
    rows = d[(d.method == ENS) & (d.run != "MEAN over windows")]
    if rows.empty:
        return None
    m = rows[METRICS].mean()
    if os.path.exists(ss):
        s = pd.read_csv(ss)
        p = s[s.run == "POOLED"]
        if not p.empty:
            m["CRPS"] = p["CRPS (ocean)"].iloc[0]
            m["Spread-skill"] = p["Spread-skill ratio"].iloc[0] if p["Spread-skill ratio"].iloc[0] > 0 else float("nan")
    return m, list(rows.RMSE.round(4)), len(rows)


def main():
    out = []
    for ds, batches in BATCHES.items():
        out.append(f"===== {ds} (test year 2021, ocean cells; 4-window mean, or the single pooled window) =====")
        table, wins, pending = {}, {}, []
        for label, b in batches:
            r = load(b)
            if r is None:
                pending.append(f"{label} [{b}]")
                continue
            table[f"{label} (n={r[2]})"], wins[label] = r[0], r[1]
        if table:
            t = pd.DataFrame(table).T
            out.append(t.round(4).to_string())
            def control(k):
                # matched det-MSE control: pooled vs pooled, EMA vs EMA, otherwise the original det-MSE
                want = ("det-MSE + EMA, pooled" if "pooled" in k else
                        "det-MSE + EMA (" if "EMA" in k else "det-MSE (original)")
                hits = [r for r in table if r.startswith(want)]
                return hits[0] if hits else None
            if any(k.startswith("det-MSE") for k in table):
                out.append("\nEngressNet rows vs their matched det-MSE control (pooled vs pooled, EMA vs EMA, else original): "
                           "metrics where EngressNet is better")
                for k in t.index:
                    if not k.startswith("EngressNet"):
                        continue
                    ref = control(k)
                    if ref is None:
                        out.append(f"  {k}: control not scored yet")
                        continue
                    base = t.loc[ref]
                    better = []
                    for c in METRICS + ["CRPS"]:
                        if c not in t.columns or pd.isna(t.loc[k, c]):
                            continue
                        bc = base["MAE"] if c == "CRPS" else base[c]  # deterministic CRPS = MAE
                        if c == "Bias":
                            ok = abs(t.loc[k, c]) < abs(bc)
                        elif c in LOWER_BETTER:
                            ok = t.loc[k, c] < bc
                        else:
                            ok = t.loc[k, c] > bc
                        better.append(c if ok else f"({c})")
                    out.append(f"  {k} vs {ref.split(' (n=')[0]}: " + ", ".join(better) + "   [parenthesized = det-MSE better]")
            out.append("\nRMSE per training window:")
            for k, v in wins.items():
                out.append(f"  {k}: {v}")
        # seed spread: seed 0 = the base batch, seeds 1-2 = *_seed1/_seed2
        out.append("\nSeed spread (seeds 0-2; 4-window mean RMSE / CRPS per seed):")
        for label, pat in SEEDS.items():
            vals = []
            for suffix in ["", "_seed1", "_seed2"]:
                r = load(pat.format(ds=ds) + suffix)
                if r is not None:
                    vals.append((round(r[0]["RMSE"], 4), round(r[0].get("CRPS", float("nan")), 4)))
            out.append(f"  {label}: {vals if vals else 'pending'}")
        if pending:
            out.append("\nNot scored yet: " + "; ".join(pending))
        out.append("")
    text = "\n".join(out)
    with open(os.path.join(D, "ensmean_sweep_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
