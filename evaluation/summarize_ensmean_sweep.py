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

# 2026-10-24: only the final batches are kept on disk (coarse-zero rule at inference + original truth); the sweep batches
# were deleted, and their CSVs in corrected_calibration/ hold pre-rule numbers, so they are no longer listed here.
BATCHES = {
    "MESA": [("det-MSE + EMA", "MESA_detmse_ema"), ("EngressNet + EMA", "MESA_final_ema")],
    "FOSI": [("det-MSE + EMA", "FOSI_detmse_ema"), ("EngressNet + EMA", "FOSI_final_ema")],
}


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
        if pending:
            out.append("\nNot scored yet: " + "; ".join(pending))
        out.append("")
    text = "\n".join(out)
    with open(os.path.join(D, "ensmean_sweep_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
