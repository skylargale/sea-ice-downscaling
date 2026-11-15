"""
Revisions 2026-11: inference cost of the revised final model for the paper's compute paragraph. GPU timing (CUDA events, after
warm-up) of one forward pass at the paper's domain size (16 x 32 coarse input -> 150 x 310 output), for
  EngressNet (results/FINAL2_MESA_random1825, noise on): one member (batch 1) and a 20-member ensemble for one date (batch 20)
  deterministic U-Net (results/FINAL2_MESA_detmse_random1825, deterministic pass): one prediction (batch 1)
plus wall-clock training time of every FINAL2 run from its files (run_config.json written at start -> model_state_dict.pt at the
end of training; excludes evaluation). Writes results/revisions_2026-11/compute_cost_final2.csv.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

V6 = "/glade/work/skygale/projects/SeaIceDownscaling"
sys.path.insert(0, os.path.join(V6, "training")); sys.path.insert(0, os.path.join(V6, "evaluation"))
import functions_engressnet as fe  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
rows = []


def time_model(run_glob, label, batch, deterministic, n=300):
    rd = glob.glob(run_glob)[0]
    c = json.load(open(os.path.join(rd, "run_config.json")))
    m = fe.UNet(in_channels=3, mask_channels=1, noise_amp_map=bool(c.get("noise_amp_map"))).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(rd, "model_state_dict.pt"), map_location=dev))
    x = torch.randn(batch, 3, 16, 32, device=dev); mk = (torch.rand(batch, 1, 150, 310, device=dev) > 0.7).float()
    with torch.inference_mode():
        for _ in range(30):
            m(x, (150, 310), mask=mk, deterministic=deterministic)
        if dev.type == "cuda":
            torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n):
                m(x, (150, 310), mask=mk, deterministic=deterministic)
            e.record(); torch.cuda.synchronize(); ms = s.elapsed_time(e) / n
        else:
            import time
            t0 = time.perf_counter()
            for _ in range(n // 10):
                m(x, (150, 310), mask=mk, deterministic=deterministic)
            ms = (time.perf_counter() - t0) * 1000 / (n // 10)
    rows.append({"item": f"inference: {label}", "value": ms, "unit": "ms per forward pass", "batch": batch,
                 "params": sum(p.numel() for p in m.parameters())})
    print(rows[-1], flush=True)


time_model(f"{V6}/results/FINAL2_MESA_random1825/*/", "EngressNet, 1 member", 1, False)
time_model(f"{V6}/results/FINAL2_MESA_random1825/*/", "EngressNet, 20 members (one date)", 20, False)
time_model(f"{V6}/results/FINAL2_MESA_detmse_random1825/*/", "deterministic U-Net, 1 prediction", 1, True)

for b in sorted(glob.glob(f"{V6}/results/FINAL2_*")):
    for rd in sorted(glob.glob(f"{b}/*/")):
        try:
            t = (os.path.getmtime(os.path.join(rd, "model_state_dict.pt")) - os.path.getmtime(os.path.join(rd, "run_config.json"))) / 3600
        except OSError:
            continue
        rows.append({"item": f"training wall-clock: {os.path.basename(b)}/{os.path.basename(os.path.normpath(rd))}", "value": t,
                     "unit": "h (1 V100, data load + training, no evaluation)", "batch": np.nan, "params": np.nan})
df = pd.DataFrame(rows)
df.to_csv(f"{V6}/results/revisions_2026-11/compute_cost_final2.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
print(df.round(4).to_string(index=False))
