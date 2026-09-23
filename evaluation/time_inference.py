#!/usr/bin/env python3
"""One-off timing script: measures real forward-pass / ensemble-generation cost of
the trained per-stage noise-cascade checkpoint, on the actual GPU/domain size used
for the paper's MESA runs. Not part of the regular pipeline -- appendix-only."""
import sys
import time
import torch

sys.path.insert(0, "/glade/work/skygale/projects/SeaIceDownscaling/Version6/training")
sys.path.insert(0, "/glade/work/skygale/projects/SeaIceDownscaling/Version6/evaluation")
import functions_engressnet as fe

CKPT = "/glade/work/skygale/projects/SeaIceDownscaling/Version6/results/MESA_noise_cascade_avg/MESA_noisecasc_avg_2000-2005_2021_5956146.casper-pbs/model_state_dict.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Real domain: subdomain lat60-75/lon-182to-151 gives a (150, 310) high-res target
# and a (16, 24)-multiple low-res encoder input in this project's convention --
# read the actual shapes from this run's own run_config.json instead of assuming.
import json
with open(CKPT.replace("model_state_dict.pt", "run_config.json")) as f:
    cfg = json.load(f)
print("context_size (low-res encoder input):", cfg["context_size"])
print("target_size (used only for patches; no-patches run uses full subdomain)")

# in_channels: MESA X has (hi_d, aice_d, U10) = 3 channels; mask_channels = 1 (land mask)
in_channels = 3
mask_channels = 1
model = fe.UNet(in_channels=in_channels, mask_channels=mask_channels).to(device)
state_dict = torch.load(CKPT, map_location=device)
model.load_state_dict(state_dict)
model.eval()

B = 1
# Use the real eval_data shapes instead of guessing.
fields = __import__("numpy").load(CKPT.replace("model_state_dict.pt", "eval_data/fields.npz"))
up_size = fields["Y_test_phys"].shape[-2:]
print("real up_size (high-res target):", up_size)
H_lr_real, W_lr_real = fields["X_test_sit_phys"].shape[-2:]
print("real low-res input shape:", (H_lr_real, W_lr_real))

x = torch.randn(B, in_channels, H_lr_real, W_lr_real, device=device)
mask = torch.zeros(B, mask_channels, *up_size, device=device)

with torch.inference_mode():
    # warm-up (first call pays CUDA kernel compilation/allocation cost, not representative)
    for _ in range(3):
        _ = model(x, up_size=tuple(up_size), mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # single forward pass, averaged over N repeats
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        _ = model(x, up_size=tuple(up_size), mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    per_pass = (t1 - t0) / N
    print(f"single forward pass (batch=1): {per_pass*1000:.1f} ms, averaged over {N} calls")

    # a full K=20 ensemble, sequential calls (matches evaluate_model's own loop)
    K = 20
    t0 = time.perf_counter()
    for _ in range(K):
        _ = model(x, up_size=tuple(up_size), mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"full K={K} ensemble (batch=1, sequential): {(t1-t0):.2f} s total, {(t1-t0)/K*1000:.1f} ms/member")
