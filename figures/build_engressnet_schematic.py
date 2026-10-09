"""
EngressNet (Version6 final config) architecture schematic.

Matches training/functions_engressnet.py UNet + the FINAL configuration in recommended_config.md
(NOISE_AMP_MAP=true, ENERGY_PATCH_SIZE=10, NOISE_LR_MULT=10, K=20, BETA=0.8, DATA_VARIANT=avg).
Writes engressnet_schematic_v6.{png,svg,pdf} next to this script.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engressnet_schematic_v6")

plt.rcParams["font.family"] = "serif"
plt.rcParams["svg.fonttype"] = "none"  # keep text editable in the SVG

C = {
    "enc": ("#eeeefc", "#8a8ae0", "#2f2f80", "#7a7acc"),
    "dec": ("#e7f5ec", "#4caf73", "#1b5e20", "#3d8b57"),
    "inp": ("#f0f0f0", "#999999", "#222222", "#666666"),
    "bot": ("#f1eee8", "#555555", "#222222", "#666666"),
    "out": ("#e6f0fb", "#6a9fd8", "#1f4e8c", "#3f78bf"),
    "loss": ("#f6d0d0", "#cc3333", "#b00000", "#b03030"),
}
GREY = "#777777"

fig, ax = plt.subplots(figsize=(10.2, 6.8))
ax.set_xlim(0, 1020)
ax.set_ylim(680, 0)  # y down, like a drawing canvas
ax.axis("off")


def box(x, y, w, h, kind, title, lines=(), dashed=False, title_size=11, line_size=9):
    fill, edge, tcol, lcol = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7", fc=fill, ec=edge,
                                lw=1.2, ls=(0, (4, 3)) if dashed else "-"))
    n = len(lines)
    step = 17
    top = y + h / 2 - (n * step) / 2
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=tcol)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, top + (i + 1) * step, ln, ha="center", va="center", fontsize=line_size, color=lcol)


def arrow(xy0, xy1, dashed=False, color="#222222"):
    ax.annotate("", xy=xy1, xytext=xy0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=10,
                                ls=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0))


def line(xs, ys, dashed=False, color="#222222", lw=1.0):
    ax.plot(xs, ys, color=color, lw=lw, ls=(0, (4, 3)) if dashed else "-", solid_capstyle="butt")


def group(x, y, w, h, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", fc="none", ec="#aaaaaa",
                                lw=1.0, ls=(0, (4, 3))))
    ax.text(x + w / 2, y + 17, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color="#444444")
    ax.text(x + w / 2, y + 33, sub, ha="center", va="center", fontsize=8.5, color="#666666")


# ---- geometry ---------------------------------------------------------------
ROWS = [205, 295, 385, 475]          # encoder e1..e4 / decoder d1..d3 centers
BH = 62                              # block height
EX, EW = 205, 150                    # encoder column
DX, DW = 575, 160                    # decoder column
Y0 = 92                              # final-stage row center

# ---- encoder -----------------------------------------------------------------
group(EX - 22, 130, EW + 44, 385, "Encoder", "ConvBlock + 2×2 max-pool")
enc = [("4 → 64", "16×32"), ("64 → 128", "8×16"), ("128 → 256", "4×8"), ("256 → 512", "2×4")]
for (t, res), yc in zip(enc, ROWS):
    box(EX, yc - BH / 2, EW, BH, "enc", t, [f"ConvBlock · {res}"])
for i in range(3):
    arrow((EX + EW / 2, ROWS[i] + BH / 2), (EX + EW / 2, ROWS[i + 1] - BH / 2))
    ax.text(EX + EW / 2 + 6, (ROWS[i] + ROWS[i + 1]) / 2, "pool", fontsize=8.5, color=GREY, va="center")

# ---- inputs ------------------------------------------------------------------
IX, IW, IY, IH = 18, 135, 255, 110
box(IX, IY, IW, IH, "inp", "Inputs", ["SIT, SIC,", "U10, V10", "1°, 16×32 crop"], title_size=12)
line([IX + IW, IX + IW + 14, IX + IW + 14], [IY + IH / 2, IY + IH / 2, ROWS[0]])
arrow((IX + IW + 14, ROWS[0]), (EX, ROWS[0]))

# ---- bottleneck ---------------------------------------------------------------
BX, BY, BW = 395, 560, 150
box(BX, BY, BW, 56, "bot", "512 → 512", ["Bottleneck · 2×4"])
line([EX + EW / 2, EX + EW / 2], [ROWS[3] + BH / 2, BY + 28])
arrow((EX + EW / 2, BY + 28), (BX, BY + 28))
line([BX + BW, DX + DW / 2], [BY + 28, BY + 28])
arrow((DX + DW / 2, BY + 28), (DX + DW / 2, ROWS[2] + BH / 2))

# ---- decoder ------------------------------------------------------------------
group(DX - 22, 130, DW + 44, 385, "Decoder", "×2 bilinear + 3×3 conv")
dec = [("128 → 64", "16×32"), ("256 → 128", "8×16"), ("512 → 256", "4×8")]
for (t, res), yc in zip(dec, ROWS[:3]):
    box(DX, yc - BH / 2, DW, BH, "dec", t, ["+ Noise, Skip", res])
for i in (2, 1):
    arrow((DX + DW / 2, ROWS[i] - BH / 2), (DX + DW / 2, ROWS[i - 1] + BH / 2))

# skips e1->d1, e2->d2, e3->d3
for yc in ROWS[:3]:
    line([EX + EW, DX], [yc, yc], dashed=True, color="#999999")
    ax.text((EX + EW + DX) / 2, yc - 8, "skip (concat)", ha="center", fontsize=8, color=GREY, style="italic")

# final stage
FX, FW, FH = DX - 40, DW + 80, 88
box(FX, Y0 - FH / 2, FW, FH, "dec", "64 → 32 → 1",
    ["×4 bilinear → 0.1° grid (150×310)", "+ Noise, Land mask", "4-conv fusion, then output conv"],
    line_size=8.5, title_size=10.5)
arrow((DX + 14, ROWS[0] - BH / 2), (DX + 14, Y0 + FH / 2))

# ---- residual add + output + loss ----------------------------------------------
PX, PY = 865, Y0
ax.add_patch(Circle((PX, PY), 13, fc="white", ec="#222222", lw=1.1))
ax.text(PX, PY, "+", ha="center", va="center", fontsize=15)
arrow((FX + FW, PY), (PX - 13, PY))
ax.text((FX + FW + PX - 13) / 2, PY - 9, "increment", ha="center", fontsize=8, color=GREY, style="italic")

# bilinear-SIT baseline path from the inputs
BLY = 28
line([IX + IW / 2, IX + IW / 2], [IY, BLY], dashed=True, color=GREY)
line([IX + IW / 2, PX], [BLY, BLY], dashed=True, color=GREY)
arrow((PX, BLY), (PX, PY - 13), dashed=True, color=GREY)
ax.text(400, BLY - 9, "Bilinear SIT baseline (1° → 0.1°)", ha="center", fontsize=9, color="#555555", style="italic")

OX, OW, OY, OH = 790, 150, 150, 100
arrow((PX, PY + 13), (PX, OY))
box(OX, OY, OW, OH, "out", "Output", ["de-normalize,", "clamp ≥ 0, land = 0", "K = 20 members"], title_size=12)

LX, LW, LY, LH = 765, 200, 290, 118
arrow((PX, OY + OH), (PX, LY), dashed=True)
box(LX, LY, LW, LH, "loss", "Loss (training)",
    ["Energy score, β = 0.8", "on 10×10-px tiles", "coastal weight ×2 (≤ 5 px)", "vs. 0.1° CESM SIT truth"],
    title_size=11.5, line_size=9)

# ---- noise callout -------------------------------------------------------------
NX, NY = 768, 450
noise_txt = [
    "One noise stage per decoder level (×4):",
    "• independent white noise at stage res.",
    "• 5×5 Gaussian kernel, learnable width",
    "• × learnable scalar amplitude",
    "• × state-dependent amplitude map",
    "   (3×3 conv + softplus on features)",
    "• 1×1 conv into the decoder features",
]
for i, t in enumerate(noise_txt):
    ax.text(NX, NY + i * 17, t, fontsize=8.5, color="#555555", va="center",
            fontweight="bold" if i == 0 else "normal")
for yc in ROWS[:3]:
    line([DX + DW, NX - 6], [yc, NY - 4], color="#888888", lw=0.8)
line([FX + FW, NX - 6], [Y0 + FH / 2 - 10, NY - 4], color="#888888", lw=0.8)

fig.tight_layout(pad=0.3)
for ext in ("png", "svg", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200 if ext == "png" else None, facecolor="white")
print(f"wrote {OUT}.png/.svg/.pdf")
