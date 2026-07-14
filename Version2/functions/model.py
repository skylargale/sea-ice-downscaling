"""EngressNet: a UNet with an Engression-style stochastic decoder for
super-resolving sea ice thickness.

This is the architecture from your notebook, functionized but with the
same math -- I have not changed any of the convolution sizes, the
encoder/decoder structure, the noise injection points, or the residual
bilinear-baseline addition at the output. The one real bug fixed here:

* The original referenced ``model.module.latent_channels`` inside the
  training/eval loops, which only works if ``model`` is wrapped in
  ``nn.DataParallel`` (since ``.module`` is DataParallel's way of exposing
  the underlying model). On a single GPU, on CPU, or under
  ``DistributedDataParallel`` instead, that attribute access breaks. Fixed
  by having callers pass ``latent_channels`` explicitly (it's already a
  config value you control, not something that needs to be fished out of
  the model object) -- see ``training.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_noise(z: torch.Tensor) -> torch.Tensor:
    """3x3 average-pool smoothing applied to the latent noise field."""
    return F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)


def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    """Two 3x3 conv + InstanceNorm + ReLU layers -- the basic UNet block."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU(inplace=True),
    )


class EngressNet(nn.Module):
    """UNet encoder/decoder with latent noise injected at each decoder
    stage (concatenated, not additive -- matching whichever branch was
    left uncommented in your notebook), trained with an energy-score loss
    for stochastic, ensemble-style super-resolution.

    Forward signature is unchanged from your notebook's ``UNet.forward``:
    ``model(x, up_size, z=None)``. ``up_size`` is the target (H, W) to
    upsample the final output to (since your patches have non-integer
    LR->HR scale ratios baked into ``up_size`` rather than the model
    inferring it from a fixed stride).
    """

    def __init__(self, in_channels: int, latent_channels: int = 8):
        super().__init__()
        self.latent_channels = latent_channels

        # ---- Encoder ----
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = conv_block(256, 512)

        # ---- Bottleneck ----
        self.bottleneck = conv_block(512, 512)

        # ---- Decoder ----
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
        )
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
        )
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
        )
        self.dec1 = conv_block(128, 64)

        # Decoder noise projections
        self.z_proj_d3 = nn.Conv2d(latent_channels, 256, 1)
        self.z_proj_d2 = nn.Conv2d(latent_channels, 128, 1)
        self.z_proj_d1 = nn.Conv2d(latent_channels, 64, 1)

        # Scaling for additive noise (unused in the concatenated-noise path
        # below, kept since your notebook kept them defined for the
        # commented-out additive variant)
        self.noise_scale_d3 = nn.Parameter(torch.tensor(0.05))
        self.noise_scale_d2 = nn.Parameter(torch.tensor(0.05))
        self.noise_scale_d1 = nn.Parameter(torch.tensor(0.05))

        # Concatenation adapters
        self.concat_d3 = nn.Conv2d(256 + 256, 256, 1)
        self.concat_d2 = nn.Conv2d(128 + 128, 128, 1)
        self.concat_d1 = nn.Conv2d(64 + 64, 64, 1)

        # ---- Output ----
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4)
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        up_size: tuple[int, int],
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        if z is None:
            z = torch.randn(B, self.latent_channels, H // 8, W // 8, device=x.device)
            z = smooth_noise(z)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d3 = self.up3(b)
        zd3 = F.interpolate(z, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        zd3 = self.z_proj_d3(zd3)
        d3 = self.concat_d3(torch.cat([d3, zd3], dim=1))
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        zd2 = F.interpolate(z, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        zd2 = self.z_proj_d2(zd2)
        d2 = self.concat_d2(torch.cat([d2, zd2], dim=1))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        zd1 = F.interpolate(z, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        zd1 = self.z_proj_d1(zd1)
        d1 = self.concat_d1(torch.cat([d1, zd1], dim=1))
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.final_up(d1)
        out = self.out_conv(out)
        out = F.interpolate(out, size=up_size, mode="bilinear", align_corners=False)

        # Residual prediction: add the network's correction on top of a
        # bilinear upsample of the FIRST input channel. This assumes
        # channel 0 of `x` is the same physical variable as the target
        # (sea ice thickness) -- see channels.py for making that
        # assumption explicit and checked rather than positional.
        base = F.interpolate(x[:, 0:1], size=up_size, mode="bilinear", align_corners=False)

        return base + out


def build_model(
    in_channels: int,
    latent_channels: int = 8,
    device: torch.device | None = None,
    data_parallel: bool = False,
) -> nn.Module:
    """Construct an EngressNet and move it to device.

    ``data_parallel=True`` wraps in ``nn.DataParallel`` as your original
    notebook did. This is now an explicit choice rather than the default --
    on a single-GPU box it's a no-op wrapper that exists only to break
    ``model.latent_channels`` access (see module docstring), so leave it
    False unless you're actually running multi-GPU.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EngressNet(in_channels=in_channels, latent_channels=latent_channels)
    if data_parallel:
        model = nn.DataParallel(model)
    return model.to(device)
