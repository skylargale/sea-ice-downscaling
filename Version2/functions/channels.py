"""Name-based channel bookkeeping for the EngressNet predictor stack.

The original notebook hardcoded channel positions as plain integers --
``X[:, :, 3:5, :, :] /= 100.0`` for "uvel, vvel are channels 3 and 4",
``X[:, :, 0, :, :]`` for "SIT is channel 0", then ``X[:, :, [0, 3, 4], :, :]``
to subset down to 3 channels. This is correct ONLY if the channel order
in your saved X array is exactly
``["hi"/SIT, Tsfc, SST, uvel, vvel]`` in that position, every time.

That assumption breaks the moment you use ``dataset_builder.py`` from this
package, whose ``low_vars`` default is ``("hi", "aice", "U", "V")`` --
4 channels, no Tsfc/SST, and "U"/"V" are ATMOSPHERE winds (already in m/s,
not POP ocean velocities in cm/s). Silently running the old positional
code against this new data would, at best, raise an IndexError; at worst
it would silently divide the wrong channel by 100 and train on garbage
without any error at all.

This module makes channel identity and units explicit and checked, so a
channel-order mismatch fails loudly at preprocessing time instead of
training silently on misaligned data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelSpec:
    """One predictor channel's identity, units, and any unit conversion
    needed before training.
    """

    name: str
    units: str
    # Multiply raw values by this to reach the units you actually want to
    # train in. 1.0 = no conversion needed.
    convert_factor: float = 1.0
    # Physically-motivated clip bounds in the POST-conversion units, or
    # None to skip clipping. e.g. (None, 6.0) clips sea ice thickness to
    # <= 6m to remove the spurious CESM1.3 thick-ice artifacts your
    # notebook clipped by hand.
    clip_min: float | None = None
    clip_max: float | None = None


# Channel specs matching what your notebook's "Channel Notes" markdown
# documented for the OLD 5-channel X (SIT, Tsfc, SST, uvel, vvel from a POP
# coarsen-to-LR pipeline). Kept here for reference / for reprocessing old
# saved files -- NOT what dataset_builder.py in this package produces.
LEGACY_POP_CHANNELS = {
    "hi": ChannelSpec("hi", units="m", clip_max=6.0),  # SIT
    "Tsfc": ChannelSpec("Tsfc", units="degC"),
    "SST": ChannelSpec("SST", units="degC"),
    "uvel": ChannelSpec("uvel", units="m/s", convert_factor=1 / 100.0),  # cm/s -> m/s
    "vvel": ChannelSpec("vvel", units="m/s", convert_factor=1 / 100.0),  # cm/s -> m/s
}

# Channel specs matching THIS package's dataset_builder.py default
# low_vars = ("hi", "aice", "U", "V"). U/V come from the CESM atm
# component (config.VAR_COMPONENT["U"] == "atm"), which are already in
# m/s -- do NOT apply the legacy cm/s->m/s conversion to these.
NEW_PIPELINE_CHANNELS = {
    "hi": ChannelSpec("hi", units="m", clip_max=6.0),  # sea ice thickness
    "aice": ChannelSpec("aice", units="fraction", clip_min=0.0, clip_max=1.0),
    "U": ChannelSpec("U", units="m/s"),  # CAM atm wind -- verify against
                                          # your actual file's units attr;
                                          # do not assume cm/s here.
    "V": ChannelSpec("V", units="m/s"),
}


def apply_channel_processing(
    X: np.ndarray,
    channel_order: list[str],
    channel_specs: dict[str, ChannelSpec],
) -> np.ndarray:
    """Apply each channel's unit conversion and clip, by NAME, regardless
    of what position that channel happens to sit in.

    ``X`` has shape (..., channel, H, W) -- channel is whichever axis
    ``channel_order`` indexes (this package's X is
    (ensemble, time, channel, lat, lon), so channel axis is -3).

    Raises if ``channel_order`` contains a name with no entry in
    ``channel_specs``, rather than silently skipping it -- an unprocessed
    channel (no unit conversion applied when one was needed) is exactly
    the kind of silent-corruption bug the original positional indexing
    was prone to.
    """
    channel_axis = X.ndim - 3  # channel is 3rd-from-last in (..., C, H, W)
    out = X.copy()
    for pos, name in enumerate(channel_order):
        if name not in channel_specs:
            raise KeyError(
                f"Channel '{name}' (position {pos}) has no ChannelSpec in "
                f"channel_specs. Add one, or remove it from channel_order, "
                f"before processing -- silently skipping unit conversion "
                f"for an unrecognized channel is how the original notebook's "
                f"positional indexing went wrong."
            )
        spec = channel_specs[name]
        idx = [slice(None)] * out.ndim
        idx[channel_axis] = pos
        idx = tuple(idx)

        if spec.convert_factor != 1.0:
            out[idx] = out[idx] * spec.convert_factor
        if spec.clip_min is not None or spec.clip_max is not None:
            out[idx] = np.clip(out[idx], spec.clip_min, spec.clip_max)
    return out


def select_channels(
    X: np.ndarray,
    channel_order: list[str],
    keep: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Subset X to a named list of channels, by name, replacing the
    original's ``X[:, :, [0, 3, 4], :, :]`` positional subsetting.

    Returns the subset array and the new (reordered) channel_order list,
    so downstream code always has an authoritative name-to-position
    mapping instead of needing to remember it.
    """
    channel_axis = X.ndim - 3
    missing = [c for c in keep if c not in channel_order]
    if missing:
        raise KeyError(
            f"Requested channels {missing} not present in channel_order "
            f"{channel_order}."
        )
    positions = [channel_order.index(c) for c in keep]
    out = np.take(X, positions, axis=channel_axis)
    return out, list(keep)


def find_channel_index(channel_order: list[str], name: str) -> int:
    """Look up a channel's position by name, replacing hardcoded literals
    like ``sit_idx = 0`` in the original eval/baseline code.
    """
    try:
        return channel_order.index(name)
    except ValueError as e:
        raise KeyError(
            f"Channel '{name}' not found in channel_order {channel_order}."
        ) from e
