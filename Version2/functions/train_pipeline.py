"""Top-level entry point for one EngressNet training run.

Mirrors ``build_dataset.run_pipeline`` for the data side: this is the
function-based replacement for the rest of your notebook (data loading,
patch extraction, training, evaluation), callable from a notebook or
script rather than only as a linear sequence of notebook cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.optim as optim
import xarray as xr

from .channels import NEW_PIPELINE_CHANNELS, ChannelSpec, apply_channel_processing, select_channels, find_channel_index
from .config import RegionBBox
from .evaluation import bilinear_baseline, compute_metrics_table, denormalize_all, evaluate_ensemble
from .land_mask import build_land_mask
from .model import build_model
from .patches import compute_normalization, extract_patches, train_test_split_fields
from .training import TrainConfig, train


@dataclass
class TrainingPipelineConfig:
    """Settings for one end-to-end training run, separate from
    ``config.PipelineConfig`` (which governs data PREPROCESSING) since you
    may want to train multiple models off of one preprocessed dataset.
    """

    x_path: str = "/glade/work/skygale/_projects/SeaIceDownscaling/data/X_perfmodexp_cons.zarr"
    y_path: str = "/glade/work/skygale/_projects/SeaIceDownscaling/data/Y_perfmodexp.zarr"

    # Channel handling -- must match what's actually in the saved X file.
    # If you saved X with this package's dataset_builder.py defaults, this
    # is ("hi", "aice", "U", "V"); if you're loading an OLD file from the
    # original 5-channel notebook pipeline, set this to
    # ("hi", "Tsfc", "SST", "uvel", "vvel") and pass
    # channel_specs=channels.LEGACY_POP_CHANNELS instead.
    channel_order: tuple[str, ...] = ("hi", "aice", "U", "V")
    channel_specs: dict[str, ChannelSpec] = field(default_factory=lambda: NEW_PIPELINE_CHANNELS)
    keep_channels: tuple[str, ...] = ("hi", "U", "V")  # subset actually fed to the model
    target_channel: str = "hi"  # which kept channel the bilinear baseline compares against

    # Land mask
    pop_grid_hr: str = "POP_tx0.1v2"
    region: RegionBBox | None = None  # set from config.PipelineConfig.region in practice
    weights_dir: str = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"

    # Patches
    context_size: tuple[int, int] = (16, 24)
    target_size: tuple[int, int] = (8, 12)
    stride: int = 4
    train_frac: float = 0.7
    split_seed: int | None = 0

    # Model / training
    latent_channels: int = 8
    train_config: TrainConfig = field(default_factory=TrainConfig)


def load_xy(config: TrainingPipelineConfig) -> tuple[xr.DataArray, xr.DataArray]:
    """Load X and Y from disk. Handles both Zarr (this package's default
    output) and NetCDF transparently based on the path suffix.
    """
    def _open(path: str):
        if path.endswith(".zarr"):
            return xr.open_zarr(path)
        return xr.open_dataset(path)

    X_ds = _open(config.x_path)
    Y_ds = _open(config.y_path)
    X_da = X_ds["X"] if "X" in X_ds else X_ds
    Y_da = Y_ds["Y"] if "Y" in Y_ds else Y_ds
    return X_da, Y_da


def prepare_training_data(config: TrainingPipelineConfig) -> dict:
    """Load, process channels, build the land mask, split, normalize, and
    extract patches -- replacing the notebook's "Data preparation" cell.

    Returns a dict with everything downstream steps need: X_train, Y_train,
    M_train, X_test, Y_test, x_stats, y_stats, channel_order (post-subset),
    target_channel_idx, land_mask, dst_lat, dst_lon.
    """
    if config.region is None:
        raise ValueError(
            "TrainingPipelineConfig.region must be set (e.g. to the same "
            "config.PipelineConfig.region used when the data was "
            "preprocessed) -- the land mask is subset to this bbox."
        )

    X_da, Y_da = load_xy(config)
    llat, llon = X_da["lat"].values, X_da["lon"].values
    hlat, hlon = Y_da["lat"].values, Y_da["lon"].values

    X = X_da.values
    Y = Y_da.values

    channel_order = list(config.channel_order)
    X = apply_channel_processing(X, channel_order, config.channel_specs)
    # Y's single channel (the target var, e.g. "hi") gets the same clip as
    # the matching X channel spec, mirroring the notebook's
    # `Y = np.clip(Y, None, 6.0)`.
    target_spec = config.channel_specs[config.target_channel]
    if target_spec.clip_min is not None or target_spec.clip_max is not None:
        Y = Y.clip(target_spec.clip_min, target_spec.clip_max)

    X, channel_order = select_channels(X, channel_order, list(config.keep_channels))
    target_channel_idx = find_channel_index(channel_order, config.target_channel)

    land_mask, dst_lat, dst_lon = build_land_mask(
        config.pop_grid_hr,
        dst_grid=xr.Dataset({"lat": ("lat", hlat), "lon": ("lon", hlon)}),
        region=config.region,
        weights_dir=config.weights_dir,
    )

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    X_train_f, Y_train_f, X_test_f, Y_test_f = train_test_split_fields(
        X_t, Y_t, train_frac=config.train_frac, seed=config.split_seed
    )

    x_stats = compute_normalization(X_train_f)
    y_stats = compute_normalization(Y_train_f)

    X_train_n = x_stats.normalize(X_train_f)
    X_test_n = x_stats.normalize(X_test_f)
    Y_train_n = y_stats.normalize(Y_train_f)
    Y_test_n = y_stats.normalize(Y_test_f)

    X_train, Y_train, M_train = extract_patches(
        X_train_n, Y_train_n, land_mask, config.context_size, config.target_size, config.stride
    )
    X_test, Y_test, _ = extract_patches(
        X_test_n, Y_test_n, land_mask, config.context_size, config.target_size, config.stride
    )

    return dict(
        X_train=X_train, Y_train=Y_train, M_train=M_train,
        X_test=X_test, Y_test=Y_test,
        x_stats=x_stats, y_stats=y_stats,
        channel_order=channel_order, target_channel_idx=target_channel_idx,
        land_mask=land_mask, dst_lat=dst_lat, dst_lon=dst_lon,
        llat=llat, llon=llon, hlat=hlat, hlon=hlon,
        # Full-domain (pre-patch) normalized fields, kept around for the
        # kind of whole-domain preview plot your original notebook made
        # against real lat/lon (X[0,0,0] vs llat/llon). X_train/Y_train
        # above are PATCHES (small windows), not full domain -- plotting
        # a patch against the full-domain lat/lon array will raise a
        # shape mismatch in contourf/pcolormesh, so don't mix the two.
        X_full=X_train_n, Y_full=Y_train_n,
    )


def run_training_pipeline(config: TrainingPipelineConfig) -> dict:
    """End-to-end: load data, train, evaluate. Returns a dict with the
    trained model, training history, and physical-units evaluation
    results / metrics table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = prepare_training_data(config)

    model = build_model(
        in_channels=data["X_train"].shape[1],
        latent_channels=config.latent_channels,
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.train_config.learning_rate)

    history = train(
        model, optimizer,
        data["X_train"], data["Y_train"], data["M_train"],
        config.train_config, device,
    )

    eval_result = evaluate_ensemble(
        model, data["X_test"], data["Y_test"],
        latent_channels=config.latent_channels, device=device,
    )
    baseline = bilinear_baseline(
        data["X_test"], data["Y_test"].shape[-2:], data["target_channel_idx"]
    )
    phys = denormalize_all(
        data["Y_test"], eval_result, data["X_test"], baseline,
        data["y_stats"], data["x_stats"], data["target_channel_idx"],
    )
    metrics = compute_metrics_table(phys)

    return dict(
        model=model, optimizer=optimizer, history=history,
        eval_result=eval_result, phys=phys, metrics=metrics,
        data=data,
    )
