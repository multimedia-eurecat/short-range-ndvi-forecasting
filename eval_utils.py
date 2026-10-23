import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from matplotlib.patches import Rectangle
from PIL import Image


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def batch_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: batch_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [batch_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(batch_to_device(v, device) for v in x)
    return x


def clone_batch(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone_batch(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone_batch(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone_batch(v) for v in x)
    return x


def zero_nextcloud_input_channel(
    batch,
    input_mode: str,
    zeroing_nextcloud: bool = False,
):
    """Zero the final image-input channel when it represents next-cloud data."""
    if not zeroing_nextcloud or "nextcloud" not in str(input_mode).lower():
        return batch

    if "dynamic" not in batch or len(batch["dynamic"]) == 0:
        raise RuntimeError(
            "Cannot zero next-cloud input: batch['dynamic'][0] is missing."
        )

    dynamic = batch["dynamic"][0]
    if not torch.is_tensor(dynamic) or dynamic.ndim != 5 or dynamic.shape[2] < 1:
        raise RuntimeError(
            "Expected batch['dynamic'][0] with shape [B,T,C,H,W], got "
            f"{getattr(dynamic, 'shape', None)}."
        )

    # The dataset adapter always appends next-cloud as the final input channel.
    dynamic[:, :, -1, :, :] = 0.0
    return batch


def safe_name(x: str) -> str:
    x = Path(str(x)).stem
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x[:180]


def get_sample_name(batch, sample_idx: int, global_idx: int) -> str:
    for key in ["cubename", "filepath", "name", "text"]:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, (list, tuple)):
            value = value[sample_idx]
        return f"{global_idx:06d}_{safe_name(value)}"
    return f"{global_idx:06d}_sample_{sample_idx:03d}"


def save_tiff_stack(path: Path, array_t_hw: np.ndarray):
    """Save a [T,H,W] float32 array as a TIFF stack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_t_hw = np.asarray(array_t_hw, dtype=np.float32)
    if array_t_hw.ndim == 2:
        array_t_hw = array_t_hw[None]
    tifffile.imwrite(
        str(path),
        array_t_hw,
        photometric="minisblack",
        metadata={"axes": "TYX"},
    )


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _json_safe_rows(rows):
    cleaned = []
    for row in rows:
        cleaned_row = {}
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                cleaned_row[key] = None
            else:
                cleaned_row[key] = value
        cleaned.append(cleaned_row)
    return cleaned


def _device_generator(device, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


# ---------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------


def identity_batch(batch, context_length: int, **kwargs):
    return batch


def shuffle_ndvi_only(
    batch,
    context_length: int,
    ndvi_channel: int = 0,
    seed: int = 0,
    **kwargs,
):
    """Shuffle temporal order of NDVI context frames only."""
    batch = clone_batch(batch)
    dynamic = batch["dynamic"][0]

    for b in range(dynamic.shape[0]):
        generator = _device_generator(dynamic.device, seed + b)
        perm = torch.randperm(
            context_length,
            device=dynamic.device,
            generator=generator,
        )
        dynamic[
            b,
            :context_length,
            ndvi_channel : ndvi_channel + 1,
        ] = dynamic[
            b,
            perm,
            ndvi_channel : ndvi_channel + 1,
        ].clone()

    batch["dynamic"][0] = dynamic
    return batch


def reverse_ndvi_only(
    batch,
    context_length: int,
    ndvi_channel: int = 0,
    **kwargs,
):
    """Reverse temporal order of NDVI context frames only."""
    batch = clone_batch(batch)
    dynamic = batch["dynamic"][0]
    idx = torch.arange(
        context_length - 1,
        -1,
        -1,
        device=dynamic.device,
    )

    dynamic[
        :,
        :context_length,
        ndvi_channel : ndvi_channel + 1,
    ] = dynamic[
        :,
        idx,
        ndvi_channel : ndvi_channel + 1,
    ].clone()

    batch["dynamic"][0] = dynamic
    return batch


def repeat_last_ndvi(
    batch,
    context_length: int,
    ndvi_channel: int = 0,
    **kwargs,
):
    """Replace all NDVI context frames with the last context NDVI."""
    batch = clone_batch(batch)
    dynamic = batch["dynamic"][0]

    last_ndvi = dynamic[
        :,
        context_length - 1 : context_length,
        ndvi_channel : ndvi_channel + 1,
    ].clone()

    dynamic[
        :,
        :context_length,
        ndvi_channel : ndvi_channel + 1,
    ] = last_ndvi.expand(
        -1,
        context_length,
        -1,
        -1,
        -1,
    )

    batch["dynamic"][0] = dynamic
    return batch


def noise_ndvi(
    batch,
    context_length: int,
    ndvi_channel: int = 0,
    ndvi_min: float = 0.0,
    ndvi_max: float = 1.0,
    seed: int = 0,
    **kwargs,
):
    """Replace NDVI context frames with deterministic uniform noise."""
    batch = clone_batch(batch)
    dynamic = batch["dynamic"][0]
    shape = dynamic[
        :,
        :context_length,
        ndvi_channel : ndvi_channel + 1,
    ].shape

    generator = _device_generator(dynamic.device, seed)
    noise = torch.rand(
        shape,
        device=dynamic.device,
        dtype=dynamic.dtype,
        generator=generator,
    )
    noise = noise * (ndvi_max - ndvi_min) + ndvi_min

    dynamic[
        :,
        :context_length,
        ndvi_channel : ndvi_channel + 1,
    ] = noise

    batch["dynamic"][0] = dynamic
    return batch


def zero_ndvi(
    batch,
    context_length: int,
    ndvi_channel: int = 0,
    **kwargs,
):
    """Replace NDVI context frames with zeros."""
    batch = clone_batch(batch)
    dynamic = batch["dynamic"][0]
    dynamic[
        :,
        :context_length,
        ndvi_channel : ndvi_channel + 1,
    ] = 0.0
    batch["dynamic"][0] = dynamic
    return batch


def shuffle_weather_context(
    batch,
    context_length: int,
    seed: int = 0,
    **kwargs,
):
    """Shuffle temporal order of weather context frames only."""
    batch = clone_batch(batch)
    if "dynamic" not in batch or len(batch["dynamic"]) < 2:
        return batch

    meteo = batch["dynamic"][1]
    dim = meteo.shape[1]
    for b in range(meteo.shape[0]):
        generator = _device_generator(meteo.device, seed + b)
        perm = torch.randperm(
            dim, #context_length,
            device=meteo.device,
            generator=generator,
        ) 
        meteo[b, :, -4:] = meteo[b, perm, -4:].clone()

    batch["dynamic"][1] = meteo
    return batch

def shuffle_date_context(
    batch,
    context_length: int,
    seed: int = 0,
    **kwargs,
):
    """Shuffle temporal order of weather context frames only."""
    batch = clone_batch(batch)
    if "dynamic" not in batch or len(batch["dynamic"]) < 2:
        return batch

    meteo = batch["dynamic"][1]
    dim = meteo.shape[1]
    for b in range(meteo.shape[0]):
        generator = _device_generator(meteo.device, seed + b)
        perm = torch.randperm(
            dim, #context_length,
            device=meteo.device,
            generator=generator,
        ) 
        meteo[b, :, :-4] = meteo[b, perm, :-4].clone()

    batch["dynamic"][1] = meteo
    return batch


def delta_t_shuffle(
    batch,
    context_length: int,
    seed: int = 0,
    mode: str = "random",
    delta_t_channel: int = 5,
    **kwargs,
):
    """Corrupt the relative-time feature across context and target frames.

    Conditioning order from ``replicate_dataset.py`` is
    [sin_doy, cos_doy, sin_month, cos_month, year_trend, delta_t,
     temperature, humidity, precipitation, solar_irradiance].

    ``mode='zero'`` removes relative-time information. ``mode='random'``
    samples independently and uniformly between the minimum and maximum
    encoded delta-t values of each sample, preserving its valid encoded range.
    """
    del context_length, kwargs
    if mode not in {"zero", "random"}:
        raise ValueError(f"Unknown delta-t corruption mode: {mode!r}")

    batch = clone_batch(batch)
    if "dynamic" not in batch or len(batch["dynamic"]) < 2:
        raise RuntimeError(
            "Cannot corrupt delta-t: batch['dynamic'][1] is missing."
        )

    meteo = batch["dynamic"][1]
    if not torch.is_tensor(meteo) or meteo.ndim != 3:
        raise RuntimeError(
            "Expected temporal conditioning [B,T,C], got "
            f"{getattr(meteo, 'shape', None)}."
        )
    if not 0 <= delta_t_channel < meteo.shape[-1]:
        raise RuntimeError(
            f"Delta-t channel {delta_t_channel} is unavailable in conditioning "
            f"with {meteo.shape[-1]} channels."
        )

    delta_t = meteo[:, :, delta_t_channel]
    if mode == "zero":
        delta_t.zero_()
    else:
        for b in range(delta_t.shape[0]):
            generator = _device_generator(meteo.device, seed + b)
            lower = delta_t[b].min()
            upper = delta_t[b].max()
            if torch.isclose(lower, upper):
                delta_t[b].fill_(lower)
            else:
                random_values = torch.rand(
                    delta_t[b].shape,
                    device=meteo.device,
                    dtype=meteo.dtype,
                    generator=generator,
                )
                delta_t[b] = random_values * (upper - lower) + lower

    meteo[:, :, delta_t_channel] = delta_t
    batch["dynamic"][1] = meteo
    return batch


def zero_delta_t(batch, context_length: int, **kwargs):
    """Ablation wrapper that removes the relative-time feature."""
    return delta_t_shuffle(
        batch=batch,
        context_length=context_length,
        mode="zero",
        **kwargs,
    )


def random_delta_t(batch, context_length: int, **kwargs):
    """Ablation wrapper that replaces relative time with in-range noise."""
    return delta_t_shuffle(
        batch=batch,
        context_length=context_length,
        mode="random",
        **kwargs,
    )


def shuffle_context_all(
    batch,
    context_length: int,
    seed: int = 0,
    **kwargs,
):
    """Shuffle image, weather, and mask context frames together."""
    batch = clone_batch(batch)
    dynamic = batch["dynamic"][0]
    meteo = batch["dynamic"][1] if len(batch["dynamic"]) > 1 else None

    for b in range(dynamic.shape[0]):
        generator = _device_generator(dynamic.device, seed + b)
        perm = torch.randperm(
            context_length,
            device=dynamic.device,
            generator=generator,
        )
        dynamic[b, :context_length] = dynamic[b, perm].clone()

        if meteo is not None:
            meteo[b, :context_length] = meteo[b, perm].clone()

        if "dynamic_mask" in batch and len(batch["dynamic_mask"]) > 0:
            mask = batch["dynamic_mask"][0]
            mask[b, :context_length] = mask[b, perm].clone()
            batch["dynamic_mask"][0] = mask

    batch["dynamic"][0] = dynamic
    if meteo is not None:
        batch["dynamic"][1] = meteo
    return batch


def get_ablation_variants(include_strong_all_shuffle: bool = False) -> Dict[str, Callable]:
    variants = {
        "normal": identity_batch,
        # "shuffle_ndvi_only": shuffle_ndvi_only,
        # "reverse_ndvi_only": reverse_ndvi_only,
        # "repeat_last_ndvi": repeat_last_ndvi,
        # "noise_ndvi": noise_ndvi,
        # "zero_ndvi": zero_ndvi,
        "shuffle_weather": shuffle_weather_context,
        "shuffle_date": shuffle_date_context,
        # "zero_delta_t": zero_delta_t,
        "random_delta_t": random_delta_t,
    }
    if include_strong_all_shuffle:
        variants["shuffle_context_all"] = shuffle_context_all
    return variants


# ---------------------------------------------------------------------
# NDVI extraction and persistence reference
# ---------------------------------------------------------------------


def _get_dynamic_mask_5d(batch):
    if "dynamic_mask" not in batch or len(batch["dynamic_mask"]) == 0:
        return None
    dm = batch["dynamic_mask"][0]
    if dm.ndim == 4:
        dm = dm.unsqueeze(2)
    if dm.ndim != 5:
        raise RuntimeError(
            f"Expected dynamic mask [B,T,C,H,W] or [B,T,H,W], got {dm.shape}."
        )
    return dm


def _validate_evaluation_window(
    dynamic: torch.Tensor,
    context_length: int,
    target_length: int,
    ndvi_channel: int,
):
    """Validate the shared context and target window used by evaluation."""
    if dynamic.ndim != 5:
        raise RuntimeError(
            f"Expected dynamic input [B,T,C,H,W], got {dynamic.shape}."
        )
    if context_length < 1 or target_length < 1:
        raise ValueError(
            "context_length and target_length must both be at least 1, got "
            f"{context_length} and {target_length}."
        )
    if not 0 <= ndvi_channel < dynamic.shape[2]:
        raise RuntimeError(
            f"NDVI channel {ndvi_channel} is unavailable in dynamic input with "
            f"{dynamic.shape[2]} channels."
        )

    required_length = context_length + target_length
    if dynamic.shape[1] < required_length:
        raise RuntimeError(
            f"Evaluation needs {required_length} frames "
            f"({context_length} context + {target_length} target), but the "
            f"batch contains only {dynamic.shape[1]}."
        )


def _validate_dynamic_mask_shape(
    dynamic_mask: Optional[torch.Tensor],
    dynamic: torch.Tensor,
    required_length: int,
):
    if dynamic_mask is None:
        return
    if dynamic_mask.shape[0] != dynamic.shape[0]:
        raise RuntimeError(
            "Dynamic-mask batch dimension does not match the dynamic input: "
            f"{dynamic_mask.shape[0]} versus {dynamic.shape[0]}."
        )
    if dynamic_mask.shape[1] < required_length:
        raise RuntimeError(
            f"Dynamic mask contains {dynamic_mask.shape[1]} frames, but "
            f"{required_length} are required."
        )
    if dynamic_mask.shape[2] < 1:
        raise RuntimeError("Dynamic mask must contain at least one channel.")
    if dynamic_mask.shape[-2:] != dynamic.shape[-2:]:
        raise RuntimeError(
            "Dynamic-mask spatial shape does not match the dynamic input: "
            f"{dynamic_mask.shape[-2:]} versus {dynamic.shape[-2:]}."
        )


def _landcover_mask_like(
    batch,
    target: torch.Tensor,
    lc_min: Optional[float],
    lc_max: Optional[float],
    use_lc_mask: bool,
):
    """Return a land-cover mask explicitly broadcast to [B,T,1,H,W]."""
    if (
        not use_lc_mask
        or lc_min is None
        or lc_max is None
        or "landcover" not in batch
    ):
        return torch.ones_like(target)

    lc_mask = (
        (batch["landcover"] >= lc_min) & (batch["landcover"] <= lc_max)
    ).to(target.dtype)

    if lc_mask.ndim == 3:  # [B,H,W]
        lc_mask = lc_mask[:, None, None, :, :]
    elif lc_mask.ndim == 4:  # [B,1,H,W]
        if lc_mask.shape[1] != 1:
            raise RuntimeError(
                "Expected landcover [B,1,H,W], got "
                f"{batch['landcover'].shape}."
            )
        lc_mask = lc_mask.unsqueeze(1)
    elif lc_mask.ndim != 5:
        raise RuntimeError(
            "Expected landcover [B,H,W], [B,1,H,W], or a broadcastable "
            f"[B,T,1,H,W] tensor, got {batch['landcover'].shape}."
        )

    expected = target.shape
    if any(
        actual not in (1, wanted)
        for actual, wanted in zip(lc_mask.shape, expected)
    ):
        raise RuntimeError(
            f"Landcover shape {lc_mask.shape} cannot broadcast to target "
            f"shape {expected}."
        )
    return lc_mask.expand_as(target)


def _extract_last_valid_context_ndvi(
    dynamic: torch.Tensor,
    dynamic_mask: Optional[torch.Tensor],
    context_length: int,
    ndvi_targ_channel: int,
):
    context_ndvi = dynamic[
        :,
        :context_length,
        ndvi_targ_channel : ndvi_targ_channel + 1,
    ]

    if dynamic_mask is None:
        last_ndvi = context_ndvi[:, -1:]
        valid = torch.ones_like(last_ndvi, dtype=torch.bool)
        return last_ndvi, valid

    context_valid = dynamic_mask[
        :,
        :context_length,
        :1,
    ] < 1.0

    reverse_valid = context_valid.flip(1)
    has_valid = reverse_valid.any(dim=1, keepdim=True)
    reverse_idx = reverse_valid.to(torch.int64).argmax(dim=1, keepdim=True)
    last_valid_idx = context_length - 1 - reverse_idx

    last_ndvi = torch.gather(
        context_ndvi,
        dim=1,
        index=last_valid_idx,
    )

    return last_ndvi, has_valid


def extract_prediction_target_mask(
    preds,
    batch,
    context_length: int,
    target_length: int,
    lc_min: Optional[float],
    lc_max: Optional[float],
    ndvi_pred_channel: int = 0,
    ndvi_targ_channel: int = 0,
    use_cloud_mask: bool = True,
    use_lc_mask: bool = True,
    use_last_valid_context: bool = False,
):
    """
    Return NDVI-only tensors.

    Shapes:
        ndvi_pred:       [B,T,1,H,W]
        ndvi_targ:       [B,T,1,H,W]
        persistence:     [B,T,1,H,W]
        target_mask:     [B,T,1,H,W]
        persistence_mask:[B,T,1,H,W]
    """
    # Match SpatioTemporalTask/test.py: dynamic[2] is the dedicated NDVI
    # target, while dynamic[0] contains the configurable model inputs.
    dynamic = batch["dynamic"][2]
    _validate_evaluation_window(
        dynamic=dynamic,
        context_length=context_length,
        target_length=target_length,
        ndvi_channel=ndvi_targ_channel,
    )

    ndvi_targ = dynamic[
        :,
        context_length : context_length + target_length,
        ndvi_targ_channel : ndvi_targ_channel + 1,
    ]

    if preds.ndim == 4:
        ndvi_pred = preds[:, -ndvi_targ.shape[1] :].unsqueeze(2)
    elif preds.ndim == 5:
        ndvi_pred = preds[
            :,
            -ndvi_targ.shape[1] :,
            ndvi_pred_channel : ndvi_pred_channel + 1,
        ]
    else:
        raise RuntimeError(
            f"Expected predictions [B,T,H,W] or [B,T,C,H,W], got {preds.shape}."
        )

    if ndvi_pred.shape != ndvi_targ.shape:
        raise RuntimeError(
            f"Prediction shape {ndvi_pred.shape} does not match target {ndvi_targ.shape}."
        )

    dynamic_mask = _get_dynamic_mask_5d(batch)
    _validate_dynamic_mask_shape(
        dynamic_mask=dynamic_mask,
        dynamic=dynamic,
        required_length=context_length + target_length,
    )

    if use_last_valid_context:
        last_ndvi, persistence_valid = _extract_last_valid_context_ndvi(
            dynamic=dynamic,
            dynamic_mask=dynamic_mask,
            context_length=context_length,
            ndvi_targ_channel=ndvi_targ_channel,
        )
    else:
        last_ndvi = dynamic[
            :,
            context_length - 1 : context_length,
            ndvi_targ_channel : ndvi_targ_channel + 1,
        ]
        if dynamic_mask is not None:
            persistence_valid = dynamic_mask[
                :,
                context_length - 1 : context_length,
                :1,
            ] < 1.0
        else:
            persistence_valid = torch.ones_like(last_ndvi, dtype=torch.bool)

    persistence = last_ndvi.expand(
        -1,
        ndvi_targ.shape[1],
        -1,
        -1,
        -1,
    )
    persistence_valid = persistence_valid.expand_as(ndvi_targ)

    if use_cloud_mask and dynamic_mask is not None:
        cloud_mask = (
            dynamic_mask[
                :,
                context_length : context_length + ndvi_targ.shape[1],
                :1,
            ]
            < 1.0
        ).to(ndvi_pred.dtype)
    else:
        cloud_mask = torch.ones_like(ndvi_targ)

    lc_mask = _landcover_mask_like(
        batch=batch,
        target=ndvi_targ,
        lc_min=lc_min,
        lc_max=lc_max,
        use_lc_mask=use_lc_mask,
    )

    target_mask = cloud_mask * lc_mask
    persistence_mask = target_mask * persistence_valid.to(ndvi_pred.dtype)

    return ndvi_pred, ndvi_targ, persistence, target_mask, persistence_mask


# ---------------------------------------------------------------------
# Standard and persistence-aware metrics
# ---------------------------------------------------------------------


def compute_common_metrics_batch(
    preds,
    batch,
    context_length: int,
    target_length: int,
    lc_min: Optional[float],
    lc_max: Optional[float],
    ndvi_pred_channel: int = 0,
    ndvi_targ_channel: int = 0,
    use_cloud_mask: bool = True,
    use_lc_mask: bool = True,
    use_last_valid_context: bool = True,
    change_threshold: float = 0.03,
    spatial_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
):
    """
    Compute NDVI metrics.

    Overall outputs have shape [B]. Per-horizon outputs have shape [B,T].
    Persistence-aware metrics use the same valid support for model and baseline.
    """
    (
        ndvi_pred,
        ndvi_targ,
        persistence,
        mask,
        persistence_mask,
    ) = extract_prediction_target_mask(
        preds=preds,
        batch=batch,
        context_length=context_length,
        target_length=target_length,
        lc_min=lc_min,
        lc_max=lc_max,
        ndvi_pred_channel=ndvi_pred_channel,
        ndvi_targ_channel=ndvi_targ_channel,
        use_cloud_mask=use_cloud_mask,
        use_lc_mask=use_lc_mask,
        use_last_valid_context=use_last_valid_context,
    )
    full_mask = mask

    if spatial_mask is not None:
        spatial_mask = torch.as_tensor(
            spatial_mask,
            device=mask.device,
            dtype=mask.dtype,
        )
        if spatial_mask.ndim == 2:
            spatial_mask = spatial_mask[None, None, None, :, :]
        try:
            spatial_mask = spatial_mask.expand_as(mask)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Field mask shape {tuple(spatial_mask.shape)} cannot broadcast "
                f"to evaluation mask shape {tuple(mask.shape)}."
            ) from exc
        mask = mask * spatial_mask
        persistence_mask = persistence_mask * spatial_mask

    model_error = ndvi_pred - ndvi_targ
    persistence_error = persistence - ndvi_targ
    model_abs_error = model_error.abs()
    persistence_abs_error = persistence_error.abs()

    true_change = ndvi_targ - persistence
    predicted_change = ndvi_pred - persistence

    changed_mask = (
        persistence_mask
        * (true_change.abs() > change_threshold).to(persistence_mask.dtype)
    )

    sample_dims = (1, 2, 3, 4)
    horizon_dims = (2, 3, 4)

    nan_sample = torch.full(
        (ndvi_pred.shape[0],),
        float("nan"),
        device=ndvi_pred.device,
        dtype=ndvi_pred.dtype,
    )
    nan_horizon = torch.full(
        (ndvi_pred.shape[0], ndvi_pred.shape[1]),
        float("nan"),
        device=ndvi_pred.device,
        dtype=ndvi_pred.dtype,
    )

    # Standard metrics on all valid target pixels.
    n_obs = mask.sum(dim=sample_dims)
    sse = (model_error.pow(2) * mask).sum(dim=sample_dims)
    sae = (model_abs_error * mask).sum(dim=sample_dims)
    bias_sum = (model_error * mask).sum(dim=sample_dims)
    pred_sum = (ndvi_pred * mask).sum(dim=sample_dims)
    target_sum = (ndvi_targ * mask).sum(dim=sample_dims)
    pred_square_sum = (ndvi_pred.pow(2) * mask).sum(dim=sample_dims)
    target_square_sum = (ndvi_targ.pow(2) * mask).sum(dim=sample_dims)
    pred_target_sum = (ndvi_pred * ndvi_targ * mask).sum(dim=sample_dims)

    # Squared Pearson correlation on the same valid support as RMSE. Computing
    # it from additive moments also permits a true pooled dataset-level value.
    covariance_numerator = n_obs * pred_target_sum - pred_sum * target_sum
    pred_variance_numerator = (
        n_obs * pred_square_sum - pred_sum.pow(2)
    ).clamp_min(0.0)
    target_variance_numerator = (
        n_obs * target_square_sum - target_sum.pow(2)
    ).clamp_min(0.0)
    correlation_denominator = pred_variance_numerator * target_variance_numerator
    pearson_r2 = torch.where(
        (n_obs >= 2) & (correlation_denominator > eps),
        covariance_numerator.pow(2) / correlation_denominator.clamp_min(eps),
        nan_sample,
    ).clamp(max=1.0)

    rmse = torch.where(
        n_obs > 0,
        torch.sqrt(sse / n_obs.clamp_min(1.0)),
        nan_sample,
    )
    mae = torch.where(
        n_obs > 0,
        sae / n_obs.clamp_min(1.0),
        nan_sample,
    )
    # Signed mean error: positive means overprediction, negative means
    # underprediction.
    bias = torch.where(
        n_obs > 0,
        bias_sum / n_obs.clamp_min(1.0),
        nan_sample,
    )

    # Persistence comparison on pixels with a valid persistence reference.
    n_persistence_obs = persistence_mask.sum(dim=sample_dims)
    skill_sse = (model_error.pow(2) * persistence_mask).sum(dim=sample_dims)
    persistence_sse = (
        persistence_error.pow(2) * persistence_mask
    ).sum(dim=sample_dims)

    persistence_rmse = torch.where(
        n_persistence_obs > 0,
        torch.sqrt(persistence_sse / n_persistence_obs.clamp_min(1.0)),
        nan_sample,
    )
    pss_mse = torch.where(
        persistence_sse > eps,
        1.0 - skill_sse / persistence_sse.clamp_min(eps),
        nan_sample,
    )

    # Changed-pixel metrics.
    n_changed = changed_mask.sum(dim=sample_dims)
    future_closer_count = (
        (model_abs_error < persistence_abs_error).to(mask.dtype) * changed_mask
    ).sum(dim=sample_dims)
    future_closer_rate = torch.where(
        n_changed > 0,
        future_closer_count / n_changed.clamp_min(1.0),
        nan_sample,
    )

    # Per-horizon additive statistics.
    n_obs_h = mask.sum(dim=horizon_dims)
    sse_h = (model_error.pow(2) * mask).sum(dim=horizon_dims)
    sae_h = (model_abs_error * mask).sum(dim=horizon_dims)
    bias_sum_h = (model_error * mask).sum(dim=horizon_dims)
    pred_sum_h = (ndvi_pred * mask).sum(dim=horizon_dims)
    target_sum_h = (ndvi_targ * mask).sum(dim=horizon_dims)
    pred_square_sum_h = (ndvi_pred.pow(2) * mask).sum(dim=horizon_dims)
    target_square_sum_h = (ndvi_targ.pow(2) * mask).sum(dim=horizon_dims)
    pred_target_sum_h = (
        ndvi_pred * ndvi_targ * mask
    ).sum(dim=horizon_dims)

    covariance_numerator_h = (
        n_obs_h * pred_target_sum_h - pred_sum_h * target_sum_h
    )
    pred_variance_numerator_h = (
        n_obs_h * pred_square_sum_h - pred_sum_h.pow(2)
    ).clamp_min(0.0)
    target_variance_numerator_h = (
        n_obs_h * target_square_sum_h - target_sum_h.pow(2)
    ).clamp_min(0.0)
    correlation_denominator_h = (
        pred_variance_numerator_h * target_variance_numerator_h
    )
    pearson_r2_h = torch.where(
        (n_obs_h >= 2) & (correlation_denominator_h > eps),
        covariance_numerator_h.pow(2) / correlation_denominator_h.clamp_min(eps),
        nan_horizon,
    ).clamp(max=1.0)

    n_persistence_obs_h = persistence_mask.sum(dim=horizon_dims)
    skill_sse_h = (
        model_error.pow(2) * persistence_mask
    ).sum(dim=horizon_dims)
    persistence_sse_h = (
        persistence_error.pow(2) * persistence_mask
    ).sum(dim=horizon_dims)

    n_changed_h = changed_mask.sum(dim=horizon_dims)
    future_closer_count_h = (
        (model_abs_error < persistence_abs_error).to(mask.dtype) * changed_mask
    ).sum(dim=horizon_dims)
    rmse_h = torch.where(
        n_obs_h > 0,
        torch.sqrt(sse_h / n_obs_h.clamp_min(1.0)),
        nan_horizon,
    )
    mae_h = torch.where(
        n_obs_h > 0,
        sae_h / n_obs_h.clamp_min(1.0),
        nan_horizon,
    )
    bias_h = torch.where(
        n_obs_h > 0,
        bias_sum_h / n_obs_h.clamp_min(1.0),
        nan_horizon,
    )
    persistence_rmse_h = torch.where(
        n_persistence_obs_h > 0,
        torch.sqrt(
            persistence_sse_h / n_persistence_obs_h.clamp_min(1.0)
        ),
        nan_horizon,
    )
    pss_mse_h = torch.where(
        persistence_sse_h > eps,
        1.0 - skill_sse_h / persistence_sse_h.clamp_min(eps),
        nan_horizon,
    )
    future_closer_rate_h = torch.where(
        n_changed_h > 0,
        future_closer_count_h / n_changed_h.clamp_min(1.0),
        nan_horizon,
    )
    return {
        # Overall per-sample metrics [B]
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "pearson_r2": pearson_r2,
        "persistence_rmse": persistence_rmse,
        "pss_mse": pss_mse,
        "future_closer_rate": future_closer_rate,
        # Overall additive statistics [B]
        "sse": sse,
        "sae": sae,
        "bias_sum": bias_sum,
        "pred_sum": pred_sum,
        "target_sum": target_sum,
        "pred_square_sum": pred_square_sum,
        "target_square_sum": target_square_sum,
        "pred_target_sum": pred_target_sum,
        "n_obs": n_obs,
        "skill_sse": skill_sse,
        "persistence_sse": persistence_sse,
        "n_persistence_obs": n_persistence_obs,
        "future_closer_count": future_closer_count,
        "n_changed": n_changed,
        # Per-horizon metrics [B,T]
        "rmse_by_horizon": rmse_h,
        "mae_by_horizon": mae_h,
        "bias_by_horizon": bias_h,
        "pearson_r2_by_horizon": pearson_r2_h,
        "persistence_rmse_by_horizon": persistence_rmse_h,
        "pss_mse_by_horizon": pss_mse_h,
        "future_closer_rate_by_horizon": future_closer_rate_h,
        # Per-horizon additive statistics [B,T]
        "sse_by_horizon": sse_h,
        "sae_by_horizon": sae_h,
        "bias_sum_by_horizon": bias_sum_h,
        "pred_sum_by_horizon": pred_sum_h,
        "target_sum_by_horizon": target_sum_h,
        "pred_square_sum_by_horizon": pred_square_sum_h,
        "target_square_sum_by_horizon": target_square_sum_h,
        "pred_target_sum_by_horizon": pred_target_sum_h,
        "n_obs_by_horizon": n_obs_h,
        "skill_sse_by_horizon": skill_sse_h,
        "persistence_sse_by_horizon": persistence_sse_h,
        "n_persistence_obs_by_horizon": n_persistence_obs_h,
        "future_closer_count_by_horizon": future_closer_count_h,
        "n_changed_by_horizon": n_changed_h,
        # Tensors for visualizations and TIFF output
        "ndvi_pred": ndvi_pred,
        "ndvi_targ": ndvi_targ,
        "persistence": persistence,
        "full_mask": full_mask,
        "mask": mask,
        "persistence_mask": persistence_mask,
        "changed_mask": changed_mask,
        "true_change": true_change,
        "predicted_change": predicted_change,
    }


# ---------------------------------------------------------------------
# TIFF saving
# ---------------------------------------------------------------------


def save_sample_tiffs(
    output_dir: Path,
    variant: str,
    sample_name: str,
    pred_t_hw: torch.Tensor,
    targ_t_hw: torch.Tensor,
    mask_t_hw: torch.Tensor,
    persistence_t_hw: Optional[torch.Tensor] = None,
    save_target: bool = True,
):
    pred_dir = output_dir / "predictions_tiff" / variant
    target_dir = output_dir / "targets_tiff"
    mask_dir = output_dir / "valid_masks_tiff"
    persistence_dir = output_dir / "persistence_tiff"

    pred_path = pred_dir / f"{sample_name}_pred.tif"
    save_tiff_stack(pred_path, pred_t_hw.detach().float().cpu().numpy())

    target_path = None
    mask_path = None
    persistence_path = None

    if save_target:
        target_path = target_dir / f"{sample_name}_target.tif"
        mask_path = mask_dir / f"{sample_name}_valid_mask.tif"
        save_tiff_stack(target_path, targ_t_hw.detach().float().cpu().numpy())
        save_tiff_stack(mask_path, mask_t_hw.detach().float().cpu().numpy())

        if persistence_t_hw is not None:
            persistence_path = persistence_dir / f"{sample_name}_persistence.tif"
            save_tiff_stack(
                persistence_path,
                persistence_t_hw.detach().float().cpu().numpy(),
            )

    return pred_path, target_path, mask_path, persistence_path


# ---------------------------------------------------------------------
# Time metadata
# ---------------------------------------------------------------------


def get_sample_times(batch, sample_idx: int, batch_size: int, seq_len: int):
    """Handle [B][T] and default-collated [T][B] time structures."""
    if "time" not in batch or batch["time"] is None:
        return None

    times = list(batch["time"]) if isinstance(batch["time"], tuple) else batch["time"]
    if not isinstance(times, list) or len(times) == 0:
        return None

    first = times[0]
    if not isinstance(first, (list, tuple)):
        return list(times)

    # Prefer PyTorch default-collated [T][B] when dimensions are ambiguous.
    if len(times) == seq_len and len(first) == batch_size:
        return [times[t][sample_idx] for t in range(seq_len)]
    if len(times) == batch_size and len(times[sample_idx]) == seq_len:
        return list(times[sample_idx])
    return None


def get_target_time_metadata(
    batch,
    sample_idx: int,
    context_length: int,
    target_len: int,
    batch_size: int,
    global_idx: int,
):
    seq_len = context_length + target_len
    sample_times = get_sample_times(
        batch=batch,
        sample_idx=sample_idx,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    rows = []
    if sample_times is None or len(sample_times) < seq_len:
        for h in range(target_len):
            rows.append(
                {
                    "target_time": f"idx_{global_idx:06d}_t{h:02d}",
                    "lead_days": float("nan"),
                }
            )
        return rows

    issue_time = sample_times[context_length - 1]
    try:
        issue_dt = pd.to_datetime(issue_time)
    except Exception:
        issue_dt = None

    for h in range(target_len):
        target_time = sample_times[context_length + h]
        lead_days = float("nan")
        if issue_dt is not None:
            try:
                target_dt = pd.to_datetime(target_time)
                lead_days = (target_dt - issue_dt).total_seconds() / 86400.0
            except Exception:
                pass
        rows.append({"target_time": str(target_time), "lead_days": lead_days})
    return rows


def get_target_time_keys(
    batch,
    sample_idx: int,
    context_length: int,
    target_len: int,
    batch_size: int,
    global_idx: int,
):
    metadata = get_target_time_metadata(
        batch=batch,
        sample_idx=sample_idx,
        context_length=context_length,
        target_len=target_len,
        batch_size=batch_size,
        global_idx=global_idx,
    )
    return [item["target_time"] for item in metadata]


# ---------------------------------------------------------------------
# Full-test-set temporal diagnostics with fixed global patches
# ---------------------------------------------------------------------


PATCH_COLORS = ["#fa8500", "#1f77b4", "#e377c2"]

CMAP_NDVI = clr.LinearSegmentedColormap.from_list(
    "ndvi",
    [
        "#cbbe9a",
        "#fffde4",
        "#bccea5",
        "#66985b",
        "#2e6a32",
        "#123f1e",
        "#0e371a",
        "#01140f",
        "#000d0a",
    ],
    N=256,
)
CMAP_NDVI.set_bad(color="white")


def extract_target_and_mask_from_batch(
    batch,
    context_length: int,
    target_length: int,
    lc_min: Optional[float],
    lc_max: Optional[float],
    ndvi_targ_channel: int = 0,
    use_cloud_mask: bool = True,
    use_lc_mask: bool = True,
):
    dynamic = batch["dynamic"][2]
    _validate_evaluation_window(
        dynamic=dynamic,
        context_length=context_length,
        target_length=target_length,
        ndvi_channel=ndvi_targ_channel,
    )
    ndvi_targ = dynamic[
        :,
        context_length : context_length + target_length,
        ndvi_targ_channel : ndvi_targ_channel + 1,
    ]

    dm = _get_dynamic_mask_5d(batch)
    _validate_dynamic_mask_shape(
        dynamic_mask=dm,
        dynamic=dynamic,
        required_length=context_length + target_length,
    )
    if use_cloud_mask and dm is not None:
        cloud_mask = (
            dm[
                :,
                context_length : context_length + target_length,
                :1,
            ]
            < 1.0
        ).to(ndvi_targ.dtype)
    else:
        cloud_mask = torch.ones_like(ndvi_targ)

    lc_mask = _landcover_mask_like(
        batch=batch,
        target=ndvi_targ,
        lc_min=lc_min,
        lc_max=lc_max,
        use_lc_mask=use_lc_mask,
    )

    return ndvi_targ, cloud_mask * lc_mask


def choose_global_patches_from_loader(
    dataloader,
    device,
    context_length: int,
    target_length: int,
    lc_min: Optional[float],
    lc_max: Optional[float],
    ndvi_targ_channel: int = 0,
    patch_size: int = 5,
    n_patches: int = 3,
    min_valid_frac: float = 0.7,
    seed: int = 0,
    use_cloud_mask: bool = True,
    use_lc_mask: bool = True,
    max_batches: Optional[int] = None,
):
    valid_count_hw = None
    target_sum_hw = None
    target_count_hw = None
    n_target_observations = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch_to_device(batch, device)
        ndvi_targ, mask = extract_target_and_mask_from_batch(
            batch=batch,
            context_length=context_length,
            target_length=target_length,
            lc_min=lc_min,
            lc_max=lc_max,
            ndvi_targ_channel=ndvi_targ_channel,
            use_cloud_mask=use_cloud_mask,
            use_lc_mask=use_lc_mask,
        )

        curr_valid = mask.sum(dim=(0, 1, 2)).detach().float().cpu()
        curr_sum = (ndvi_targ * mask).sum(dim=(0, 1, 2)).detach().float().cpu()
        curr_count = mask.sum(dim=(0, 1, 2)).detach().float().cpu()

        if valid_count_hw is None:
            valid_count_hw = curr_valid
            target_sum_hw = curr_sum
            target_count_hw = curr_count
        else:
            valid_count_hw += curr_valid
            target_sum_hw += curr_sum
            target_count_hw += curr_count

        n_target_observations += int(mask.shape[0] * mask.shape[1])

    if valid_count_hw is None:
        raise RuntimeError("No batches found while selecting global patches.")

    valid_fraction_hw = valid_count_hw.numpy() / max(
        float(n_target_observations), 1.0
    )
    height, width = valid_fraction_hw.shape
    candidates = []

    for top in range(0, height - patch_size + 1):
        for left in range(0, width - patch_size + 1):
            patch_frac = valid_fraction_hw[
                top : top + patch_size,
                left : left + patch_size,
            ].mean()
            if patch_frac >= min_valid_frac:
                candidates.append((top, left, float(patch_frac)))

    if not candidates:
        for top in range(0, height - patch_size + 1):
            for left in range(0, width - patch_size + 1):
                patch_frac = valid_fraction_hw[
                    top : top + patch_size,
                    left : left + patch_size,
                ].mean()
                candidates.append((top, left, float(patch_frac)))
        candidates.sort(key=lambda x: x[2], reverse=True)

    rng = np.random.default_rng(seed)
    if len(candidates) <= n_patches:
        chosen = candidates
    else:
        idx = rng.choice(len(candidates), size=n_patches, replace=False)
        chosen = [candidates[i] for i in idx]

    patches = [
        {
            "patch_id": i + 1,
            "top": int(top),
            "left": int(left),
            "patch_size": int(patch_size),
            "valid_fraction": float(valid_frac),
            "color": PATCH_COLORS[i % len(PATCH_COLORS)],
        }
        for i, (top, left, valid_frac) in enumerate(chosen)
    ]

    mean_target_hw = (
        target_sum_hw / target_count_hw.clamp_min(1.0)
    ).numpy()
    return patches, mean_target_hw, valid_fraction_hw


def _new_stats_dict():
    return {
        "target_sum": 0.0,
        "target_sumsq": 0.0,
        "target_n": 0.0,
        "pred_sum": 0.0,
        "pred_sumsq": 0.0,
        "pred_n": 0.0,
    }


def update_series_stats(
    series_acc,
    variant: str,
    scope: str,
    time_key: str,
    target_values: torch.Tensor,
    pred_values: torch.Tensor,
):
    if target_values.numel() == 0:
        return

    target_values = target_values.detach().float().cpu()
    pred_values = pred_values.detach().float().cpu()
    stats = series_acc[variant][scope][str(time_key)]

    stats["target_sum"] += float(target_values.sum())
    stats["target_sumsq"] += float((target_values**2).sum())
    stats["target_n"] += float(target_values.numel())
    stats["pred_sum"] += float(pred_values.sum())
    stats["pred_sumsq"] += float((pred_values**2).sum())
    stats["pred_n"] += float(pred_values.numel())


def finalize_series_accumulator(series_acc):
    rows = []
    for variant, scopes in series_acc.items():
        for scope, time_dict in scopes.items():
            for time_key, stats in time_dict.items():
                target_n = max(stats["target_n"], 1.0)
                pred_n = max(stats["pred_n"], 1.0)
                target_mean = stats["target_sum"] / target_n
                pred_mean = stats["pred_sum"] / pred_n
                target_var = max(
                    stats["target_sumsq"] / target_n - target_mean**2,
                    0.0,
                )
                pred_var = max(
                    stats["pred_sumsq"] / pred_n - pred_mean**2,
                    0.0,
                )
                rows.append(
                    {
                        "variant": variant,
                        "scope": scope,
                        "time": time_key,
                        "target_mean": target_mean,
                        "target_std": math.sqrt(target_var),
                        "target_n": stats["target_n"],
                        "pred_mean": pred_mean,
                        "pred_std": math.sqrt(pred_var),
                        "pred_n": stats["pred_n"],
                    }
                )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["variant", "scope", "time"]).reset_index(drop=True)
    return df


def colorize_ndvi_hw(
    ndvi_hw: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: bool = True,
):
    arr = np.asarray(ndvi_hw, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        arr = np.zeros_like(arr)
        valid = np.ones_like(arr, dtype=bool)

    if vmin is None or vmax is None:
        vals = arr[valid]
        if robust:
            vmin_auto = np.percentile(vals, 2)
            vmax_auto = np.percentile(vals, 98)
        else:
            vmin_auto = vals.min()
            vmax_auto = vals.max()
        if vmin is None:
            vmin = float(vmin_auto)
        if vmax is None:
            vmax = float(vmax_auto)

    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-8

    arr01 = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = CMAP_NDVI(arr01)[..., :3]
    rgb[~valid] = 1.0
    return rgb


def load_field_evaluation_mask(mask_path, target_hw) -> np.ndarray:
    """Load a categorical field mask and resize it with nearest-neighbor sampling."""
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Field evaluation mask not found: {mask_path}")

    target_height, target_width = map(int, target_hw)
    with Image.open(mask_path) as image:
        mask_image = image.convert("L")
        source_width, source_height = mask_image.size
        if (source_height, source_width) != (target_height, target_width):
            mask_image = mask_image.resize(
                (target_width, target_height),
                resample=Image.Resampling.NEAREST,
            )
            print(
                "Resized categorical field mask with nearest-neighbor sampling: "
                f"{source_height}x{source_width} -> {target_height}x{target_width}."
            )
        else:
            print(
                f"Field mask already matches evaluation shape: "
                f"{target_height}x{target_width}."
            )

        field_mask = np.asarray(mask_image, dtype=np.uint8) > 0

    if not field_mask.any():
        raise ValueError(f"Field evaluation mask contains no selected pixels: {mask_path}")
    return field_mask


def save_field_mask_overlay_image(
    mean_target_hw: np.ndarray,
    field_mask_hw: np.ndarray,
    save_path,
    title: str = "Field evaluation mask",
    color=(0.1, 0.1, 1.0),
    alpha: float = 0.35,
):
    """Overlay the selected field pixels on the mean target NDVI image."""
    rgb = colorize_ndvi_hw(mean_target_hw, robust=True)
    field_mask_hw = np.asarray(field_mask_hw, dtype=bool)
    if field_mask_hw.shape != rgb.shape[:2]:
        raise ValueError(
            f"Field mask shape {field_mask_hw.shape} does not match reference "
            f"image shape {rgb.shape[:2]}."
        )

    overlay = np.zeros((*field_mask_hw.shape, 4), dtype=np.float32)
    overlay[field_mask_hw, :3] = color
    overlay[field_mask_hw, 3] = alpha

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(rgb)
    ax.imshow(overlay)
    ax.contour(field_mask_hw.astype(np.uint8), levels=[0.5], colors=[color], linewidths=1.5)
    ax.set_title(title)
    ax.axis("off")
    handle = Rectangle(
        (0, 0),
        1,
        1,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        label=f"Selected field ({int(field_mask_hw.sum())} pixels)",
    )
    ax.legend(
        handles=[handle],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_global_patch_overlay_image(
    mean_target_hw: np.ndarray,
    patches: list,
    save_path,
    title: str = "Global patch locations",
    alpha: float = 0.28,
):
    rgb = colorize_ndvi_hw(mean_target_hw, robust=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

    handles = []
    for patch in patches:
        color = patch["color"]
        top = patch["top"]
        left = patch["left"]
        patch_size = patch["patch_size"]
        rect = Rectangle(
            (left, top),
            patch_size,
            patch_size,
            linewidth=2.5,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(
            left + 0.5,
            max(top - 0.8, 0.5),
            f"P{patch['patch_id']}",
            color=color,
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.5,
            ),
        )
        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                label=f"Patch {patch['patch_id']}",
            )
        )

    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True,)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_full_dataset_series(
    series_df: pd.DataFrame,
    patches: list,
    output_dir,
    include_field_mask: bool = False,
    y_min: float = 0.0,
    y_max: float = 1.0,
    show_target_std: bool = True, # TODO SETTARE MANUALMENTE
    show_prediction_std: bool = False, # TODO SETTARE MANUALMENTE
):
    """
    Plot all degradation variants together, separately for each scope.

    Generated figures:
        global_all_variants_ndvi_timeseries.png
        patch_1_all_variants_ndvi_timeseries.png
        patch_2_all_variants_ndvi_timeseries.png
        ...

    Each figure contains:
        - target mean;
        - optional target standard deviation;
        - normal model prediction;
        - all degradation/ablation predictions;
        - optional prediction standard deviations.

    Expected series_df columns:
        variant
        scope
        time
        target_mean
        target_std
        pred_mean
        pred_std
    """

    output_dir = Path(output_dir)
    plot_dir = output_dir / "full_dataset_timeseries"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if series_df.empty:
        print("No time-series data available for plotting.")
        return

    required_columns = {
        "variant",
        "scope",
        "time",
        "target_mean",
        "target_std",
        "pred_mean",
        "pred_std",
    }

    missing_columns = required_columns - set(series_df.columns)

    if missing_columns:
        raise ValueError(
            "series_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 1. Order variants according to their interpretation
    # ---------------------------------------------------------
    preferred_order = [
        "normal",
        "shuffle_ndvi_only",
        "reverse_ndvi_only",
        "repeat_last_ndvi",
        "noise_ndvi",
        "zero_ndvi",
        "shuffle_weather",
        "shuffle_date",
        "zero_delta_t",
        "random_delta_t",
        "shuffle_context_all",
    ]

    available_variants = set(
        series_df["variant"].dropna().unique()
    )

    variants = [
        name
        for name in preferred_order
        if name in available_variants
    ]

    # Include any additional variants not explicitly listed.
    variants.extend(
        sorted(
            available_variants - set(variants)
        )
    )

    if not variants:
        print("No variants available for plotting.")
        return

    # ---------------------------------------------------------
    # 2. Reader-friendly legend labels
    # ---------------------------------------------------------
    variant_labels = {
        "normal": "Normal input",
        "shuffle_ndvi_only": "Shuffled NDVI order",
        "reverse_ndvi_only": "Reversed NDVI order",
        "repeat_last_ndvi": "Repeated last NDVI",
        "noise_ndvi": "NDVI replaced by noise",
        "zero_ndvi": "NDVI removed",
        "shuffle_weather": "Shuffled weather order",
        "shuffle_date": "Shuffled date/time order",
        "zero_delta_t": "Relative time removed",
        "random_delta_t": "Relative time randomized",
        "shuffle_context_all": "All context shuffled",
    }

    # ---------------------------------------------------------
    # 3. Spatial scopes: full image and selected patches
    # ---------------------------------------------------------
    scopes = ["global"]

    scopes.extend(
        f"patch_{patch['patch_id']}"
        for patch in patches
    )
    if include_field_mask:
        scopes.append("field_mask")

    for scope in scopes:
        scope_df = series_df[
            series_df["scope"] == scope
        ].copy()

        if scope_df.empty:
            print(
                f"Skipping scope '{scope}': no data available."
            )
            continue

        fig, ax = plt.subplots(
            figsize=(13, 5.5)
        )

        # -----------------------------------------------------
        # 4. Construct the title
        # -----------------------------------------------------
        if scope == "global":
            title = "Whole-image mean NDVI"
            output_name = (
                "global_all_variants_ndvi_timeseries.png"
            )

        elif scope.startswith("patch_"):
            patch_id = int(
                scope.split("_")[1]
            )

            matching_patches = [
                patch
                for patch in patches
                if patch["patch_id"] == patch_id
            ]

            if matching_patches:
                patch = matching_patches[0]

                title = (
                    f"Patch {patch_id}: mean NDVI "
                    f"(top={patch['top']}, "
                    f"left={patch['left']}, "
                    f"size={patch['patch_size']}×"
                    f"{patch['patch_size']})"
                )
            else:
                title = f"Patch {patch_id}: mean NDVI"

            output_name = (
                f"patch_{patch_id}_"
                "all_variants_ndvi_timeseries.png"
            )
        else:
            title = "Field-mask mean NDVI"
            output_name = "field_mask_all_variants_ndvi_timeseries.png"

        # -----------------------------------------------------
        # 5. Use the normal variant as the target reference
        # -----------------------------------------------------
        if "normal" in available_variants:
            reference_variant = "normal"
        else:
            reference_variant = variants[0]

        target_df = scope_df[
            scope_df["variant"] == reference_variant
        ].copy()

        if target_df.empty:
            print(
                f"Skipping scope '{scope}': "
                "target reference data are missing."
            )
            plt.close(fig)
            continue

        # Parse dates when possible.
        target_dates = pd.to_datetime(
            target_df["time"],
            errors="coerce",
        )

        if target_dates.notna().all():
            target_df["_plot_time"] = target_dates
            target_df = target_df.sort_values(
                "_plot_time"
            )
            target_x = target_df["_plot_time"]
            x_is_datetime = True
        else:
            target_df["_plot_time"] = (
                target_df["time"].astype(str)
            )
            target_df = target_df.sort_values(
                "_plot_time"
            )
            target_x = target_df["_plot_time"]
            x_is_datetime = False

        # -----------------------------------------------------
        # 6. Plot target once
        # -----------------------------------------------------
        ax.plot(
            target_x,
            target_df["target_mean"],
            linewidth=2.6,
            linestyle="-",
            color="black",
            label="Observed target",
            zorder=10,
        )

        if show_target_std:
            ax.fill_between(
                target_x,
                target_df["target_mean"]
                - target_df["target_std"],
                target_df["target_mean"]
                + target_df["target_std"],
                color="black",
                alpha=0.10,
                linewidth=0,
                label="Target ±1 SD",
                zorder=1,
            )

        # -----------------------------------------------------
        # 7. Plot every prediction variant
        # -----------------------------------------------------
        for variant in variants:
            variant_df = scope_df[
                scope_df["variant"] == variant
            ].copy()

            if variant_df.empty:
                continue

            variant_dates = pd.to_datetime(
                variant_df["time"],
                errors="coerce",
            )

            if (
                x_is_datetime
                and variant_dates.notna().all()
            ):
                variant_df["_plot_time"] = (
                    variant_dates
                )
            else:
                variant_df["_plot_time"] = (
                    variant_df["time"].astype(str)
                )

            variant_df = variant_df.sort_values(
                "_plot_time"
            )

            variant_x = variant_df["_plot_time"]

            # Make the normal prediction more visible.
            if variant == "normal":
                line_style = "-"
                line_width = 2.3
                line_alpha = 1.0
                zorder = 8
            else:
                line_style = "--"
                line_width = 1.8
                line_alpha = 0.9
                zorder = 5

            line = ax.plot(
                variant_x,
                variant_df["pred_mean"],
                linestyle=line_style,
                linewidth=line_width,
                alpha=line_alpha,
                label=variant_labels.get(
                    variant,
                    variant.replace("_", " ").title(),
                ),
                zorder=zorder,
            )[0]

            # Prediction uncertainty bands can make the figure
            # crowded, so they are disabled by default.
            if show_prediction_std:
                ax.fill_between(
                    variant_x,
                    variant_df["pred_mean"]
                    - variant_df["pred_std"],
                    variant_df["pred_mean"]
                    + variant_df["pred_std"],
                    color=line.get_color(),
                    alpha=0.08,
                    linewidth=0,
                    zorder=2,
                )

        # -----------------------------------------------------
        # 8. Figure formatting
        # -----------------------------------------------------
        ax.set_title(
            title,
            loc="left",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("NDVI")

        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend(
            loc="upper right",
            frameon=False,
            ncol=2,
            fontsize=9,
        )

        if x_is_datetime:
            fig.autofmt_xdate(
                rotation=45
            )
        else:
            ax.tick_params(
                axis="x",
                labelrotation=45,
            )

        fig.tight_layout()

        fig.savefig(
            plot_dir / output_name,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved time-series comparison: "
            f"{plot_dir / output_name}"
        )


def plot_ndvi_timeseries_comparisons(
    series_df: pd.DataFrame,
    output_dir,
    y_min: float = 0.0,
    y_max: float = 1.0,
):
    """Plot observed and predicted NDVI at whole-image and field-mask levels."""
    plot_dir = Path(output_dir) / "full_dataset_timeseries"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Remove obsolete figures that displayed the input-ablation variants.
    for legacy_name in (
        "field_mask_all_variants_ndvi_timeseries.png",
        "global_all_variants_ndvi_timeseries.png",
    ):
        (plot_dir / legacy_name).unlink(missing_ok=True)

    required_columns = {
        "variant",
        "scope",
        "time",
        "target_mean",
        "target_std",
        "pred_mean",
    }
    missing_columns = required_columns - set(series_df.columns)
    if missing_columns:
        raise ValueError(
            "series_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    def prepare(scope, variant):
        data = series_df[
            (series_df["scope"] == scope)
            & (series_df["variant"] == variant)
        ].copy()
        if data.empty:
            return data, None, False

        parsed_time = pd.to_datetime(data["time"], errors="coerce")
        is_datetime = parsed_time.notna().all()
        data["_plot_time"] = (
            parsed_time if is_datetime else data["time"].astype(str)
        )
        data = data.sort_values("_plot_time")
        return data, data["_plot_time"], is_datetime

    def add_target(ax, x, data, line_color, band_color, label, zorder):
        mean = data["target_mean"].to_numpy(dtype=float)
        std = data["target_std"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=band_color,
            alpha=0.28,
            linewidth=0,
            label=f"{label} ±1 SD",
            zorder=zorder,
        )
        ax.plot(
            x,
            mean,
            color=line_color,
            linewidth=2.5,
            label=label,
            zorder=zorder + 1,
        )

    def format_axis(fig, ax, title, uses_datetime):
        ax.set_title(title, loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("NDVI")
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
        if uses_datetime:
            fig.autofmt_xdate(rotation=45)
        else:
            ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()

    # Figure 1: unmodified model, comparing whole-image and field-mask NDVI.
    global_normal, global_x, global_datetime = prepare("global", "normal")
    field_normal, field_x, field_datetime = prepare("field_mask", "normal")
    if not global_normal.empty:
        fig, ax = plt.subplots(figsize=(13, 5.8))
        add_target(
            ax,
            global_x,
            global_normal,
            line_color="#14532d",
            band_color="#b7dfb9",
            label="Observed NDVI — whole image",
            zorder=2,
        )
        ax.plot(
            global_x,
            global_normal["pred_mean"].to_numpy(dtype=float),
            color="#66c266",
            linestyle="--",
            linewidth=2.5,
            label="Predicted NDVI — whole image",
            zorder=8,
        )

        if not field_normal.empty:
            add_target(
                ax,
                field_x,
                field_normal,
                line_color="#123b6d",
                band_color="#b7d9f5",
                label="Observed NDVI — field mask",
                zorder=3,
            )
            ax.plot(
                field_x,
                field_normal["pred_mean"].to_numpy(dtype=float),
                color="#56b4e9",
                linestyle="--",
                linewidth=2.5,
                label="Predicted NDVI — field mask",
                zorder=9,
            )

        format_axis(
            fig,
            ax,
            "Observed and predicted NDVI: whole image vs field mask",
            global_datetime or field_datetime,
        )
        comparison_path = (
            plot_dir / "global_and_field_observed_predicted_ndvi_timeseries.png"
        )
        fig.savefig(comparison_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved time-series comparison: {comparison_path}")


def plot_metrics_by_horizon(horizon_summary_df: pd.DataFrame, output_dir):
    if len(horizon_summary_df) == 0:
        return
    output_dir = Path(output_dir) / "horizon_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("rmse", "RMSE", False),
        ("pearson_r2", "Squared Pearson correlation", True),
        ("pss_mse", "Persistence skill score", True),
        ("future_closer_rate", "Future-Closer Rate", True),
    ]

    for metric, ylabel, unit_interval in metric_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, group in horizon_summary_df.groupby("variant"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group[metric], marker="o", label=variant)
        ax.set_xlabel("forecast horizon")
        ax.set_ylabel(ylabel)
        if unit_interval:
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}_by_horizon.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------
# Dataset-level aggregation helpers
# ---------------------------------------------------------------------


TOTAL_KEYS = [
    "sse",
    "sae",
    "bias_sum",
    "pred_sum",
    "target_sum",
    "pred_square_sum",
    "target_square_sum",
    "pred_target_sum",
    "n_obs",
    "skill_sse",
    "persistence_sse",
    "n_persistence_obs",
    "future_closer_count",
    "n_changed",
]

HORIZON_TOTAL_KEYS = [
    "sse_by_horizon",
    "sae_by_horizon",
    "bias_sum_by_horizon",
    "pred_sum_by_horizon",
    "target_sum_by_horizon",
    "pred_square_sum_by_horizon",
    "target_square_sum_by_horizon",
    "pred_target_sum_by_horizon",
    "n_obs_by_horizon",
    "skill_sse_by_horizon",
    "persistence_sse_by_horizon",
    "n_persistence_obs_by_horizon",
    "future_closer_count_by_horizon",
    "n_changed_by_horizon",
]


def _pooled_pearson_r2(
    n_obs: float,
    pred_sum: float,
    target_sum: float,
    pred_square_sum: float,
    target_square_sum: float,
    pred_target_sum: float,
    eps: float = 1e-8,
) -> float:
    """Squared Pearson correlation from pooled additive moments."""
    if n_obs < 2:
        return float("nan")
    covariance_numerator = n_obs * pred_target_sum - pred_sum * target_sum
    pred_variance_numerator = max(
        n_obs * pred_square_sum - pred_sum**2,
        0.0,
    )
    target_variance_numerator = max(
        n_obs * target_square_sum - target_sum**2,
        0.0,
    )
    denominator = pred_variance_numerator * target_variance_numerator
    if denominator <= eps:
        return float("nan")
    return min(max(covariance_numerator**2 / denominator, 0.0), 1.0)


def _summarize_totals(variant_name: str, item: dict, eps: float = 1e-8):
    n_obs = item["n_obs"]
    n_persistence_obs = item["n_persistence_obs"]
    n_changed = item["n_changed"]

    rmse = math.sqrt(item["sse"] / n_obs) if n_obs > 0 else float("nan")
    mae = item["sae"] / n_obs if n_obs > 0 else float("nan")
    bias = item["bias_sum"] / n_obs if n_obs > 0 else float("nan")
    pearson_r2 = _pooled_pearson_r2(
        n_obs=n_obs,
        pred_sum=item["pred_sum"],
        target_sum=item["target_sum"],
        pred_square_sum=item["pred_square_sum"],
        target_square_sum=item["target_square_sum"],
        pred_target_sum=item["pred_target_sum"],
        eps=eps,
    )
    persistence_rmse = (
        math.sqrt(item["persistence_sse"] / n_persistence_obs)
        if n_persistence_obs > 0
        else float("nan")
    )
    pss_mse = (
        1.0 - item["skill_sse"] / item["persistence_sse"]
        if item["persistence_sse"] > eps
        else float("nan")
    )
    future_closer_rate = (
        item["future_closer_count"] / n_changed
        if n_changed > 0
        else float("nan")
    )
    return {
        "variant": variant_name,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "pearson_r2": pearson_r2,
        "persistence_rmse": persistence_rmse,
        "pss_mse": pss_mse,
        "future_closer_rate": future_closer_rate,
        "n_obs": n_obs,
        "n_persistence_obs": n_persistence_obs,
        "n_changed": n_changed,
        "changed_fraction": n_changed / n_persistence_obs
        if n_persistence_obs > 0
        else float("nan"),
    }


def _summarize_horizon_totals(
    variant_name: str,
    item: dict,
    target_length: int,
    eps: float = 1e-8,
):
    rows = []
    for h in range(target_length):
        n_obs = float(item["n_obs_by_horizon"][h])
        n_persistence_obs = float(item["n_persistence_obs_by_horizon"][h])
        n_changed = float(item["n_changed_by_horizon"][h])
        sse = float(item["sse_by_horizon"][h])
        sae = float(item["sae_by_horizon"][h])
        bias_sum = float(item["bias_sum_by_horizon"][h])
        pearson_r2 = _pooled_pearson_r2(
            n_obs=n_obs,
            pred_sum=float(item["pred_sum_by_horizon"][h]),
            target_sum=float(item["target_sum_by_horizon"][h]),
            pred_square_sum=float(item["pred_square_sum_by_horizon"][h]),
            target_square_sum=float(item["target_square_sum_by_horizon"][h]),
            pred_target_sum=float(item["pred_target_sum_by_horizon"][h]),
            eps=eps,
        )
        skill_sse = float(item["skill_sse_by_horizon"][h])
        persistence_sse = float(item["persistence_sse_by_horizon"][h])

        rows.append(
            {
                "variant": variant_name,
                "horizon": h + 1,
                "rmse": math.sqrt(sse / n_obs) if n_obs > 0 else float("nan"),
                "mae": sae / n_obs if n_obs > 0 else float("nan"),
                "bias": bias_sum / n_obs if n_obs > 0 else float("nan"),
                "pearson_r2": pearson_r2,
                "persistence_rmse": math.sqrt(
                    persistence_sse / n_persistence_obs
                )
                if n_persistence_obs > 0
                else float("nan"),
                "pss_mse": 1.0 - skill_sse / persistence_sse
                if persistence_sse > eps
                else float("nan"),
                "future_closer_rate": float(
                    item["future_closer_count_by_horizon"][h]
                )
                / n_changed
                if n_changed > 0
                else float("nan"),
                "n_obs": n_obs,
                "n_persistence_obs": n_persistence_obs,
                "n_changed": n_changed,
                "changed_fraction": n_changed / n_persistence_obs
                if n_persistence_obs > 0
                else float("nan"),
            }
        )
    return rows


# ---------------------------------------------------------------------
# Main ablation evaluation
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate_ablation_to_tiff(
    model,
    dataloader,
    output_dir,
    device,
    context_length: int,
    target_length: int,
    lc_min: Optional[float] = 10,
    lc_max: Optional[float] = 40,
    ndvi_pred_channel: int = 0,
    ndvi_targ_channel: int = 0,
    use_cloud_mask: bool = True,
    use_lc_mask: bool = True,
    use_last_valid_context: bool = True,
    change_threshold: float = 0.03,
    ndvi_noise_min: float = 0.0,
    ndvi_noise_max: float = 1.0,
    ablation_seed: int = 42,
    include_strong_all_shuffle: bool = False,
    max_batches: Optional[int] = None,
    save_tiffs: bool = True,
    save_targets: bool = True,
    save_full_dataset_timeseries: bool = True,
    save_horizon_plots: bool = True,
    input_mode: str = "",
    zeroing_nextcloud: bool = False,
    field_eval_mask_path: Optional[str] = None,
):
    """
    Evaluate normal inference and conditioning ablations. Unprefixed metrics
    cover the complete valid image; field_* metrics cover field_eval_mask_path.

    Returns:
        summary_df
        per_sample_df
        horizon_summary_df
        per_horizon_df
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = get_ablation_variants(
        include_strong_all_shuffle=include_strong_all_shuffle
    )
    model.eval()

    zero_nextcloud_active = (
        zeroing_nextcloud and "nextcloud" in str(input_mode).lower()
    )
    if zeroing_nextcloud:
        if zero_nextcloud_active:
            print(
                "Next-cloud ablation enabled: zeroing the final channel of "
                f"dynamic image input (input_mode={input_mode!r})."
            )
        else:
            print(
                "Next-cloud ablation requested but input_mode does not contain "
                f"'nextcloud'; leaving the batch unchanged (input_mode={input_mode!r})."
            )

    series_acc = None
    field_mask_hw = None
    field_mask_tensor = None
    reference_target_sum_hw = None
    reference_target_count_hw = None

    if save_full_dataset_timeseries:
        series_acc = {
            variant: defaultdict(lambda: defaultdict(_new_stats_dict))
            for variant in variants
        }

    totals = {
        variant: {key: 0.0 for key in TOTAL_KEYS}
        for variant in variants
    }
    field_totals = {
        variant: {key: 0.0 for key in TOTAL_KEYS}
        for variant in variants
    }
    horizon_totals = {
        variant: {
            key: torch.zeros(target_length, dtype=torch.float64)
            for key in HORIZON_TOTAL_KEYS
        }
        for variant in variants
    }
    field_horizon_totals = {
        variant: {
            key: torch.zeros(target_length, dtype=torch.float64)
            for key in HORIZON_TOTAL_KEYS
        }
        for variant in variants
    }

    per_sample_rows = []
    per_horizon_rows = []
    global_idx = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = batch_to_device(batch, device)
        batch = zero_nextcloud_input_channel(
            batch=batch,
            input_mode=input_mode,
            zeroing_nextcloud=zero_nextcloud_active,
        )
        batch_size = batch["dynamic"][0].shape[0]

        if field_eval_mask_path and field_mask_tensor is None:
            target_hw = tuple(batch["dynamic"][0].shape[-2:])
            field_mask_hw = load_field_evaluation_mask(
                mask_path=field_eval_mask_path,
                target_hw=target_hw,
            )
            field_mask_tensor = torch.as_tensor(
                field_mask_hw,
                dtype=torch.bool,
                device=device,
            )
            print(
                "Image-level metrics use all valid pixels; separate field_* "
                "metrics use the "
                f"{int(field_mask_hw.sum())} pixels selected by the field mask."
            )

        if field_mask_tensor is not None:
            reference_target, reference_valid = extract_target_and_mask_from_batch(
                batch=batch,
                context_length=context_length,
                target_length=target_length,
                lc_min=lc_min,
                lc_max=lc_max,
                ndvi_targ_channel=ndvi_targ_channel,
                use_cloud_mask=use_cloud_mask,
                use_lc_mask=use_lc_mask,
            )
            batch_target_sum_hw = (
                reference_target * reference_valid
            ).sum(dim=(0, 1, 2)).detach().float().cpu()
            batch_target_count_hw = (
                reference_valid.sum(dim=(0, 1, 2)).detach().float().cpu()
            )
            if reference_target_sum_hw is None:
                reference_target_sum_hw = batch_target_sum_hw
                reference_target_count_hw = batch_target_count_hw
            else:
                reference_target_sum_hw += batch_target_sum_hw
                reference_target_count_hw += batch_target_count_hw

        for variant_index, (variant_name, variant_fn) in enumerate(variants.items()):
            variant_seed = (
                ablation_seed
                + batch_idx * 100000
                + variant_index * 10000
            )
            ablated_batch = variant_fn(
                batch,
                context_length=context_length,
                ndvi_channel=ndvi_targ_channel,
                ndvi_min=ndvi_noise_min,
                ndvi_max=ndvi_noise_max,
                seed=variant_seed,
            )

            output = model(
                ablated_batch,
                pred_start=context_length,
                preds_length=target_length,
            )
            preds = output[0] if isinstance(output, (tuple, list)) else output

            metrics = compute_common_metrics_batch(
                preds=preds,
                batch=batch,
                context_length=context_length,
                target_length=target_length,
                lc_min=lc_min,
                lc_max=lc_max,
                ndvi_pred_channel=ndvi_pred_channel,
                ndvi_targ_channel=ndvi_targ_channel,
                use_cloud_mask=use_cloud_mask,
                use_lc_mask=use_lc_mask,
                use_last_valid_context=use_last_valid_context,
                change_threshold=change_threshold,
                spatial_mask=None,
            )
            field_metrics = None
            if field_mask_tensor is not None:
                field_metrics = compute_common_metrics_batch(
                    preds=preds,
                    batch=batch,
                    context_length=context_length,
                    target_length=target_length,
                    lc_min=lc_min,
                    lc_max=lc_max,
                    ndvi_pred_channel=ndvi_pred_channel,
                    ndvi_targ_channel=ndvi_targ_channel,
                    use_cloud_mask=use_cloud_mask,
                    use_lc_mask=use_lc_mask,
                    use_last_valid_context=use_last_valid_context,
                    change_threshold=change_threshold,
                    spatial_mask=field_mask_tensor,
                )

            for key in TOTAL_KEYS:
                totals[variant_name][key] += float(
                    metrics[key].sum().detach().cpu()
                )

            for key in HORIZON_TOTAL_KEYS:
                horizon_totals[variant_name][key] += (
                    metrics[key]
                    .sum(dim=0)
                    .detach()
                    .double()
                    .cpu()
                )

            if field_metrics is not None:
                for key in TOTAL_KEYS:
                    field_totals[variant_name][key] += float(
                        field_metrics[key].sum().detach().cpu()
                    )
                for key in HORIZON_TOTAL_KEYS:
                    field_horizon_totals[variant_name][key] += (
                        field_metrics[key]
                        .sum(dim=0)
                        .detach()
                        .double()
                        .cpu()
                    )

            for i in range(batch_size):
                sample_global_idx = global_idx + i
                sample_name = get_sample_name(batch, i, sample_global_idx)
                pred_t_hw = metrics["ndvi_pred"][i, :, 0]
                targ_t_hw = metrics["ndvi_targ"][i, :, 0]
                mask_t_hw = metrics["mask"][i, :, 0]
                persistence_t_hw = metrics["persistence"][i, :, 0]

                time_metadata = get_target_time_metadata(
                    batch=batch,
                    sample_idx=i,
                    context_length=context_length,
                    target_len=target_length,
                    batch_size=batch_size,
                    global_idx=sample_global_idx,
                )

                if save_full_dataset_timeseries and series_acc is not None:
                    for t in range(targ_t_hw.shape[0]):
                        global_valid = mask_t_hw[t] > 0
                        time_key = time_metadata[t]["target_time"]
                        update_series_stats(
                            series_acc=series_acc,
                            variant=variant_name,
                            scope="global",
                            time_key=time_key,
                            target_values=targ_t_hw[t][global_valid],
                            pred_values=pred_t_hw[t][global_valid],
                        )
                        if field_metrics is not None:
                            field_valid = field_metrics["mask"][i, t, 0] > 0
                            update_series_stats(
                                series_acc=series_acc,
                                variant=variant_name,
                                scope="field_mask",
                                time_key=time_key,
                                target_values=targ_t_hw[t][field_valid],
                                pred_values=pred_t_hw[t][field_valid],
                            )

                pred_path = target_path = mask_path = persistence_path = None
                if save_tiffs:
                    (
                        pred_path,
                        target_path,
                        mask_path,
                        persistence_path,
                    ) = save_sample_tiffs(
                        output_dir=output_dir,
                        variant=variant_name,
                        sample_name=sample_name,
                        pred_t_hw=pred_t_hw,
                        targ_t_hw=targ_t_hw,
                        mask_t_hw=mask_t_hw,
                        persistence_t_hw=persistence_t_hw,
                        save_target=(save_targets and variant_name == "normal"),
                    )

                per_sample_rows.append(
                    {
                        "variant": variant_name,
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "global_idx": sample_global_idx,
                        "name": sample_name,
                        "rmse": _scalar(metrics["rmse"][i]),
                        "mae": _scalar(metrics["mae"][i]),
                        "bias": _scalar(metrics["bias"][i]),
                        "pearson_r2": _scalar(metrics["pearson_r2"][i]),
                        "persistence_rmse": _scalar(
                            metrics["persistence_rmse"][i]
                        ),
                        "pss_mse": _scalar(metrics["pss_mse"][i]),
                        "future_closer_rate": _scalar(
                            metrics["future_closer_rate"][i]
                        ),
                        "field_rmse": _scalar(field_metrics["rmse"][i])
                        if field_metrics is not None
                        else float("nan"),
                        "field_mae": _scalar(field_metrics["mae"][i])
                        if field_metrics is not None
                        else float("nan"),
                        "field_bias": _scalar(field_metrics["bias"][i])
                        if field_metrics is not None
                        else float("nan"),
                        "field_pearson_r2": _scalar(
                            field_metrics["pearson_r2"][i]
                        )
                        if field_metrics is not None
                        else float("nan"),
                        "field_persistence_rmse": _scalar(
                            field_metrics["persistence_rmse"][i]
                        )
                        if field_metrics is not None
                        else float("nan"),
                        "field_pss_mse": _scalar(field_metrics["pss_mse"][i])
                        if field_metrics is not None
                        else float("nan"),
                        "field_future_closer_rate": _scalar(
                            field_metrics["future_closer_rate"][i]
                        )
                        if field_metrics is not None
                        else float("nan"),
                        "n_obs": _scalar(metrics["n_obs"][i]),
                        "n_persistence_obs": _scalar(
                            metrics["n_persistence_obs"][i]
                        ),
                        "n_changed": _scalar(metrics["n_changed"][i]),
                        "field_n_obs": _scalar(field_metrics["n_obs"][i])
                        if field_metrics is not None
                        else 0.0,
                        "field_n_persistence_obs": _scalar(
                            field_metrics["n_persistence_obs"][i]
                        )
                        if field_metrics is not None
                        else 0.0,
                        "field_n_changed": _scalar(
                            field_metrics["n_changed"][i]
                        )
                        if field_metrics is not None
                        else 0.0,
                        "pred_tiff": str(pred_path) if pred_path else "",
                        "target_tiff": str(target_path) if target_path else "",
                        "mask_tiff": str(mask_path) if mask_path else "",
                        "persistence_tiff": str(persistence_path)
                        if persistence_path
                        else "",
                    }
                )

                for h in range(target_length):
                    per_horizon_rows.append(
                        {
                            "variant": variant_name,
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            "global_idx": sample_global_idx,
                            "name": sample_name,
                            "horizon": h + 1,
                            "target_time": time_metadata[h]["target_time"],
                            "lead_days": time_metadata[h]["lead_days"],
                            "rmse": _scalar(metrics["rmse_by_horizon"][i, h]),
                            "mae": _scalar(metrics["mae_by_horizon"][i, h]),
                            "bias": _scalar(metrics["bias_by_horizon"][i, h]),
                            "pearson_r2": _scalar(
                                metrics["pearson_r2_by_horizon"][i, h]
                            ),
                            "persistence_rmse": _scalar(
                                metrics["persistence_rmse_by_horizon"][i, h]
                            ),
                            "pss_mse": _scalar(
                                metrics["pss_mse_by_horizon"][i, h]
                            ),
                            "future_closer_rate": _scalar(
                                metrics["future_closer_rate_by_horizon"][i, h]
                            ),
                            "field_rmse": _scalar(
                                field_metrics["rmse_by_horizon"][i, h]
                            )
                            if field_metrics is not None
                            else float("nan"),
                            "field_mae": _scalar(
                                field_metrics["mae_by_horizon"][i, h]
                            )
                            if field_metrics is not None
                            else float("nan"),
                            "field_bias": _scalar(
                                field_metrics["bias_by_horizon"][i, h]
                            )
                            if field_metrics is not None
                            else float("nan"),
                            "field_pearson_r2": _scalar(
                                field_metrics["pearson_r2_by_horizon"][i, h]
                            )
                            if field_metrics is not None
                            else float("nan"),
                            "field_persistence_rmse": _scalar(
                                field_metrics[
                                    "persistence_rmse_by_horizon"
                                ][i, h]
                            )
                            if field_metrics is not None
                            else float("nan"),
                            "field_pss_mse": _scalar(
                                field_metrics["pss_mse_by_horizon"][i, h]
                            )
                            if field_metrics is not None
                            else float("nan"),
                            "field_future_closer_rate": _scalar(
                                field_metrics[
                                    "future_closer_rate_by_horizon"
                                ][i, h]
                            )
                            if field_metrics is not None
                            else float("nan"),
                            "n_obs": _scalar(metrics["n_obs_by_horizon"][i, h]),
                            "n_persistence_obs": _scalar(
                                metrics["n_persistence_obs_by_horizon"][i, h]
                            ),
                            "n_changed": _scalar(
                                metrics["n_changed_by_horizon"][i, h]
                            ),
                            "field_n_obs": _scalar(
                                field_metrics["n_obs_by_horizon"][i, h]
                            )
                            if field_metrics is not None
                            else 0.0,
                            "field_n_persistence_obs": _scalar(
                                field_metrics[
                                    "n_persistence_obs_by_horizon"
                                ][i, h]
                            )
                            if field_metrics is not None
                            else 0.0,
                            "field_n_changed": _scalar(
                                field_metrics["n_changed_by_horizon"][i, h]
                            )
                            if field_metrics is not None
                            else 0.0,
                        }
                    )

        global_idx += batch_size
        print(f"Finished batch {batch_idx}")

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_df.to_csv(
        output_dir / "per_sample_ablation_metrics.csv",
        index=False,
    )

    per_horizon_df = pd.DataFrame(per_horizon_rows)
    per_horizon_df.to_csv(
        output_dir / "per_sample_horizon_metrics.csv",
        index=False,
    )

    summary_rows = []
    for variant_name, item in totals.items():
        row = _summarize_totals(variant_name, item)
        field_row = _summarize_totals(
            variant_name,
            field_totals[variant_name],
        )
        row.update(
            {
                f"field_{key}": value
                for key, value in field_row.items()
                if key != "variant"
            }
        )
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "pss_mse",
        ascending=False,
        na_position="last",
    )
    summary_df.to_csv(
        output_dir / "summary_ablation_metrics.csv",
        index=False,
    )
    image_summary_df = summary_df[
        [column for column in summary_df.columns if not column.startswith("field_")]
    ]
    image_summary_df.to_csv(
        output_dir / "summary_image_metrics.csv",
        index=False,
    )
    if field_mask_tensor is not None:
        field_columns = [
            column for column in summary_df.columns if column.startswith("field_")
        ]
        field_summary_df = summary_df[["variant", *field_columns]].rename(
            columns={column: column.removeprefix("field_") for column in field_columns}
        )
        field_summary_df.to_csv(
            output_dir / "summary_field_metrics.csv",
            index=False,
        )
    with open(output_dir / "summary_ablation_metrics.json", "w") as fp:
        json.dump(_json_safe_rows(summary_rows), fp, indent=2)

    horizon_summary_rows = []
    for variant_name, item in horizon_totals.items():
        image_rows = _summarize_horizon_totals(
            variant_name=variant_name,
            item=item,
            target_length=target_length,
        )
        field_rows = _summarize_horizon_totals(
            variant_name=variant_name,
            item=field_horizon_totals[variant_name],
            target_length=target_length,
        )
        for image_row, field_row in zip(image_rows, field_rows):
            image_row.update(
                {
                    f"field_{key}": value
                    for key, value in field_row.items()
                    if key not in {"variant", "horizon"}
                }
            )
            horizon_summary_rows.append(image_row)
    horizon_summary_df = pd.DataFrame(horizon_summary_rows)
    horizon_summary_df.to_csv(
        output_dir / "summary_metrics_by_horizon.csv",
        index=False,
    )
    image_horizon_df = horizon_summary_df[
        [
            column
            for column in horizon_summary_df.columns
            if not column.startswith("field_")
        ]
    ]
    image_horizon_df.to_csv(
        output_dir / "summary_image_metrics_by_horizon.csv",
        index=False,
    )
    if field_mask_tensor is not None:
        field_horizon_columns = [
            column
            for column in horizon_summary_df.columns
            if column.startswith("field_")
        ]
        field_horizon_df = horizon_summary_df[
            ["variant", "horizon", *field_horizon_columns]
        ].rename(
            columns={
                column: column.removeprefix("field_")
                for column in field_horizon_columns
            }
        )
        field_horizon_df.to_csv(
            output_dir / "summary_field_metrics_by_horizon.csv",
            index=False,
        )

    if save_horizon_plots:
        plot_metrics_by_horizon(horizon_summary_df, output_dir)

    if field_mask_hw is not None and reference_target_sum_hw is not None:
        mean_target_hw = (
            reference_target_sum_hw
            / reference_target_count_hw.clamp_min(1.0)
        ).numpy()
        mean_target_hw[reference_target_count_hw.numpy() == 0] = np.nan
        save_field_mask_overlay_image(
            mean_target_hw=mean_target_hw,
            field_mask_hw=field_mask_hw,
            save_path=output_dir / "field_eval_mask_overlay.png",
        )

    if save_full_dataset_timeseries and series_acc is not None:
        series_df = finalize_series_accumulator(series_acc)
        series_df.to_csv(
            output_dir / "full_dataset_timeseries_stats.csv",
            index=False,
        )
        plot_ndvi_timeseries_comparisons(
            series_df=series_df,
            output_dir=output_dir,
            y_min=0.0,
            y_max=1.0,
        )
        plot_full_dataset_series(
            series_df=series_df,
            patches=[],
            output_dir=output_dir,
            include_field_mask=field_mask_hw is not None,
            y_min=0.0,
            y_max=1.0,
        )

    print("\nAblation summary:")
    print(summary_df)

    return summary_df, per_sample_df, horizon_summary_df, per_horizon_df


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------
#
# summary_df, per_sample_df, horizon_summary_df, per_horizon_df = (
#     evaluate_ablation_to_tiff(
#         model=model,
#         dataloader=test_loader,
#         output_dir="evaluation_outputs",
#         device=device,
#         context_length=3,
#         target_length=1,
#         lc_min=10,
#         lc_max=40,
#         ndvi_pred_channel=0,
#         ndvi_targ_channel=0,
#         use_cloud_mask=True,
#         use_lc_mask=True,
#         use_last_valid_context=True,
#         change_threshold=0.03,
#     )
# )
