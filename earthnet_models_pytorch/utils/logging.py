# Based on the visualization utilities from:
# https://github.com/earthnet2021/earthnet-models-pytorch
#
# Original implementation licensed under the MIT License.
#
# Substantially modified and extended by Cynthia Ugwu (@cugwu) for:
# https://github.com/multimedia-eurecat/short-range-ndvi-forecasting
#
# Reworked the visualization pipeline for short-range NDVI forecasting,
# including temporal sequence handling, cloud and validity masks,
# prediction residuals, per-sample RMSE, and comparative visualization
# of conditioning, predicted, and target frames.
# Added configurable PNG export and TensorBoard logging.

import copy
import re
from pathlib import Path
from typing import Optional

import matplotlib.colors as clr
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

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

# CMAP_KNDVI = clr.LinearSegmentedColormap.from_list('kndvi', ["#cbbe9a","#fffde4","#bccea5", "#83AD75", "#689B5F", "#528A4F", "#38773B", "#286531", "#195226", "#0F441F", "#0D421D", "#0B3E1C", "#072F17", "#021D0E", "#01120B", "#010C09", "#000d0a"], N=256)
# CMAP_KNDVI.set_bad(color='white')

# CMAPS = {"ndvi": CMAP_NDVI, "kndvi": CMAP_KNDVI}

def _safe_name(value: str, max_len: int = 140) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value[:max_len]

def _to_time_hw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert common NDVI/mask shapes to [T,H,W].

    Accepted:
        [H,W]
        [1,H,W]
        [T,H,W]
        [T,1,H,W]
        [1,T,H,W]
    """
    if x.ndim == 2:
        return x.unsqueeze(0)

    if x.ndim == 3:
        return x

    if x.ndim == 4:
        if x.shape[1] == 1:
            return x[:, 0]      # [T,1,H,W] -> [T,H,W]
        if x.shape[0] == 1:
            return x[0]         # [1,T,H,W] -> [T,H,W]

    raise ValueError(f"Cannot convert shape {tuple(x.shape)} to [T,H,W]")

def _repeat_to_t(mask: Optional[torch.Tensor], t: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    mask = _to_time_hw(mask)

    if mask.shape[0] == t:
        return mask

    if mask.shape[0] == 1:
        return mask.repeat(t, 1, 1)

    raise ValueError(f"Cannot repeat mask with shape {tuple(mask.shape)} to T={t}")


def _text_bar(text: str, width: int, device, dtype, height_pad: int = 6) -> torch.Tensor:
    """
    Create a white text bar as [3,H,W].
    """
    font = ImageFont.load_default()

    try:
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = font.getsize(text)

    height = max(text_h + height_pad, 16)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    x = max((width - text_w) // 2, 0)
    y = max((height - text_h) // 2, 0)
    draw.text((x, y), text, font=font, fill=(0, 0, 0))

    arr = np.asarray(canvas).astype(np.float32) / 255.0
    return torch.as_tensor(arr, dtype=dtype, device=device).permute(2, 0, 1)

def ndvi_colorize(
        ndvi: torch.Tensor, 
        invalid_mask: Optional[torch.Tensor] = None,     
        vmin: Optional[float] = -1,
        vmax: Optional[float] = 1,
        ) -> torch.Tensor:
    """
    Colorize NDVI.

    Args:
        ndvi: [T,H,W], values expected in [-1,1].
        invalid_mask: [T,H,W], True means invalid/cloud/masked.

    Returns:
        [T,3,H,W] in [0,1].
    """
    ndvi = _to_time_hw(ndvi).detach().float().clamp(-1.0, 1.0)
    ndvi = (ndvi - vmin) / (vmax - vmin)
    ndvi = ndvi.clamp(0.0, 1.0)
    
    t, h, w = ndvi.shape

    flat = copy.deepcopy(ndvi.reshape(-1)).detach().cpu().numpy()

    if invalid_mask is not None:
        invalid_mask = _repeat_to_t(invalid_mask, t).bool()
        flat = np.ma.array(
            flat,
            mask=copy.deepcopy(invalid_mask.reshape(-1)).detach().cpu().numpy(),
        )

    rgb = (
        torch.as_tensor(CMAP_NDVI(flat)[:, :3], dtype=ndvi.dtype, device=ndvi.device)
        .reshape(t, h, w, 3)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    return rgb

def residual_colorize(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    scale: float = 0.5,
) -> torch.Tensor:
    """
    Red/blue residual visualization.

    Args:
        pred: [T,H,W]
        target: [T,H,W]
        valid_mask: [T,H,W], True means valid. Invalid pixels become gray.
        scale: absolute residual mapped to full saturation.

    Meaning:
        red   = prediction higher than target
        blue  = prediction lower than target
        white = small residual
        gray  = invalid/cloud
    """
    pred = _to_time_hw(pred).detach().float()
    target = _to_time_hw(target).detach().float()

    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    residual = pred - target
    scale = max(float(scale), 1e-6)

    pos = (residual.clamp(min=0.0) / scale).clamp(0.0, 1.0)
    neg = ((-residual).clamp(min=0.0) / scale).clamp(0.0, 1.0)

    vis = torch.ones(
        residual.shape[0],
        3,
        residual.shape[1],
        residual.shape[2],
        dtype=residual.dtype,
        device=residual.device,
    )

    # Positive residual -> red.
    vis[:, 1] = (vis[:, 1] - pos - neg).clamp(0.0, 1.0)
    vis[:, 2] = (vis[:, 2] - pos).clamp(0.0, 1.0)

    # Negative residual -> blue.
    vis[:, 0] = (vis[:, 0] - neg).clamp(0.0, 1.0)

    if valid_mask is not None:
        valid_mask = _repeat_to_t(valid_mask, residual.shape[0]).bool()
        invalid = ~valid_mask
        gray = torch.full_like(vis, 0.65)
        vis = torch.where(invalid.unsqueeze(1), gray, vis)

    return vis.contiguous()


def cloud_colorize(cloud: torch.Tensor) -> torch.Tensor:
    """
    Visualize cloud / invalid mask or probability.

    Args:
        cloud: [T,H,W], values in [0,1], where 1 means cloud/invalid.

    Returns:
        [T,3,H,W]
    """
    cloud = _to_time_hw(cloud).detach().float().clamp(0.0, 1.0)

    # White = clear, blue/dark = cloud.
    clear = 1.0 - cloud
    rgb = clear.unsqueeze(1).repeat(1, 3, 1, 1)

    # Blue tint on cloud/probability.
    rgb[:, 2] = torch.maximum(rgb[:, 2], 0.55 * cloud)

    return rgb.contiguous()


def _strip(frames: torch.Tensor, frame_pad: int = 3) -> torch.Tensor:
    """
    Convert [T,3,H,W] into one horizontal strip [3,H,W_total].
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"Expected [T,3,H,W], got {tuple(frames.shape)}")

    frames = frames.detach().float().clamp(0.0, 1.0)

    parts = []
    for t in range(frames.shape[0]):
        if t > 0 and frame_pad > 0:
            spacer = torch.ones(
                3,
                frames.shape[2],
                frame_pad,
                dtype=frames.dtype,
                device=frames.device,
            )
            parts.append(spacer)

        parts.append(frames[t])

    return torch.cat(parts, dim=2)


def _label_row(row: torch.Tensor, label: str) -> torch.Tensor:
    """
    Add text label above one row.
    """
    text = _text_bar(label, width=row.shape[-1], device=row.device, dtype=row.dtype)
    return torch.cat([text, row], dim=1)


def _local_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    pred = _to_time_hw(pred).detach().float()
    target = _to_time_hw(target).detach().float()

    err = (pred - target) ** 2

    if valid_mask is not None:
        valid_mask = _repeat_to_t(valid_mask, pred.shape[0]).bool()
        denom = valid_mask.sum().clamp_min(1)
        rmse = torch.sqrt((err * valid_mask).sum() / denom)
    else:
        rmse = torch.sqrt(err.mean())

    return float(rmse.detach().cpu())


def _hcat_blocks(blocks: list[torch.Tensor], block_pad: int = 8) -> torch.Tensor:
    """
    Concatenate visualization blocks horizontally.

    Each block must be [3,H,W].
    Blocks are padded to the same height if needed.
    """
    max_h = max(block.shape[1] for block in blocks)

    padded = []
    for i, block in enumerate(blocks):
        if block.shape[1] < max_h:
            pad_h = max_h - block.shape[1]
            pad = torch.ones(
                block.shape[0],
                pad_h,
                block.shape[2],
                dtype=block.dtype,
                device=block.device,
            )
            block = torch.cat([block, pad], dim=1)

        if i > 0 and block_pad > 0:
            spacer = torch.ones(
                3,
                max_h,
                block_pad,
                dtype=block.dtype,
                device=block.device,
            )
            padded.append(spacer)

        padded.append(block)

    return torch.cat(padded, dim=2).clamp(0.0, 1.0)

def make_ndvi_residual_grid(
    batch: dict,
    preds: torch.Tensor,
    sample_idx: int,
    context_length: int,
    target_length: int,
    setting: str,
    score: Optional[dict] = None,
    residual_scale: float = 0.5,
    vmin: Optional[float] = -1,
    vmax: Optional[float] = 1,
) -> tuple[torch.Tensor, float]:
    """
    Build one horizontal visualization grid for one sample.

    Layout:

        condition NDVI | predicted NDVI | target NDVI | residual | cloud mask

    Supports:
        greenearthnet
        replicate

    Expected batch format:
        batch["dynamic"][2]      [B,T,C,H,W]
        batch["dynamic_mask"][0] [B,T,1,H,W], 0=valid, 1=cloud/invalid
        preds                   [B,T_pred,1,H,W]

    Channels:
        dynamic[0][...,0,:,:] = NDVI
        dynamic[0][...,1,:,:] = current cloud / cloud probability, if present
        dynamic[0][...,2,:,:] = next cloud, if present

    Returns:
        grid: [3,H_grid,W_grid] in [0,1]
        local_rmse: float
    """
    if setting not in ["greenearthnet", "replicate"]:
        raise ValueError(
            f"make_ndvi_residual_grid only supports greenearthnet/replicate, got {setting}"
        )

    if preds.ndim != 5:
        raise ValueError(f"Expected preds [B,T,C,H,W], got {tuple(preds.shape)}")

    dynamic = batch["dynamic"][2]  # [B,T,C,H,W]

    if dynamic.ndim != 5:
        raise ValueError(
            f"Expected batch['dynamic'][2] [B,T,C,H,W], got {tuple(dynamic.shape)}"
        )

    x = dynamic[sample_idx].detach()  # [T,C,H,W]

    total_t = x.shape[0]

    pred_ndvi = preds[sample_idx, -target_length:, 0].detach()  # [T_pred,H,W]

    target_start = context_length
    target_end = min(context_length + pred_ndvi.shape[0], total_t)

    # NDVI used as condition/context.
    condition_ndvi = x[:context_length, 0]  # [T_ctx,H,W]

    # Target and prediction for the predicted time window.
    target_ndvi = x[target_start:target_end, 0]       # [T_tgt,H,W]
    pred_ndvi = pred_ndvi[: target_ndvi.shape[0]]     # [T_tgt,H,W]

    if target_ndvi.shape[0] == 0:
        raise RuntimeError(
            f"No target NDVI available. total_t={total_t}, "
            f"context_length={context_length}, target_length={target_length}"
        )

    # Validity mask from EarthNet-style dynamic_mask.
    # Convention: 0 = valid/clear, 1 = invalid/cloud.
    if len(batch.get("dynamic_mask", [])) > 0:
        invalid_all = batch["dynamic_mask"][0][sample_idx, :, 0].detach().float()
    elif x.shape[1] > 1:
        invalid_all = x[:, 1].detach().float()
    else:
        invalid_all = torch.zeros_like(x[:, 0])

    invalid_all = invalid_all.clamp(0.0, 1.0)

    target_invalid = invalid_all[target_start:target_end] >= 1.0
    target_valid = ~target_invalid

    # Cloud mask/probability shown only for the predicted-frame window.
    # Prefer the input cloud channel if available because for replicate it can be probability.
    if x.shape[1] > 1:
        cloud_all = x[:, 1].detach().float().clamp(0.0, 1.0)
    else:
        cloud_all = invalid_all

    predicted_frame_cloud = cloud_all[target_start:target_end]

    # Important:
    # Do NOT mask NDVI panels here, otherwise cloudy pixels become white and the image is hard to read.
    # Cloud/invalid information is shown in the cloud-mask block instead.
    # vmin-vmax choose just for visualization
    condition_rgb = ndvi_colorize(condition_ndvi, invalid_mask=None, vmin=vmin, vmax=vmax)
    pred_rgb = ndvi_colorize(pred_ndvi, invalid_mask=None, vmin=vmin, vmax=vmax)
    target_rgb = ndvi_colorize(target_ndvi, invalid_mask=None, vmin=vmin, vmax=vmax)

    residual_rgb = residual_colorize(
        pred=pred_ndvi,
        target=target_ndvi,
        valid_mask=target_valid,
        scale=residual_scale,
    )

    cloud_rgb = cloud_colorize(predicted_frame_cloud)

    local_rmse = _local_rmse(
        pred_ndvi,
        target_ndvi,
        valid_mask=target_valid,
    )

    # One horizontal PNG line:
    # condition | predicted | target | residual | cloud mask
    blocks = [
        _label_row(_strip(condition_rgb), "Condition NDVI"),
        _label_row(_strip(pred_rgb), "Predicted NDVI"),
        _label_row(_strip(target_rgb), "Target NDVI"),
        _label_row(_strip(residual_rgb), "Residual"),
        _label_row(_strip(cloud_rgb), "Cloud mask"),
    ]

    grid = _hcat_blocks(blocks, block_pad=10)

    return grid, local_rmse

def log_or_save_ndvi_residual_viz(
    tensorboard_logger,
    batch: dict,
    preds: torch.Tensor,
    batch_idx: int,
    current_epoch: int,
    context_length: int,
    target_length: int,
    setting: str,
    log_dir: Optional[Path] = None,
    scores: Optional[list[dict]] = None,
    save_to_disk: bool = True,
    log_to_tensorboard: bool = True,
    residual_scale: float = 0.5,
    vmin: Optional[float] = -1,
    vmax: Optional[float] = 1,
    split_name: str = "test",
) -> None:
    """
    Main function to call from validation_step or test_step.

    It creates one visualization per sample and optionally:
        - saves it as PNG locally
        - logs it to TensorBoard

    Output directory:
        <log_dir>/visualizations/<split_name>/epoch_XXXX/
    """
    if setting not in ["greenearthnet", "replicate"]:
        return

    if preds.ndim != 5:
        return

    bsz = preds.shape[0]
    n_vis = bsz

    out_dir = None
    if save_to_disk and log_dir is not None:
        out_dir = Path(log_dir) / "visualizations" / split_name / f"epoch_{int(current_epoch):04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(n_vis):
        score = None
        if scores is not None and sample_idx < len(scores):
            score = scores[sample_idx]

        grid, local_rmse = make_ndvi_residual_grid(
            batch=batch,
            preds=preds,
            sample_idx=sample_idx,
            context_length=context_length,
            target_length=target_length,
            setting=setting,
            score=score,
            residual_scale=residual_scale,
            vmin=vmin,
            vmax=vmax
        )

        if "cubename" in batch:
            name = str(batch["cubename"][sample_idx])
        elif "filepath" in batch:
            name = Path(str(batch["filepath"][sample_idx])).stem
        else:
            name = f"sample_{sample_idx}"

        safe_name = _safe_name(name)

        tag = f"{split_name}/ndvi_residual/batch_{batch_idx:04d}_sample_{sample_idx:02d}_{safe_name}"

        if log_to_tensorboard and tensorboard_logger is not None:
            tensorboard_logger.add_image(tag, grid, current_epoch)

        if out_dir is not None:
            out_path = (out_dir / f"batch_{batch_idx:04d}_sample_{sample_idx:02d}_localRMSE-{local_rmse:.4f}_{safe_name}.png")
            torchvision.utils.save_image(grid.detach().cpu(), str(out_path))