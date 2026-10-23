# Adapted from:
# https://github.com/earthnet2021/earthnet-models-pytorch
#
# Original implementation licensed under the MIT License.
#
# Modified by Cynthia Ugwu (@cugwu) for:
# https://github.com/multimedia-eurecat/short-range-ndvi-forecasting
#
# Changes include adaptations to input channels,
# prediction outputs, and training configurations.

import sys

import torch
import torch.distributions as distrib
import torch.nn.functional as F
from torch import nn


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=-1, eps=1e-8):
    """
    Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int
        Dimensions of raw_params so that the first half corresponds to the mean, and the second half to the scale.
    eps : float
        Minimum possible value of the final scale parameter.

    Returns
    -------
    torch.distributions.Normal
        Normal distributions with the input mean and eps + softplus(raw scale) * scale_stddev as scale.
    """
    loc, raw_scale = torch.chunk(raw_params, 2, dim)
    assert loc.shape[dim] == raw_scale.shape[dim]
    scale = F.softplus(raw_scale) + eps
    normal = distrib.Normal(loc, scale * scale_stddev)
    return normal


class MaskedLoss(nn.Module):
    def __init__(self, distance_type="L2"):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, preds, targets, mask):
        assert preds.shape == targets.shape
        if mask.ndim != preds.ndim:
            raise RuntimeError(
                f"Mask rank {mask.ndim} does not match prediction rank "
                f"{preds.ndim}: mask={mask.shape}, predictions={preds.shape}."
            )
        try:
            mask = mask.expand_as(preds)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Mask shape {mask.shape} cannot be expanded to prediction "
                f"shape {preds.shape}."
            ) from exc
        predsmasked = preds * mask
        targetsmasked = targets * mask

        if self.distance_type == "L2":
            return F.mse_loss(predsmasked, targetsmasked, reduction="sum") / mask.sum().clamp_min(1.0)  
        elif self.distance_type == "L1":
            return F.l1_loss(predsmasked, targetsmasked, reduction="sum") / mask.sum().clamp_min(1.0)
        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}. Only 'L2' and 'L1' are accepted")

class Loss(nn.Module):
    def __init__(self, distance_type="L2"):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, preds, targets, mask):
        assert preds.shape == targets.shape

        if self.distance_type == "L2":
            return F.mse_loss(preds, targets) 
        elif self.distance_type == "L1":
            return F.l1_loss(preds, targets)


LOSSES = {"masked-land": MaskedLoss, "unmasked-land": Loss}


class PixelwiseLoss(nn.Module):
    def __init__(self, setting: dict):
        super().__init__()

        self.distance = LOSSES[setting["name"]](setting["distance_type"])
        self.lc_min = 10 if "lc_min" not in setting else setting["lc_min"]
        self.lc_max = 104 if "lc_max" not in setting else setting["lc_max"]
        print(
            f"Using Masked L2 NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
        )

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}

        targs = batch["dynamic"][2][
            :, -preds.shape[1] :, ...
        ]  # the number of sample to predict

        lc = batch["landcover"]

        masks = (
            ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool())
            .type_as(preds)
            .unsqueeze(1)
            .repeat(1, preds.shape[1], 1, 1, 1)
        )  # mask for outlayers using lc threshold

        # masks = torch.where(masks.bool(), (preds >= 0).type_as(masks), masks)

        loss = self.distance(preds, targs, masks)

        logs["loss"] = loss
        return loss, logs


class MaskedPixelwiseLoss(nn.Module):
    def __init__(
        self,
        lc_min=None,
        lc_max=None,
        context_length=None,
        target_length=None,
        pred_mask_value=None,
        scale_by_std=False,
        extra_aux_loss_term=None,
        extra_aux_loss_weight=1,
        distance_type='L2',
        **kwargs,
    ):
        super().__init__()

        self.lc_min = lc_min
        self.lc_max = lc_max
        self.use_lc = lc_min is not None and lc_max is not None
        if not self.use_lc:
            print(
                f"WARNING. The boundaries of the landcover map are not definite. Loss calculated on all pixels including non-vegetation pixels."
            )
        self.context_length = context_length
        self.target_length = target_length
        self.pred_mask_value = pred_mask_value
        self.scale_by_std = scale_by_std
        if self.scale_by_std:
            print(
                f"Using Masked L2/Std NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )
        else:
            print(
                f"Using Masked L2 NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )

        self.extra_aux_loss_term = extra_aux_loss_term
        self.extra_aux_loss_weight = extra_aux_loss_weight

        self.distance = MaskedLoss(distance_type=distance_type)

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}
        # Targets
        targets = batch["dynamic"][2][
            :, self.context_length : self.context_length + self.target_length, :, ...
        ]

        # Masks on the non vegetation pixels
        lc = batch["landcover"]
        lc_mask = (
            ((lc >= self.lc_min).bool() & (lc <= self.lc_max).bool())
            .type_as(preds)
            .unsqueeze(1)
            .repeat(1, preds.shape[1], 1, 1, 1)
        )
        
        # Dynamic cloud mask available
        if len(batch["dynamic_mask"]) > 0:
            s2_mask = batch["dynamic_mask"][0][
                :,
                self.context_length : self.context_length + self.target_length,
                ...,
            ]
            # Support both legacy [B,T,H,W] masks and the current adapter's
            # explicit-channel [B,T,1,H,W] masks.
            if s2_mask.ndim == 4:
                s2_mask = s2_mask.unsqueeze(2)
            if s2_mask.ndim != 5:
                raise RuntimeError(
                    "Expected dynamic cloud mask [B,T,H,W] or [B,T,1,H,W], "
                    f"got {s2_mask.shape}."
                )
            s2_mask = (s2_mask < 1.0).type_as(preds)
            mask = s2_mask * lc_mask    
        else:
            mask = lc_mask

        # loss computation
        loss = self.distance(preds, targets, mask)
        logs["loss"] = loss
        return loss, logs


class L2NDVILoss(nn.Module):
    def __init__(
        self,
        lc_min=None,
        lc_max=None,
        context_length=None,
        target_length=None,
        n_pred_channels=1,
        n_targ_channels=1,
        scale_by_std=False,
        weight_by_std=False,
        use_lc_mask=False,
        use_cloud_mask=False,
        extra_aux_loss_term=None,
        extra_aux_loss_weight=1,
        mask_hq_only=False,
        **kwargs,
    ):
        super().__init__()

        self.lc_min = lc_min
        self.lc_max = lc_max
        self.use_lc = lc_min is not None and lc_max is not None
        if not self.use_lc:
            print(
                f"WARNING. The boundaries of the landcover map are not definite. Loss calculated on all pixels including non-vegetation pixels."
            )
        self.context_length = context_length
        self.target_length = target_length
        self.n_pred_channels = n_pred_channels  # index of the NDVI band
        self.n_targ_channels = n_targ_channels  # index of the NDVI band
        self.scale_by_std = scale_by_std
        self.weight_by_std = weight_by_std
        self.use_lc_mask = use_lc_mask
        self.use_cloud_mask = use_cloud_mask
        if self.scale_by_std:
            print(
                f"Using Masked L2/Std NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )
        else:
            print(
                f"Using Masked L2 NDVI Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
            )

        self.extra_aux_loss_term = extra_aux_loss_term
        self.extra_aux_loss_weight = extra_aux_loss_weight
        self.mask_hq_only = mask_hq_only

    def forward(self, preds, batch, aux, current_step=None):
        # Target NDVI: [B,T,1,H,W]
        ndvi_targ = batch["dynamic"][2][
            :,
            self.context_length : self.context_length + self.target_length,
            :self.n_targ_channels,
            ...,
            ]

        # Predicted NDVI: [B,T,1,H,W]
        ndvi_pred = preds[
            :,
            -ndvi_targ.shape[1] :,
            :self.n_pred_channels,
            ...,
            ]
        
        if ndvi_targ.shape != ndvi_pred.shape:
            raise RuntimeError(
                f"Prediction shape {ndvi_pred.shape} does not match "
                f"target shape {ndvi_targ.shape}."
            )
        
        # Cloud mask: [B,T,1,H,W]
        # dynamic_mask convention: 0 = valid/clear, 1 = invalid/cloud
        if self.use_cloud_mask and len(batch["dynamic_mask"]) > 0:
            s2_mask = (
                batch["dynamic_mask"][0][
                    :,
                    self.context_length : self.context_length + self.target_length,
                    :1,
                    ...,
                ]
                < 1.0
            ).type_as(preds)
        else:
            s2_mask = torch.ones_like(ndvi_targ)

        # Landcover mask after temporal reduction: [B,1,H,W]
        if self.use_lc_mask and self.use_lc:
            lc = batch["landcover"]
            lc_mask = (
                (lc >= self.lc_min) & (lc <= self.lc_max)
            ).type_as(preds)
            if lc_mask.ndim == 3:
                lc_mask = lc_mask.unsqueeze(1)
            if lc_mask.ndim != 4:
                raise RuntimeError(
                    "Expected landcover mask [B,1,H,W] or [B,H,W], got "
                    f"{lc_mask.shape}."
                )
        else:
            # Do not retain the target-time dimension here: the squared error
            # and cloud mask are reduced over time below.
            lc_mask = torch.ones_like(ndvi_targ[:, 0])

        valid_count = s2_mask.sum(1)  # [B,1,H,W]
        valid_count_safe = valid_count.clamp_min(1.0)

        sum_squared_error = (((ndvi_targ - ndvi_pred) * s2_mask)**2).sum(1) 
        mse = sum_squared_error / valid_count_safe  # [B,1,H,W]

        if self.scale_by_std:
            mean_ndvi_targ = (
                (ndvi_targ * s2_mask).sum(1).unsqueeze(1)
                / valid_count_safe.unsqueeze(1)
            )

            sum_squared_deviation = (
                ((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2
            ).sum(1)

            mse = sum_squared_error / sum_squared_deviation.clamp_min(0.01)

        elif self.weight_by_std:
            mean_ndvi_targ = (
                (ndvi_targ * s2_mask).sum(1).unsqueeze(1)
                / valid_count_safe.unsqueeze(1)
            )

            sum_squared_deviation = (
                ((ndvi_targ - mean_ndvi_targ) * s2_mask) ** 2
            ).sum(1)

            mse = sum_squared_error * (
                ((sum_squared_deviation / valid_count_safe) ** 0.5) / 0.1
            ).clamp(min=0.01, max=100.0)

        # Pixels with no valid target frame should not count as zero-error pixels.
        pixel_valid = (valid_count > 0).type_as(preds)  # [B,1,H,W]

        # Final mask combines cloud-valid pixels and landcover-valid pixels.
        final_mask = pixel_valid * lc_mask  # [B,1,H,W]

        mse_lc = (mse * final_mask).sum() / final_mask.sum().clamp_min(1.0)

        logs = {"loss": mse_lc}

        if self.extra_aux_loss_term:
            extra_loss = aux[self.extra_aux_loss_term]
            logs["mse_lc"] = mse_lc
            logs[self.extra_aux_loss_term] = extra_loss
            mse_lc = mse_lc + self.extra_aux_loss_weight * extra_loss
            logs["loss"] = mse_lc

        return mse_lc, logs


class L2NDVIGradRankLoss(nn.Module):
    def __init__(
            self, 
            lc_min=None,
            lc_max=None,
            context_length=None,
            target_length=None,
            n_pred_channels=1,
            n_targ_channels=1,
            use_lc_mask=False,
            use_cloud_mask=False,
            change_threshold= 0.03,
            gradient_weight=0.1,
            ranking_weight=0.1,
            margin_ratio=0.5,
            **kwargs):
        super().__init__()

        self.lc_min = lc_min
        self.lc_max = lc_max
        self.use_lc = lc_min is not None and lc_max is not None
        if not self.use_lc:
            print(
                f"WARNING. The boundaries of the landcover map are not definite. Loss calculated on all pixels including non-vegetation pixels."
            )
        self.context_length = context_length
        self.target_length = target_length
        self.n_pred_channels = n_pred_channels  # index of the NDVI band
        self.n_targ_channels = n_targ_channels  # index of the NDVI band
        self.use_lc_mask = use_lc_mask
        self.use_cloud_mask = use_cloud_mask
        self.change_threshold = change_threshold
        self.gradient_weight = gradient_weight
        self.ranking_weight = ranking_weight
        self.margin_ratio = margin_ratio

    @staticmethod
    def masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
            
        if mask.shape != values.shape:
            mask = mask.expand_as(values)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def masked_gradient_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shapes:
            prediction: [B, T, C, H, W]
            target:      [B, T, C, H, W]
            valid_mask:  [B, T, 1 or C, H, W]
        """

        # Horizontal gradients
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]

        valid_dx = (
            valid_mask[..., :, 1:]
            * valid_mask[..., :, :-1]
        )

        # Vertical gradients
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]

        valid_dy = (
            valid_mask[..., 1:, :]
            * valid_mask[..., :-1, :]
        )

        loss_x = self.masked_mean(
            torch.abs(pred_dx - target_dx),
            valid_dx,
        )

        loss_y = self.masked_mean(
            torch.abs(pred_dy - target_dy),
            valid_dy,
        )

        return loss_x + loss_y


    def forward(self, preds, batch, aux, current_step=None):
        # ---------------------------------------------------------
        # 1. Extract every supervised output channel
        # ---------------------------------------------------------
        target_all = batch["dynamic"][2][
            :,
            self.context_length:
            self.context_length + self.target_length,
            :self.n_targ_channels,
            ...,
        ]

        prediction_all = preds[
            :,
            -target_all.shape[1]:,
            :self.n_pred_channels,
            ...,
        ]

        if prediction_all.shape != target_all.shape:
            raise RuntimeError(
                f"Prediction shape {prediction_all.shape} does not match "
                f"target shape {target_all.shape}."
            )

        # ---------------------------------------------------------
        # 2. Select NDVI only for NDVI-specific terms
        # ---------------------------------------------------------
        target_ndvi = target_all[:, :, :1]

        prediction_ndvi = prediction_all[:, :, :1]

        last_ndvi = batch["dynamic"][2][:, self.context_length - 1: self.context_length, :1]


        if last_ndvi.shape[1] == 1 and target_ndvi.shape[1] > 1:
            last_ndvi = last_ndvi.expand(
                -1,
                target_ndvi.shape[1],
                -1,
                -1,
                -1,
            )

        # ---------------------------------------------------------
        # 3. NDVI validity mask
        # ---------------------------------------------------------
        # This mask is applied only to NDVI-related losses.
        if self.use_cloud_mask and len(batch["dynamic_mask"]) > 0:
            ndvi_valid_mask = (
                batch["dynamic_mask"][0][
                    :,
                    self.context_length:
                    self.context_length + self.target_length,
                    :1,
                    ...,
                ]
                < 1.0
            ).to(prediction_all.dtype)
        else:
            ndvi_valid_mask = torch.ones_like(target_ndvi)

        if self.use_lc_mask and self.use_lc:
            lc = batch["landcover"]

            lc_mask = (
                (lc >= self.lc_min)
                & (lc <= self.lc_max)
            ).to(prediction_all.dtype)

            while lc_mask.ndim < target_ndvi.ndim:
                lc_mask = lc_mask.unsqueeze(1)

            ndvi_valid_mask = ndvi_valid_mask * lc_mask
        
        
        ############ Loss ###########
        true_change = torch.abs(target_ndvi - last_ndvi)

        pixel_error = (target_all - prediction_all)**2
        # F.smooth_l1_loss(
        #     prediction_all,
        #     target_all,
        #     beta=0.05,
        #     reduction="none",
        # )

        reconstruction_mask = torch.ones_like(target_all)
        if self.use_cloud_mask or self.use_lc_mask:
            reconstruction_mask[:, :, :1] = ndvi_valid_mask

        pixel_loss = self.masked_mean(pixel_error, reconstruction_mask)

        # Preserve spatial boundaries and prevent constant-image collapse.
        gradient_loss = self.masked_gradient_loss(
            prediction_ndvi,
            target_ndvi,
            ndvi_valid_mask,
        )

        # Penalize copying the past only where the future changed.
        changed_mask = (
            true_change > self.change_threshold
        ).to(prediction_ndvi.dtype)

        ranking_mask = ndvi_valid_mask * changed_mask

        distance_to_future = torch.abs(prediction_ndvi - target_ndvi)
        distance_to_past = torch.abs(prediction_ndvi - last_ndvi)

        adaptive_margin = self.margin_ratio * true_change

        ranking_error = F.relu(
            distance_to_future
            - distance_to_past
            + adaptive_margin
        )

        ranking_loss = self.masked_mean(
            ranking_error,
            ranking_mask,
        )

        total_loss = (
            pixel_loss
            + self.gradient_weight * gradient_loss
            + self.ranking_weight * ranking_loss
        )

        logs = {
            "loss": total_loss.detach(),
            "loss_pixel": pixel_loss.detach(),
            "loss_gradient": gradient_loss.detach(),
            "loss_ranking": ranking_loss.detach(),
            "target_change": self.masked_mean(
                true_change,
                ndvi_valid_mask,
            ).detach(),
            "prediction_std": prediction_all.std(unbiased=False).detach(),
            "target_std": target_all.std(unbiased=False).detach(),
            "valid_fraction": ndvi_valid_mask.mean().detach(),
            "changed_fraction": self.masked_mean(
                changed_mask,
                ndvi_valid_mask,
            ).detach(),
        }

        return total_loss, logs


def setup_loss(args):
    if args["name"] == "L2NDVILoss":
        return L2NDVILoss(**args)
    elif args["name"] == "MaskedPixelwiseLoss":
        return MaskedPixelwiseLoss(**args)
    elif args["name"] == "PixelwiseLoss":
        return PixelwiseLoss(**args)
    elif args["name"] in {
        "L2NDVIGradRankLoss",
        "CostumeLoss",
    }:
        return L2NDVIGradRankLoss(**args)
    else:
        raise ValueError(f"Unknown loss: {args['name']}")
