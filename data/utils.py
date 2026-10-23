import random
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def plot_image(im_t, im_name, is_cloud=False):
    if im_t.ndim == 3:
        im_t = im_t.unsqueeze(1).float()
    
    assert im_t.ndim == 4, "the dimension of the tensor is not correct"
    
    if is_cloud:
        im_t = im_t.clamp(0,1)
        im_t = im_t.permute(1, 0, 2, 3)
    else:
        im_t = (im_t + 1) / 2.0
        im_t = im_t.permute(1, 0, 2, 3)
        im_t = im_t.clamp(0,1)

    grid = make_grid(im_t, nrow=5)
    img = transforms.ToPILImage()(grid)
    img.save(f"./{im_name}.png")

    print("Image saved!")


@dataclass
class SpatialAugConfig:
    out_size: Tuple[int, int] = (128, 384)   # final H, W
    crop_scale: Tuple[float, float] = (0.8, 1.0)
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    do_resize_back: bool = True


class TemporalSpatialAugment:
    """
    Apply the SAME spatial transform to all frames in a sequence.

    Expected tensor shape per sample:
        video / spectral:  [C, T, H, W]
        cloud_mask:        [1 or C, T, H, W]

    Can also be adapted to batched tensors, but dataset-level per-sample use is simplest.
    """

    def __init__(self, cfg: SpatialAugConfig):
        self.cfg = cfg

    def _sample_crop_params(self, h: int, w: int):
        scale = random.uniform(*self.cfg.crop_scale)
        crop_h = max(1, int(round(h * scale)))
        crop_w = max(1, int(round(w * scale)))

        if crop_h > h:
            crop_h = h
        if crop_w > w:
            crop_w = w

        top = 0 if crop_h == h else random.randint(0, h - crop_h)
        left = 0 if crop_w == w else random.randint(0, w - crop_w)

        do_hflip = random.random() < self.cfg.hflip_p
        do_vflip = random.random() < self.cfg.vflip_p

        return {
            "top": top,
            "left": left,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "do_hflip": do_hflip,
            "do_vflip": do_vflip,
        }

    @staticmethod
    def _apply_same_spatial_transform(
        x: torch.Tensor,
        params: Dict,
        out_size: Tuple[int, int],
        interpolation: InterpolationMode,
        do_resize_back: bool,
    ) -> torch.Tensor:
        """
        x: [C, T, H, W]
        Applies same spatial transform to all T.
        """
        assert x.ndim == 4, f"Expected [C,T,H,W], got {tuple(x.shape)}"
        c, t, h, w = x.shape

        # [C,T,H,W] -> [T,C,H,W]
        x = x.permute(1, 0, 2, 3)

        # crop
        x = x[
            :,
            :,
            params["top"]: params["top"] + params["crop_h"],
            params["left"]: params["left"] + params["crop_w"],
        ]

        # flips
        if params["do_hflip"]:
            x = torch.flip(x, dims=[3])  # W
        if params["do_vflip"]:
            x = torch.flip(x, dims=[2])  # H

        # resize back
        if do_resize_back:
            x = F.interpolate(
                x,
                size=out_size,
                mode="nearest" if interpolation == InterpolationMode.NEAREST else "bilinear",
                align_corners=False if interpolation != InterpolationMode.NEAREST else None,
            )

        # [T,C,H,W] -> [C,T,H,W]
        x = x.permute(1, 0, 2, 3)
        return x

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        sample keys expected:
            - video: [C,T,H,W]
            - rgb/ndvi/ndwi: [C,T,H,W] (optional)
            - cloud_mask: [C,T,H,W] or [1,T,H,W] (optional)

        Uses bilinear for continuous images, nearest for masks.
        """
        video = sample["video"]
        assert video.ndim == 4, f"Expected video [C,T,H,W], got {tuple(video.shape)}"
        _, _, h, w = video.shape

        params = self._sample_crop_params(h, w)

        out = dict(sample)

        if "video" in out:
            out["video"] = self._apply_same_spatial_transform(
                out["video"],
                params,
                self.cfg.out_size,
                interpolation=InterpolationMode.BILINEAR,
                do_resize_back=self.cfg.do_resize_back,
            )

        for key in ("rgb", "ndvi", "ndwi"):
            if key in out:
                out[key] = self._apply_same_spatial_transform(
                    out[key],
                    params,
                    self.cfg.out_size,
                    interpolation=InterpolationMode.BILINEAR,
                    do_resize_back=self.cfg.do_resize_back,
                )

        if "cloud_mask" in out:
            out["cloud_mask"] = self._apply_same_spatial_transform(
                out["cloud_mask"],
                params,
                self.cfg.out_size,
                interpolation=InterpolationMode.NEAREST,
                do_resize_back=self.cfg.do_resize_back,
            )

        return out
