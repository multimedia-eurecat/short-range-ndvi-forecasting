from __future__ import annotations
from typing import Optional
import torch


def make_static_placeholders(
    h: int,
    w: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EarthNet2021X normally provides static variables such as DEM and landcover.
    Our data does not necessarily have these. The function use a valid 
    vegetation-like class so lc_min/lc_max masking does not remove all pixels.
    """
    staticarr = torch.zeros(2, h, w, dtype=dtype, device=device)

    landcover = torch.full(
        (1, h, w),
        20.0,
        dtype=dtype,
        device=device,
    )

    return staticarr.contiguous(), landcover.contiguous()

def to_earthnet2021x_format(
    sample: dict,
    static: list = None,
    input_mode: str = "ndvi",
    filepath: str = "",
    cubename: str = "",
    fp16: bool = False,
    mask_invalid_value: float = 1.0,
) -> dict:
    """
    Converts an internal sample to the EarthNet2021X data format.

    Expected sample fields:
        sample["rgb"]        : [3, T, H, W]
        sample["ndvi"]       : [1, T, H, W]
        sample["ndwi"]       : [1, T, H, W]
        sample["cloud_mask"] : [1, T, H, W]
        sample["meteo"]      : [T, C_meteo]

    The returned dynamic[0] shape is [T, C, H, W].
    """
    out_dtype = torch.float16 if fp16 else torch.float32

    spectral_tensors = {
        key: sample[key].float()
        for key in ("rgb", "ndvi", "ndwi")
    }
    cloud = sample["cloud_mask"].float()
    meteo = sample["meteo"].float()

    expected_channels = {"rgb": 3, "ndvi": 1, "ndwi": 1}
    for key, tensor in spectral_tensors.items():
        if tensor.ndim != 4 or tensor.shape[0] != expected_channels[key]:
            raise ValueError(
                f"Expected sample['{key}'] with shape "
                f"[{expected_channels[key]},T,H,W], got {tuple(tensor.shape)}"
            )
    if cloud.ndim != 4:
        raise ValueError(f"Expected sample['cloud_mask'] with shape [1,T,H,W], got {tuple(cloud.shape)}")
    if meteo.ndim != 2:
        raise ValueError(f"Expected sample['meteo'] with shape [T,C_meteo], got {tuple(meteo.shape)}")

    cloud = cloud[:1]

    if input_mode.endswith("_cloud_nextcloud"):
        spectral_mode = input_mode.removesuffix("_cloud_nextcloud")
        cloud_mode = "cloud_nextcloud"
    elif input_mode.endswith("_cloud"):
        spectral_mode = input_mode.removesuffix("_cloud")
        cloud_mode = "cloud"
    else:
        spectral_mode = input_mode
        cloud_mode = "none"

    if spectral_mode == "rgb":
        sen2 = spectral_tensors["rgb"]
    elif spectral_mode == "ndvi":
        sen2 = spectral_tensors["ndvi"]
    elif spectral_mode == "rgb_ndvi":
        sen2 = torch.cat(
            [
                spectral_tensors["rgb"],
                spectral_tensors["ndvi"],
            ],
            dim=0,
        )
    elif spectral_mode == "ndvi_ndwi":
        sen2 = torch.cat(
            [
                spectral_tensors["ndvi"],
                spectral_tensors["ndwi"],
            ],
            dim=0,
        )
    elif spectral_mode == "rgb_ndvi_ndwi":
        sen2 = torch.cat(
            [
                spectral_tensors["rgb"],
                spectral_tensors["ndvi"],
                spectral_tensors["ndwi"],
            ],
            dim=0,
        )
    else:
        raise ValueError(
            "input_mode spectral base must be one of: "
            "'rgb', 'ndvi', 'ndvi_ndwi'; optionally followed by "
            "'_cloud' or '_cloud_nextcloud'"
        )

    ndvi = spectral_tensors["ndvi"]

    # Build next-cloud/future-cloud channel: at time t, use cloud at t+1.
    next_cloud = torch.zeros_like(cloud)
    if cloud.shape[1] > 1:
        next_cloud[:, :-1] = cloud[:, 1:]

    if cloud_mode == "cloud":
        sen2 = torch.cat([sen2, cloud], dim=0)
        ndvi = torch.cat([ndvi, cloud], dim=0)
    elif cloud_mode == "cloud_nextcloud":
        sen2 = torch.cat([sen2, cloud, next_cloud], dim=0)
        ndvi = torch.cat([ndvi, cloud, next_cloud], dim=0)

    # Internal format [C,T,H,W] -> EarthNet2021X format [T,C,H,W].
    sen2arr = sen2.permute(1, 0, 2, 3).contiguous()
    ndvi2arr = ndvi.permute(1, 0, 2, 3).contiguous()

    # dynamic_mask remains separate from the input channels.
    # Convert cloud probability/binary cloud into a binary invalid mask.
    sen2mask = (cloud > 0.5).float() * mask_invalid_value
    sen2mask = sen2mask.permute(1, 0, 2, 3).contiguous()

    # sen2arr = pad_or_trim_time(sen2arr, seq_len, dim=0)
    # sen2mask = pad_or_trim_time(sen2mask, seq_len, dim=0)
    # meteo = pad_or_trim_time(meteo, seq_len, dim=0)
    
    if static is None:
        _, _, h, w = sen2arr.shape
        staticarr, landcover = make_static_placeholders(
            h=h,
            w=w,
            dtype=out_dtype,
            device=sen2arr.device,
        )
    else:
        landcover, elevation, geomorpho = static

        elevation = torch.as_tensor(elevation, dtype=out_dtype, device=sen2arr.device)
        geomorpho = torch.as_tensor(geomorpho, dtype=out_dtype, device=sen2arr.device)
        landcover = torch.as_tensor(landcover, dtype=out_dtype, device=sen2arr.device)

        if elevation.ndim == 2:
            elevation = elevation.unsqueeze(0)
        if geomorpho.ndim == 2:
            geomorpho = geomorpho.unsqueeze(0)
        if landcover.ndim == 2:
            landcover = landcover.unsqueeze(0)

        elevation = elevation / 2000.0

        staticarr = torch.cat([elevation, geomorpho], dim=0).contiguous()
        landcover = landcover.contiguous()

    return {
        "dynamic": [
            torch.nan_to_num(sen2arr.to(out_dtype), nan=0.0, posinf=1.0, neginf=-1.0),
            torch.nan_to_num(meteo.to(out_dtype), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(ndvi2arr.to(out_dtype), nan=0.0, posinf=1.0, neginf=-1.0),
        ],
        "dynamic_mask": [
            sen2mask.to(out_dtype),
        ], 
        "static": [
            torch.nan_to_num(staticarr.to(out_dtype), nan=0.0, posinf=0.0, neginf=0.0).contiguous(),
        ],
        "static_mask": [],
        "landcover": landcover.to(out_dtype),
        "filepath": str(filepath),
        "cubename": str(cubename),
        "time": sample.get("time", None),
    }
