import calendar
import glob
import math
import os

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .earthnet_adapter import to_earthnet2021x_format   # TODO per debug togli il .
from .utils import TemporalSpatialAugment, SpatialAugConfig, plot_image # TODO per debug togli il .
from .utils_replicate import ( # TODO per debug togli il .
    GencatWeatherMeasures,
    get_YYYYmmdd,
    load_json_as_dict,
    replicate_numerical_metadata,
    replicate_weather_metadata,
)


DATA_PATH = ".../path/to/replicate_full_sentinel_bands/images_aligned_128"
CLOUD_PROB_PATH = ".../path/to/replicate_full_sentinel_bands/clouds_aligned_128"
WEATHER_DATA_PATH = ".../path/to/gencat_historical_data/gencat"
BASE_JSON = ".../path/to/base.json"

# Band order written by sentinel2_download_pipeline-main/final_pipeline.
_BAND_ORDER = (
    "B01", # COSTAL AEREOSOL
    "B02", # BLUE
    "B03", # GREEN
    "B04", # RED
    "B05", # RED EDGE1
    "B08", # NIR
    "B8A", # NARROW NIR
    "B09", # WATER VAPOR
    "B10", # CIRRUS
    "B11", # SWIR1
    "B12", # SWIR2
)
_BAND_INDICES = {band: index for index, band in enumerate(_BAND_ORDER)}


def safe_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Safely divide tensors, returning zero for invalid pixels."""
    numerator = numerator.to(torch.float32)
    denominator = denominator.to(torch.float32)

    valid = (
        torch.isfinite(numerator)
        & torch.isfinite(denominator)
        & (denominator.abs() > eps)
    )

    # Avoid performing division by zero, even in the masked branch.
    safe_denominator = torch.where(
        valid,
        denominator,
        torch.ones_like(denominator),
    )

    result = numerator / safe_denominator

    return torch.where(
        valid,
        result,
        torch.zeros_like(result),
    )


def safe_normalized_difference(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calculate (first - second) / (first + second)."""
    result = safe_ratio(
        numerator=first - second,
        denominator=first + second,
        eps=eps,
    )

    return result.clamp(-1.0, 1.0)


def spectral_metrics(multispectral: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Calculate RGB, NDVI, and NDWI from an 11-band Sentinel-2 tensor.

    Input shape is [11,H,W]. Outputs keep a channel dimension:
        rgb [3,H,W], ndvi/ndwi [1,H,W].
    """
    if multispectral.ndim != 3 or multispectral.shape[0] != len(_BAND_ORDER):
        raise ValueError(
            f"Expected multispectral tensor [{len(_BAND_ORDER)},H,W], got "
            f"{tuple(multispectral.shape)}"
        )

    blue = multispectral[_BAND_INDICES["B02"]]
    green = multispectral[_BAND_INDICES["B03"]]
    red = multispectral[_BAND_INDICES["B04"]]
    nir = multispectral[_BAND_INDICES["B08"]]
    swir1 = multispectral[_BAND_INDICES["B11"]]

    rgb = torch.stack([red, green, blue], dim=0)
    rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    return {
        "rgb": rgb,
        "ndvi": safe_normalized_difference(nir, red).unsqueeze(0),
        "ndwi": safe_normalized_difference(nir, swir1).unsqueeze(0), # moisture idex
    }

def date_features_from_acquired(acquired, base_year=2016, end_year=2030, scale=1.0):
    date = pd.Timestamp(str(acquired)).date()
    year = date.year
    month = date.month
    days_in_year = 366 if calendar.isleap(year) else 365
    doy = date.timetuple().tm_yday

    angle_year = 2 * math.pi * (doy - 1) / days_in_year
    sin_doy = math.sin(angle_year)
    cos_doy = math.cos(angle_year)

    angle_month = 2 * math.pi * (month - 1) / 12
    sin_month = math.sin(angle_month)
    cos_month = math.cos(angle_month)

    year_trend = (year - base_year) / (end_year - base_year)

    return torch.tensor(
        [
            sin_doy * scale,
            cos_doy * scale,
            sin_month * scale,
            cos_month * scale,
            year_trend * scale,
        ],
        dtype=torch.float32,
    )


def horizon_feature(acquired, anchor_date, max_horizon_days=60, scale=1.0):
    acquired = pd.Timestamp(str(acquired)).date()
    anchor_date = pd.Timestamp(str(anchor_date)).date()
    delta_days = (acquired - anchor_date).days

    sign = 1.0 if delta_days >= 0 else -1.0
    norm_delta = sign * math.log1p(abs(delta_days)) / math.log1p(max_horizon_days)
    return torch.tensor([norm_delta * scale], dtype=torch.float32)


def build_conditioning_metadata(
    raw_metadata,
    acquired,
    anchor_date=None,
    base_year=2016,
    end_year=2030,
    max_horizon_days=60,
    scale=1.0,
):
    # raw_metadata is kept in the signature so you can re-add lon/lat/gsd/cloud_cover later.
    del raw_metadata

    features = [
        date_features_from_acquired(
            acquired,
            base_year=base_year,
            end_year=end_year,
            scale=scale,
        )
    ]

    if anchor_date is not None:
        features.append(
            horizon_feature(
                acquired,
                anchor_date,
                max_horizon_days=max_horizon_days,
                scale=scale,
            )
        )

    return torch.cat(features, dim=0)


class ReplicateDataset(Dataset):
    def __init__(
        self,
        img_data_path,
        weather_data_path=None,
        clouds_data_path=None,
        spatial_size=(128, 384),
        mode=None,
        sequence_len=4,
        context_len=None,
        no_overlap=False,
        stride=4,
        start_sequence_from=0,
        do_augmentations=False,
        return_format="diffusionsat",
        input_mode="ndvi_cloud_nextcloud",
        fp16=False,
        mask_invalid_value=1.0,
    ):
        """
        return_format:
            "diffusionsat"   -> text/video/rgb/ndvi/ndwi/cloud_mask/meteo
            "earthnet2021x"  -> EarthNet2021X benchmark output: dynamic/dynamic_mask/static/landcover

        input_mode is used only when return_format="earthnet2021x":
            spectral base: "rgb", "ndvi", or "ndvi_ndwi"
            optional suffix: "_cloud" or "_cloud_nextcloud"
        """
        self.DATA_PATH = img_data_path
        self.mode = mode
        self.sequence_len = int(sequence_len)
        self.context_len = self.sequence_len - 1 if context_len is None else int(context_len)
        if not 1 <= self.context_len < self.sequence_len:
            raise ValueError(
                f"context_len must be between 1 and sequence_len - 1; got "
                f"context_len={self.context_len}, sequence_len={self.sequence_len}"
            )
        self.no_overlap = no_overlap
        self.stride = int(stride)
        self.do_augm = do_augmentations
        self.return_format = return_format
        self.input_mode = input_mode
        self.fp16 = fp16
        self.mask_invalid_value = mask_invalid_value

        assert os.path.exists(self.DATA_PATH), f"Image path does not exist: {self.DATA_PATH}"

        all_ims = sorted(glob.glob(os.path.join(self.DATA_PATH, "*.tif")))
        assert len(all_ims) > 0, f"No .tif files found in {self.DATA_PATH}"

        self.use_cloud_probs = clouds_data_path is not None and len(str(clouds_data_path)) > 0
        self.clouds_data_path = clouds_data_path if self.use_cloud_probs else None

        if self.use_cloud_probs:
            all_cld = sorted(glob.glob(os.path.join(self.clouds_data_path, "*.tif")))
        else:
            all_cld = []

        all_samples = [
            {
                "img": p,
                "acquired": get_YYYYmmdd(os.path.basename(p).split("_")[1][:8]),
            }
            for p in all_ims
        ]
        self.all_clouds = [
            {
                "cld": p,
                "acquired": get_YYYYmmdd(os.path.basename(p).split("_")[1][:8]),
            }
            for p in all_cld
        ]

        all_samples = sorted(all_samples, key=lambda x: x["acquired"])
        
        
        if self.mode is not None:
            split_80_20 = int(len(all_samples) * 0.80)
            if self.mode == "train":
                all_samples = all_samples[:split_80_20] # if you start from 21 you remove 2016 which has more incosistent revisit interval
            else:
                all_samples = all_samples[split_80_20:]

        assert len(all_samples) > 0, "No samples left after train/val split. Check your data size."

        self.start_date = all_samples[0]["acquired"]
        self.end_date = all_samples[-1]["acquired"]
        self.all_samples = all_samples
        self.truncated_samples = all_samples[start_sequence_from:]

        self.ALL_SAMPLES = {}
        self.ALL_SAMPLES[self.all_samples[0]["img"]] = {
            "acquired": self.all_samples[0]["acquired"],
            "previous": self.all_samples[0],
        }
        for i, el in enumerate(self.all_samples[1:]):
            self.ALL_SAMPLES[el["img"]] = {
                "acquired": el["acquired"],
                "previous": self.all_samples[i],
            }

        self.use_weather = weather_data_path is not None and len(str(weather_data_path)) > 0
        if self.use_weather:
            self.gencat_weather = GencatWeatherMeasures(
                weather_data_path,
                start_day=self.start_date,
                end_day=self.end_date,
            )
            self.meteo_dim = 10
        else:
            self.meteo_dim = 6

        self.T = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(spatial_size),
            ]
        )

        self.aug = TemporalSpatialAugment(
            SpatialAugConfig(
                out_size=spatial_size,
                crop_scale=(0.4, 1.0),
                hflip_p=0.5,
                vflip_p=0.5,
                do_resize_back=True,
            )
        )

    def read_image_from_key(self, img_key, json=None, anchor_date=None):
        image = tiff.imread(img_key)
        if image.ndim != 3:
            raise ValueError(
                f"Expected a 3D multispectral TIFF at {img_key}, got shape {image.shape}"
            )
        if image.shape[-1] == len(_BAND_ORDER):
            image_hwc = image
        elif image.shape[0] == len(_BAND_ORDER):
            image_hwc = np.moveaxis(image, 0, -1)
        else:
            raise ValueError(
                f"Expected {len(_BAND_ORDER)} Sentinel-2 bands at {img_key}, "
                f"got shape {image.shape}"
            )
        px_vals = self.T(image_hwc).float()

        if json is None:
            json_md_path = img_key.replace(".tif", ".json")
            json_md_dict = load_json_as_dict(json_md_path)
        else:
            json_md_dict = json

        acquired = json_md_dict["properties"]["acquired"]
        query_date = pd.Timestamp(str(acquired)).strftime("%Y-%m-%d")

        raw_metadata = replicate_numerical_metadata(json_md_dict, gsd_scaling=1)
        metadata = build_conditioning_metadata(
            raw_metadata,
            acquired=acquired,
            anchor_date=anchor_date,
            base_year=2016,
            end_year=2030,
            max_horizon_days=60,
            scale=1.0,
        )

        if self.use_weather:
            weather_md_dict = self.gencat_weather.get_weather_at_input_date(query_date)
            weather_cond_md = replicate_weather_metadata(
                weather_md_dict,
                regularize=False,
                random=False,
            )
            metadata = torch.cat([metadata, weather_cond_md], dim=0)

        if self.use_cloud_probs:
            cloud_probs_img = self.read_cloud_probs_from_key(img_key, query_date)
            return px_vals, metadata, cloud_probs_img.float()

        _, height, width = px_vals.shape
        return (
            px_vals,
            metadata,
            torch.zeros((1, height, width), dtype=px_vals.dtype),
        )

    def read_cond_ims_and_metadata(self, img_key, num_cond=4, json=None):
        """
        Returns chronological sequence:
            px_vals     -> [11, T, H, W]
            metadata    -> [T, C_meteo]
            cloud_probs -> [1, T, H, W]
        """
        img_id = img_key
        entries = []

        entries.append(
            {
                "img": img_id,
                "acquired": self.ALL_SAMPLES[img_id]["acquired"],
            }
        )

        for _ in range(num_cond - 1):
            prev_entry = self.ALL_SAMPLES[img_id]["previous"]
            prev_img_id = prev_entry["img"]
            prev_acquired = prev_entry["acquired"]

            entries.append(
                {
                    "img": prev_img_id,
                    "acquired": prev_acquired,
                }
            )
            img_id = prev_img_id

        entries = entries[::-1]
        # The forecast is issued at the final context frame. Using an explicit
        # context length also keeps the horizon correct for multi-frame targets.
        anchor_date = entries[self.context_len - 1]["acquired"]

        img_ids_list = []
        px_vals_list = []
        metadata_list = []
        clouds_probs_list = []
        acquired_list = []

        for entry in entries:
            curr_img_id = entry["img"]
            curr_acquired = entry["acquired"]

            if json is not None:
                json["properties"]["acquired"] = curr_acquired

            px_vals, metadata, cloud_probs = self.read_image_from_key(
                curr_img_id,
                json,
                anchor_date=anchor_date,
            )

            img_ids_list.append(curr_img_id)
            px_vals_list.append(px_vals)
            metadata_list.append(metadata)
            clouds_probs_list.append(cloud_probs)
            acquired_list.append(str(pd.Timestamp(str(curr_acquired)).date()))

        px_vals = torch.stack(px_vals_list, dim=1)
        metadata = torch.stack(metadata_list)
        cloud_probs = torch.stack(clouds_probs_list, dim=1)

        return img_ids_list, px_vals, metadata, cloud_probs, acquired_list

    def read_cloud_probs_from_key(self, img_key, date=None):
        if date is None:
            date_str = os.path.basename(img_key).split(".")[0].split("_")[-1]
            cloud_probs_img_path = os.path.join(self.clouds_data_path, f"clouds_{date_str}.tif")
            cloud_probs_img = np.array(Image.open(cloud_probs_img_path))
            return self.T(cloud_probs_img).float()

        try:
            cld_path = next(item["cld"] for item in self.all_clouds if item["acquired"] == date)
        except StopIteration as exc:
            raise FileNotFoundError(f"No cloud probability file found for date {date}") from exc

        return self.T(tiff.imread(cld_path)).float()

    def __len__(self):
        if self.no_overlap:
            return (len(self.truncated_samples) + self.stride - 1) // self.stride
        return len(self.truncated_samples)

    def __getitem__(self, idx):
        idx = idx * self.stride if self.no_overlap else idx
        img_key = self.truncated_samples[idx]["img"]

        tuned_json = load_json_as_dict(BASE_JSON)
        tuned_json["properties"]["acquired"] = self.truncated_samples[idx]["acquired"]

        img_id, px_vals, metadata, clouds_probs, time_values = self.read_cond_ims_and_metadata(
            img_key,
            self.sequence_len,
            tuned_json,
        )

        clouds_probs = clouds_probs[:1].float()
        clouds_probs = torch.nan_to_num(
            clouds_probs,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        per_frame_metrics = [
            spectral_metrics(px_vals[:, frame_index])
            for frame_index in range(px_vals.shape[1])
        ]
        rgb = torch.stack([metrics["rgb"] for metrics in per_frame_metrics], dim=1)
        ndvi = torch.stack([metrics["ndvi"] for metrics in per_frame_metrics], dim=1)
        ndwi = torch.stack([metrics["ndwi"] for metrics in per_frame_metrics], dim=1)

        # Dataset video contains spectral information only. Cloud feature
        # composition is intentionally delegated to earthnet_adapter.py.
        video = torch.cat([rgb, ndvi, ndwi], dim=0)

        sample = {
            "text": img_id,
            "video": video,
            "rgb": rgb,
            "ndvi": ndvi,
            "ndwi": ndwi,
            "actual_length": video.shape[1],
            "cloud_mask": clouds_probs,
            "meteo": metadata,
            "time": time_values,
        }

        if self.mode == "train" and self.do_augm:
            sample = self.aug(sample)

        if self.return_format == "earthnet2021x":
            return to_earthnet2021x_format(
                sample,
                None,
                input_mode=self.input_mode,
                filepath=img_key,
                cubename=os.path.basename(img_key),
                fp16=self.fp16,
                mask_invalid_value=self.mask_invalid_value,
            )

        return sample


if __name__ == "__main__":
    df = ReplicateDataset(
        DATA_PATH,
        weather_data_path=WEATHER_DATA_PATH,
        clouds_data_path=CLOUD_PROB_PATH,
        sequence_len=4,
        no_overlap=True,
        stride=2,
        start_sequence_from=4,
        return_format="earthnet2021x",
        input_mode="ndvi_ndwi_cloud_nextcloud",
    )
    sample = df[0]
    print(sample["dynamic"][0].shape)       # [T, 4, H, W]
    print(sample["dynamic"][1].shape)       # [T, C_meteo]
    print(sample["dynamic_mask"][0].shape)  # [T, 1, H, W]
    # plot_image(im_t, im_name, is_cloud=False) for visualization
