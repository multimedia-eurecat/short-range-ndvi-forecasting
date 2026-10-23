"""
Dataset class to handle GreenEarthNet
Follow https://github.com/vitusbenson/greenearthnet for dataset download
"""
import glob
import os
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .earthnet_adapter import to_earthnet2021x_format 
from .utils import TemporalSpatialAugment, SpatialAugConfig, plot_image


class GreenEarthDataset(Dataset):
    def __init__(
        self,
        data_path,
        sub_class="iid",
        rgb_mode=True,
        eobs_vars=("fg", "hu", "pp", "qq", "rr", "tg", "tn", "tx"),
        eobs_roll_days=5,
        cloud_ratio=0.3,
        trim_time=-1,
        use_relative_timesteps=True,
        use_absolute_timesteps=True,
        mode=None,
        do_augmentations=False,
        return_format="diffusionsat",
        input_mode="ndvi_cloud_nextcloud",
        fp16=False,
        spatial_size=(128, 128),
        mask_invalid_value=1.0,
    ):
        self.sub_class = sub_class
        self.rgb_mode = rgb_mode
        self.cloud_ratio = cloud_ratio
        self.T = None if trim_time is None or trim_time <= 0 else int(trim_time)
        self.images = self.load_images(data_path)
        self.mode = mode
        self.return_format = return_format
        self.input_mode = input_mode
        self.fp16 = fp16
        self.mask_invalid_value = mask_invalid_value #TODO: ATTENZIONE

        if self.mode is not None:
            split_90_10 = int(len(self.images) * 0.90)
            if self.mode == "train":
                self.images = self.images[:split_90_10]
            else:
                self.images = self.images[split_90_10:]

        self.use_relative_timesteps = use_relative_timesteps
        self.use_absolute_timesteps = use_absolute_timesteps
        self.meteo_dim = len(eobs_vars) * 2
        if self.use_relative_timesteps:
            self.meteo_dim += 1
        if self.use_absolute_timesteps:
            self.meteo_dim += 2

        self.do_augm = do_augmentations
        self.aug = TemporalSpatialAugment(
            SpatialAugConfig(
                out_size=spatial_size,
                crop_scale=(0.3, 0.5),
                hflip_p=0.5,
                vflip_p=0.5,
                do_resize_back=True,
            )
        )

        self.eobs_vars = eobs_vars
        self.eobs_roll_days = eobs_roll_days
        self.eobs_mean = xr.DataArray(
            data=[
                8.90661030749754,
                2.732927619847993,
                77.54440854529798,
                1014.330962704611,
                126.47924227500346,
                1.7713217310829938,
                4.770701430461286,
                13.567999825718509,
            ],
            coords={
                "variable": [
                    "eobs_tg",
                    "eobs_fg",
                    "eobs_hu",
                    "eobs_pp",
                    "eobs_qq",
                    "eobs_rr",
                    "eobs_tn",
                    "eobs_tx",
                ]
            },
        )
        self.eobs_std = xr.DataArray(
            data=[
                9.75620252236597,
                1.4870108944469236,
                13.511387994026359,
                10.262645403460999,
                97.05522895011327,
                4.147967261223076,
                9.044987677752898,
                11.08198777356161,
            ],
            coords={
                "variable": [
                    "eobs_tg",
                    "eobs_fg",
                    "eobs_hu",
                    "eobs_pp",
                    "eobs_qq",
                    "eobs_rr",
                    "eobs_tn",
                    "eobs_tx",
                ]
            },
        )

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"images path {im_path} does not exist"

        # Keep your original tile restriction. If you want all tiles, replace this
        # by: glob.glob(os.path.join(im_path, self.sub_class, "**/*.nc"), recursive=True)
        ims = glob.glob(os.path.join(im_path, self.sub_class, "32UNC/*.nc"))
        return sorted(ims)

    def __len__(self):
        return len(self.images)

    def _compute_time_and_cloud_idx(self, ds):
        idx = np.where(ds["s2_avail"].fillna(0).to_numpy() == 1)[0]

        visible = (ds.s2_dlmask < 1) & ds.s2_SCL.isin([1, 2, 4, 5, 6, 7])
        visible_mask = visible.isel(time=idx).to_numpy().astype(bool)  # [T,H,W]

        cloud_frac = (~visible_mask).mean(axis=(1, 2))
        cloud_idx = cloud_frac <= self.cloud_ratio
        idx = idx[cloud_idx]

        return idx, visible_mask[cloud_idx]

    def _compute_meteo_history_daily(self, ds):
        """
        Returns [time, 2 * len(eobs_vars)] with rolling mean and rolling std.
        """
        eobs_names = [f"eobs_{v}" for v in self.eobs_vars]
        eobs_daily = ds[eobs_names].to_array("variable").transpose("time", "variable")

        mean = self.eobs_mean.sel(variable=eobs_names)
        std = self.eobs_std.sel(variable=eobs_names)

        eobs_daily = (eobs_daily - mean) / (std + 1e-6)
        eobs_daily = eobs_daily.fillna(0.0)

        k = self.eobs_roll_days
        roll_mean = eobs_daily.rolling(time=k, min_periods=1).mean()
        roll_std = eobs_daily.rolling(time=k, min_periods=1).std().fillna(0.0)

        meteo_hist_daily = xr.concat([roll_mean, roll_std], dim="stat").transpose(
            "time", "stat", "variable"
        )
        return meteo_hist_daily.values.reshape(len(ds.time), -1).astype(np.float32)

    def _compute_day_of_the_year(self, ds, idx):
        kept_times = ds["time"].to_numpy()[idx]
        doy = (kept_times.astype("datetime64[D]") - kept_times.astype("datetime64[Y]")).astype(int) + 1
        doy = doy.astype(np.float32)

        doy_rad = 2.0 * np.pi * (doy / 365.25)
        doy_sin = np.sin(doy_rad).astype(np.float32)
        doy_cos = np.cos(doy_rad).astype(np.float32)

        return torch.from_numpy(np.stack([doy_sin, doy_cos], axis=1))

    def __getitem__(self, index):
        filepath = self.images[index]
        ds = xr.open_dataset(filepath)

        idx, visible_mask_kept = self._compute_time_and_cloud_idx(ds)
        if len(idx) == 0:
            raise RuntimeError(
                f"No frames satisfy cloud_ratio={self.cloud_ratio} for {filepath}."
            )
        
        time_values = ds["time"].to_numpy()[idx]
        time_values = [np.datetime_as_string(t, unit="D") for t in time_values]
        
        meteo_hist_daily = self._compute_meteo_history_daily(ds)
        meteo_t = torch.from_numpy(meteo_hist_daily[idx])

        if self.use_relative_timesteps:
            times = ds["time"].to_numpy()
            kept_times = times[idx]
            if len(kept_times) > 1:
                dt_days = (kept_times[1:] - kept_times[:-1]) / np.timedelta64(1, "D")
                dt_days = np.concatenate([[0.0], dt_days]).astype(np.float32)
            else:
                dt_days = np.array([0.0], dtype=np.float32)
            dt_t = torch.from_numpy(dt_days).unsqueeze(1)
            meteo_t = torch.cat([meteo_t, dt_t], dim=1)

        if self.use_absolute_timesteps:
            time_features = self._compute_day_of_the_year(ds, idx)
            meteo_t = torch.cat([meteo_t, time_features], dim=1)

        # 1 means cloud/invalid, 0 means visible/valid.
        cloud_mask_t = torch.from_numpy((~visible_mask_kept).astype(np.float32)).unsqueeze(0)

        b02 = ds["s2_B02"].isel(time=idx)
        b03 = ds["s2_B03"].isel(time=idx)
        b04 = ds["s2_B04"].isel(time=idx)
        b8a = ds["s2_B8A"].isel(time=idx)

        red = b04.to_numpy().astype(np.float32)
        nir = b8a.to_numpy().astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        ndvi_t = torch.from_numpy(ndvi).unsqueeze(0)
        ndvi_t = torch.clip(ndvi_t, -1.0, 1.0)

        if self.rgb_mode:
            rgb = (
                xr.concat([b04, b03, b02], dim="channel")
                .transpose("time", "lat", "lon", "channel")
                .to_numpy()
            )
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            video_t = torch.from_numpy(rgb).permute(3, 0, 1, 2)
            video_t = torch.clip(video_t, 0.0, 1.0)
            video_t = video_t * 2.0 - 1.0
        else:
            next_clouds = torch.zeros_like(cloud_mask_t)
            if cloud_mask_t.shape[1] > 1:
                next_clouds[:, :-1] = cloud_mask_t[:, 1:]
            video_t = torch.cat([ndvi_t, cloud_mask_t, next_clouds], dim=0)

        if self.T is not None:
            video_t = video_t[:, : self.T].contiguous()
            ndvi_t = ndvi_t[:, : self.T].contiguous()
            cloud_mask_t = cloud_mask_t[:, : self.T].contiguous()
            meteo_t = meteo_t[: self.T].contiguous()
            time_values = time_values[: self.T]
        else:
            video_t = video_t.contiguous()
            ndvi_t = ndvi_t.contiguous()
            cloud_mask_t = cloud_mask_t.contiguous()
            meteo_t = meteo_t.contiguous()  

        sample = {
            "text": "A satellite image",
            "video": video_t,
            "ndvi": ndvi_t,
            "actual_length": video_t.shape[1],
            "cloud_mask": cloud_mask_t,
            "meteo": meteo_t,
            "time": time_values,
        }
        # Static samples
        landcover = ds["esawc_lc"].to_numpy().astype(np.float32) 
        geomorpho = ds["geom_cls"].to_numpy().astype(np.float32) 
        elevation = ds["nasa_dem"].to_numpy().astype(np.float32)

        if self.mode == "train" and self.do_augm:
            sample = self.aug(sample)

        if self.return_format == "earthnet2021x":
            return to_earthnet2021x_format(
                sample,
                [landcover, elevation, geomorpho],
                input_mode=self.input_mode,
                filepath=filepath,
                cubename=os.path.basename(filepath),
                fp16=self.fp16,
                mask_invalid_value=self.mask_invalid_value,
            )

        return sample


if __name__ == "__main__":
    df = GreenEarthDataset(
        data_path="../path/to/greenearthnet",
        sub_class="iid",
        rgb_mode=True,
        use_absolute_timesteps=True,
        use_relative_timesteps=True,
        cloud_ratio=1.0,
        trim_time=-1,
        return_format="earthnet2021x", # diffusionsat/earthnet2021x
        input_mode="ndvi_cloud_nextcloud",
    )
    sample = df[0]
    print(sample["dynamic"][0].shape)       # [T, 3, H, W]
    print(sample["dynamic"][1].shape)       # [T, C_meteo]
    print(sample["dynamic_mask"][0].shape)  # [T, 1, H, W]
    # plot_image(im_t, im_name, is_cloud=False) for visualization