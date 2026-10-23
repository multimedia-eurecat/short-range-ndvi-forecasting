import argparse
import copy
import multiprocessing
from pathlib import Path
from typing import Optional, Union
from functools import partial
import pytorch_lightning as pl
import torch
from earthnet_models_pytorch.utils import str2bool
from torch.utils.data import DataLoader, random_split

# Adjust these imports to your package/module names.
from .greenearth_dataset import GreenEarthDataset
from .replicate_dataset import (
    CLOUD_PROB_PATH,
    DATA_PATH,
    WEATHER_DATA_PATH,
    ReplicateDataset,
)
import torch

def earthnet2021x_chunk_collate_fn(
    batch,
    chunk_len: int,
    stride: int = 1,
    drop_last: bool = True,
):
    dynamic0_chunks = []
    meteo_chunks = []
    ndvi_chunks = []
    mask_chunks = []
    static_chunks = []
    landcover_chunks = []
    filepath_chunks = []
    cubename_chunks = []
    time_chunks = []
    metadata = []

    def append_chunk(
        dynamic0_chunk,
        meteo_chunk,
        ndvi_chunk,
        mask_chunk,
        static0,
        landcover,
        filepath,
        cubename,
        time_values,
        sample_idx,
        start,
        end,
        actual_length,
        padded,
    ):
        dynamic0_chunks.append(dynamic0_chunk.contiguous())
        meteo_chunks.append(meteo_chunk.contiguous())
        ndvi_chunks.append(ndvi_chunk.contiguous())
        mask_chunks.append(mask_chunk.contiguous())
        static_chunks.append(static0.contiguous())
        landcover_chunks.append(landcover.contiguous())
        filepath_chunks.append(str(filepath))

        suffix = f"__s{start:04d}_e{end:04d}"
        if padded:
            suffix += "_pad"
        cubename_chunks.append(f"{str(cubename)}{suffix}")

        if time_values is not None:
            curr_times = list(time_values[start:end])
            if padded:
                curr_times = curr_times + [None] * (chunk_len - len(curr_times))
            time_chunks.append(curr_times)
        else:
            time_chunks.append([None] * chunk_len)

        metadata.append(
            {
                "sample_idx": sample_idx,
                "start": start,
                "end": end,
                "actual_length": actual_length,
                "padded": padded,
            }
        )

    for sample_idx, sample in enumerate(batch):
        dynamic0 = sample["dynamic"][0]          # [T,C,H,W]
        meteo = sample["dynamic"][1]             # [T,C_meteo]
        ndvi = sample["dynamic"][2]              # [T,1,H,W]
        dynamic_mask = sample["dynamic_mask"][0] # [T,1,H,W]
        static0 = sample["static"][0]            # [C_static,H,W]
        landcover = sample["landcover"]          # [1,H,W]
        time_values = sample.get("time", None)

        filepath = sample.get("filepath", "")
        cubename = sample.get("cubename", "")

        assert dynamic0.ndim == 4, f"dynamic[0] must be [T,C,H,W], got {dynamic0.shape}"
        assert meteo.ndim == 2, f"dynamic[1] must be [T,C_meteo], got {meteo.shape}"
        assert ndvi.ndim == 4, f"dynamic[2] must be [T,1,H,W], got {ndvi.shape}"
        assert dynamic_mask.ndim == 4, f"dynamic_mask[0] must be [T,1,H,W], got {dynamic_mask.shape}"
        assert static0.ndim == 3, f"static[0] must be [C_static,H,W], got {static0.shape}"
        assert landcover.ndim == 3, f"landcover must be [1,H,W], got {landcover.shape}"

        T = dynamic0.shape[0]

        assert meteo.shape[0] == T
        assert dynamic_mask.shape[0] == T

        if time_values is not None:
            assert len(time_values) == T, (
                f"time length must match T. Got len(time)={len(time_values)} and T={T}"
            )

        # ------------------------------------------------------------
        # Case 1: sequence shorter than chunk_len
        # ------------------------------------------------------------
        if T < chunk_len:
            if drop_last:
                continue

            pad_t = chunk_len - T

            dynamic0_pad = torch.zeros(
                pad_t,
                dynamic0.shape[1],
                dynamic0.shape[2],
                dynamic0.shape[3],
                dtype=dynamic0.dtype,
                device=dynamic0.device,
            )
            meteo_pad = torch.zeros(
                pad_t,
                meteo.shape[1],
                dtype=meteo.dtype,
                device=meteo.device,
            )
            ndvi_pad = torch.zeros(
                pad_t,
                ndvi.shape[1],
                ndvi.shape[2],
                ndvi.shape[3],
                dtype=ndvi.dtype,
                device=ndvi.device,
            )
            mask_pad = torch.ones(
                pad_t,
                dynamic_mask.shape[1],
                dynamic_mask.shape[2],
                dynamic_mask.shape[3],
                dtype=dynamic_mask.dtype,
                device=dynamic_mask.device,
            )

            append_chunk(
                dynamic0_chunk=torch.cat([dynamic0, dynamic0_pad], dim=0),
                meteo_chunk=torch.cat([meteo, meteo_pad], dim=0),
                ndvi_chunk=torch.cat([ndvi, ndvi_pad], dim=0),
                mask_chunk=torch.cat([dynamic_mask, mask_pad], dim=0),
                static0=static0,
                landcover=landcover,
                filepath=filepath,
                cubename=cubename,
                time_values=time_values,
                sample_idx=sample_idx,
                start=0,
                end=T,
                actual_length=T,
                padded=True,
            )

            continue

        # ------------------------------------------------------------
        # Case 2: full chunks
        # ------------------------------------------------------------
        starts = list(range(0, T - chunk_len + 1, stride))

        for start in starts:
            end = start + chunk_len

            append_chunk(
                dynamic0_chunk=dynamic0[start:end],
                meteo_chunk=meteo[start:end],
                ndvi_chunk=ndvi[start:end],
                mask_chunk=dynamic_mask[start:end],
                static0=static0,
                landcover=landcover,
                filepath=filepath,
                cubename=cubename,
                time_values=time_values,
                sample_idx=sample_idx,
                start=start,
                end=end,
                actual_length=chunk_len,
                padded=False,
            )

        # ------------------------------------------------------------
        # Case 3: optional final padded chunk
        # ------------------------------------------------------------
        if not drop_last:
            last_start = starts[-1]
            next_start = last_start + stride

            if next_start < T and next_start not in starts:
                dynamic0_rem = dynamic0[next_start:]
                meteo_rem = meteo[next_start:]
                ndvi_rem = ndvi[next_start:]
                mask_rem = dynamic_mask[next_start:]

                T_rem = dynamic0_rem.shape[0]

                if 0 < T_rem < chunk_len:
                    pad_t = chunk_len - T_rem

                    dynamic0_pad = torch.zeros(
                        pad_t,
                        dynamic0.shape[1],
                        dynamic0.shape[2],
                        dynamic0.shape[3],
                        dtype=dynamic0.dtype,
                        device=dynamic0.device,
                    )
                    meteo_pad = torch.zeros(
                        pad_t,
                        meteo.shape[1],
                        dtype=meteo.dtype,
                        device=meteo.device,
                    )
                    ndvi_pad = torch.zeros(
                        pad_t,
                        ndvi.shape[1],
                        ndvi.shape[2],
                        ndvi.shape[3],
                        dtype=ndvi.dtype,
                        device=ndvi.device,
                    )
                    mask_pad = torch.ones(
                        pad_t,
                        dynamic_mask.shape[1],
                        dynamic_mask.shape[2],
                        dynamic_mask.shape[3],
                        dtype=dynamic_mask.dtype,
                        device=dynamic_mask.device,
                    )

                    append_chunk(
                        dynamic0_chunk=torch.cat([dynamic0_rem, dynamic0_pad], dim=0),
                        meteo_chunk=torch.cat([meteo_rem, meteo_pad], dim=0),
                        ndvi_chunk=torch.cat([ndvi_rem, ndvi_pad], dim=0),
                        mask_chunk=torch.cat([mask_rem, mask_pad], dim=0),
                        static0=static0,
                        landcover=landcover,
                        filepath=filepath,
                        cubename=cubename,
                        time_values=time_values,
                        sample_idx=sample_idx,
                        start=next_start,
                        end=T,
                        actual_length=T_rem,
                        padded=True,
                    )

    if len(dynamic0_chunks) == 0:
        raise RuntimeError(
            "No chunks created. Check chunk_len, stride, drop_last, and cloud_ratio."
        )

    return {
        "dynamic": [
            torch.stack(dynamic0_chunks, dim=0),
            torch.stack(meteo_chunks, dim=0),
            torch.stack(ndvi_chunks, dim=0),
        ],
        "dynamic_mask": [
            torch.stack(mask_chunks, dim=0),
        ],
        "static": [
            torch.stack(static_chunks, dim=0),
        ],
        "static_mask": [],
        "landcover": torch.stack(landcover_chunks, dim=0),
        "filepath": filepath_chunks,
        "cubename": cubename_chunks,
        "time": time_chunks,
        "metadata": metadata,
    }


class EarthNet2021XCustomDataModule(pl.LightningDataModule):
    """
    Drop-in replacement for the EarthNet2021XDataModule.

    dataset_name options:
        "greenearth"  -> your GreenEarthDataset adapted to EarthNet2021X format
        "replicate"   -> your ReplicateDataset adapted to EarthNet2021X format

    input_mode options:
        Spectral base: "rgb", "ndvi", or "ndvi_ndwi"
        Optional suffix: "_cloud" or "_cloud_nextcloud"
    """

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))
        self.base_dir = Path(hparams.base_dir)
        self.collate_fn = partial(
            earthnet2021x_chunk_collate_fn,
            chunk_len=self.hparams.seq_len,
            stride=self.hparams.seq_len,
            drop_last=False,
        )

    @staticmethod
    def add_data_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None,
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--base_dir", type=str, default="data/datasets/")
        # parser.add_argument("--test_track", type=str, default="iid")

        parser.add_argument("--dataset_name", type=str, default="earthnet")
        parser.add_argument("--input_mode", type=str, default="ndvi")
        parser.add_argument("--seq_len", type=int, default=30)
        parser.add_argument("--context_len", type=int, default=None)

        parser.add_argument("--fp16", type=str2bool, default=False)
        # parser.add_argument("--dl_cloudmask", type=str2bool, default=False)
        # parser.add_argument("--allow_fastaccess", type=str2bool, default=False)
        parser.add_argument("--shuffle_train", type=str2bool, default=True)

        # parser.add_argument("--new_valset", type=str2bool, default=False)
        # parser.add_argument("--val_pct", type=float, default=0.05)
        # parser.add_argument("--val_split_seed", type=float, default=42)

        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)

        parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())

        # Optional paths for Replicate. Defaults use constants from replicate_dataset_earthnet.py.
        parser.add_argument("--replicate_img_dir", type=str, default=DATA_PATH)
        parser.add_argument("--replicate_weather_dir", type=str, default=WEATHER_DATA_PATH)
        parser.add_argument("--replicate_cloud_dir", type=str, default=CLOUD_PROB_PATH)
        parser.add_argument("--replicate_stride", type=int, default=1)

        # Optional GreenEarth settings.
        parser.add_argument("--greenearth_sub_class", type=str, default="iid")
        parser.add_argument("--greenearth_cloud_ratio", type=float, default=1.0)
        parser.add_argument("--greenearth_trim_time", type=int, default=-1)

        return parser

    def _make_train_dataset(self):
        if self.hparams.dataset_name == "greenearthnet":
            return GreenEarthDataset(
                data_path=str(self.base_dir),
                sub_class=self.hparams.greenearth_sub_class,
                rgb_mode=False,
                cloud_ratio=self.hparams.greenearth_cloud_ratio,
                trim_time=self.hparams.greenearth_trim_time,
                mode="train",
                return_format="earthnet2021x",
                input_mode=self.hparams.input_mode,
                fp16=self.hparams.fp16,
            )

        if self.hparams.dataset_name == "replicate":
            return ReplicateDataset(
                img_data_path=self.hparams.replicate_img_dir,
                weather_data_path=self.hparams.replicate_weather_dir,
                clouds_data_path=self.hparams.replicate_cloud_dir,
                sequence_len=self.hparams.seq_len,
                context_len=self.hparams.context_len,
                no_overlap=True,
                stride=self.hparams.replicate_stride,
                start_sequence_from=self.hparams.seq_len,
                mode="train",
                return_format="earthnet2021x",
                input_mode=self.hparams.input_mode,
                fp16=self.hparams.fp16,
            )

        raise ValueError("dataset_name must be one of: greenearth, replicate")

    def _make_test_dataset(self):
        if self.hparams.dataset_name == "greenearthnet":
            return GreenEarthDataset(
                data_path=str(self.base_dir),
                sub_class=self.hparams.greenearth_sub_class,
                rgb_mode=False,
                cloud_ratio=self.hparams.greenearth_cloud_ratio,
                trim_time=self.hparams.greenearth_trim_time,
                mode="val",
                return_format="earthnet2021x",
                input_mode=self.hparams.input_mode,
                fp16=self.hparams.fp16,
            )

        if self.hparams.dataset_name == "replicate":
            return ReplicateDataset(
                img_data_path=self.hparams.replicate_img_dir,
                weather_data_path=self.hparams.replicate_weather_dir,
                clouds_data_path=self.hparams.replicate_cloud_dir,
                sequence_len=self.hparams.seq_len,
                context_len=self.hparams.context_len,
                no_overlap=True,
                stride=1,
                start_sequence_from=self.hparams.seq_len,
                mode="val",
                return_format="earthnet2021x",
                input_mode=self.hparams.input_mode,
                fp16=self.hparams.fp16,
            )

        raise ValueError("dataset_name must be one of: earthnet, greenearth, replicate")

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.earthnet_train = self._make_train_dataset()
            self.earthnet_val = self._make_test_dataset() # TODO DA SISTEMARE

        if stage == "test" or stage is None:
            self.earthnet_test = self._make_test_dataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.earthnet_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=self.hparams.shuffle_train,
            collate_fn= self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.earthnet_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn= self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.earthnet_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn= self.collate_fn,
        )
