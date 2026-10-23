"""Evaluate one or more model seeds and aggregate dataset-level metrics."""

import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import EarthNet2021XCustomDataModule
from earthnet_models_pytorch.model import MODELS
from earthnet_models_pytorch.task import SpatioTemporalTask
from earthnet_models_pytorch.utils import parse_setting
from eval_utils import evaluate_ablation_to_tiff


BASE_METRIC_COLUMNS = [
    "rmse",
    "bias",
    "pearson_r2",
    "pss_mse",
    "future_closer_rate",
]
METRIC_COLUMNS = [
    *BASE_METRIC_COLUMNS,
    *(f"field_{metric}" for metric in BASE_METRIC_COLUMNS),
]


def evaluation_output_name(args):
    if args.output_dir_name is not None:
        return args.output_dir_name
    return (
        "model_evaluation_zero_nextcloud"
        if args.zeroing_nextcloud
        else "model_evaluation"
    )


def find_seed_dirs(setting_root: Path):
    """Find seed directories, also accepting a single seed as the root."""
    if (
        (setting_root / "setting.yaml").is_file()
        and (setting_root / "checkpoints" / "last.ckpt").is_file()
    ):
        return [setting_root]

    return sorted(
        path
        for path in setting_root.iterdir()
        if path.is_dir()
        and (path / "setting.yaml").is_file()
        and (path / "checkpoints" / "last.ckpt").is_file()
    )


def evaluate_seed(seed_dir: Path, args, device):
    setting_path = seed_dir / "setting.yaml"
    checkpoint_path = seed_dir / "checkpoints" / "last.ckpt"
    setting_dict = parse_setting(str(setting_path))

    requested_workers = int(setting_dict["Data"].get("num_workers", 0))
    available_workers = int(
        os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
    )
    if requested_workers > available_workers:
        print(
            f"Reducing DataLoader workers from {requested_workers} to "
            f"{available_workers} to match the allocated CPUs."
        )
        setting_dict["Data"]["num_workers"] = available_workers

    data_args = [
        "--{}={}".format(key, value)
        for key, value in setting_dict["Data"].items()
    ]
    data_parser = ArgumentParser()
    data_parser = EarthNet2021XCustomDataModule.add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = EarthNet2021XCustomDataModule(data_params)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    model_args = [
        "--{}={}".format(key, value)
        for key, value in setting_dict["Model"].items()
    ]
    model_parser = ArgumentParser()
    model_parser = MODELS[setting_dict["Architecture"]].add_model_specific_args(
        model_parser
    )
    model_params = model_parser.parse_args(model_args)
    model = MODELS[setting_dict["Architecture"]](model_params)

    # Build and load the complete Lightning task, as test.py does.  Besides
    # keeping inference consistent across architectures, the task wrapper adds
    # runtime fields such as global_step before forwarding a batch to the model.
    task_args = [
        "--{}={}".format(key, value)
        for key, value in setting_dict["Task"].items()
    ]
    task_parser = ArgumentParser()
    task_parser = SpatioTemporalTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = SpatioTemporalTask(model=model, hparams=task_params).to(device)
    task.load_model_weights_from_checkpoint(checkpoint_path=str(checkpoint_path))
    print(f"Loaded task weights for {seed_dir.name} from {checkpoint_path}")

    output_dir = seed_dir / evaluation_output_name(args)

    results = evaluate_ablation_to_tiff(
        model=task,
        dataloader=test_loader,
        output_dir=output_dir,
        device=device,
        context_length=setting_dict["Task"]["context_length"],
        target_length=setting_dict["Task"]["target_length"],
        lc_min=setting_dict["Task"]["loss"]["lc_min"],
        lc_max=setting_dict["Task"]["loss"]["lc_max"],
        ndvi_pred_channel=0,
        ndvi_targ_channel=0,
        use_cloud_mask=True,
        use_lc_mask=True,
        use_last_valid_context=False,
        change_threshold=0.03,
        ndvi_noise_min=0.0,
        ndvi_noise_max=1.0,
        ablation_seed=357,
        include_strong_all_shuffle=False,
        max_batches=None,
        save_tiffs=False,
        save_targets=True,
        save_full_dataset_timeseries=True,
        save_horizon_plots=False,
        input_mode=data_params.input_mode,
        zeroing_nextcloud=args.zeroing_nextcloud,
        field_eval_mask_path=args.field_eval_mask or None,
    )

    summary_df, per_sample_df, horizon_summary_df, per_horizon_df = results
    return summary_df, per_sample_df, horizon_summary_df, per_horizon_df


def aggregate_seed_metrics(seed_summaries, output_dir: Path):
    per_seed_df = pd.concat(seed_summaries, ignore_index=True)
    per_seed_df.to_csv(output_dir / "per_seed_ablation_metrics.csv", index=False)

    seed_count = per_seed_df["seed"].nunique()
    if seed_count == 1:
        columns = ["seed", "variant", *METRIC_COLUMNS]
        aggregate_df = per_seed_df[columns].copy()
        aggregate_df.to_csv(
            output_dir / "cross_seed_ablation_metrics.csv",
            index=False,
        )
        print("\nSingle-seed metrics (standard deviation is not reported):")
        print(aggregate_df.to_string(index=False))
        return aggregate_df

    rows = []
    for variant, group in per_seed_df.groupby("variant", sort=False):
        row = {"variant": variant, "n_seeds": group["seed"].nunique()}
        for metric in METRIC_COLUMNS:
            values = group[metric].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                mean = population_std = float("nan")
            else:
                mean = float(np.mean(finite_values))
                population_std = float(np.std(finite_values, ddof=0))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_population_std"] = population_std
        rows.append(row)

    aggregate_df = pd.DataFrame(rows)
    aggregate_df.to_csv(
        output_dir / "cross_seed_ablation_metrics.csv",
        index=False,
    )
    print("\nCross-seed metrics (population standard deviation, ddof=0):")
    print(aggregate_df.to_string(index=False))
    return aggregate_df


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--setting_root",
        type=Path,
        metavar="path/to/seed/directories",
        default=Path("/path/to/experiments"),
        help=(
            "Directory containing seed subdirectories, or one seed directory "
            "containing setting.yaml and checkpoints/last.ckpt."
        ),
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default=None,
        help="Evaluation directory name created inside every seed directory.",
    )
    parser.add_argument(
        "--zeroing_nextcloud",
        action="store_true",
        help=(
            "Set the final dynamic-image channel to zero during evaluation when "
            "Data.input_mode contains 'nextcloud'."
        ),
    )
    parser.add_argument(
        "--field_eval_mask",
        type=str,
        default="",
        help=(
            "Categorical field mask used for separate field_* NDVI metrics. "
            "Primary metrics always use every otherwise-valid image pixel. "
            "Pass an empty string to disable field-level metrics."
        ),
    )
    args = parser.parse_args()

    if not args.setting_root.is_dir():
        raise NotADirectoryError(f"Setting root does not exist: {args.setting_root}")

    seed_dirs = find_seed_dirs(args.setting_root)
    if not seed_dirs:
        raise RuntimeError(f"No valid seed directories found in {args.setting_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_summaries = []

    for seed_dir in seed_dirs:
        print(f"\nEvaluating {seed_dir.name}")
        summary_df, _, _, _ = evaluate_seed(seed_dir, args, device)
        summary_df = summary_df.copy()
        summary_df.insert(0, "seed", seed_dir.name)
        seed_summaries.append(summary_df)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_output_dir = args.setting_root / evaluation_output_name(args)
    aggregate_output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_seed_metrics(seed_summaries, aggregate_output_dir)


if __name__ == "__main__":
    main()
