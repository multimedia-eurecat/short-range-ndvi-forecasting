import json
import os
from pathlib import Path

import numpy as np
from argparse import ArgumentParser

import pytorch_lightning as pl
from data import EarthNet2021XCustomDataModule
from earthnet_models_pytorch.model import MODELS
from earthnet_models_pytorch.task import SpatioTemporalTask
from earthnet_models_pytorch.utils import parse_setting
from pytorch_lightning.callbacks import TQDMProgressBar


def test_model(setting_dict: dict, checkpoint: str):
    # Data
    data_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Data"].items()
    ]
    data_parser = ArgumentParser()
    data_parser = EarthNet2021XCustomDataModule.add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = EarthNet2021XCustomDataModule(data_params)

    # Model
    model_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Model"].items()
    ]
    model_parser = ArgumentParser()
    model_parser = MODELS[setting_dict["Architecture"]].add_model_specific_args(
        model_parser
    )
    model_params = model_parser.parse_args(model_args)
    model = MODELS[setting_dict["Architecture"]](model_params)

    # Task
    task_args = [
        "--{}={}".format(key, value) for key, value in setting_dict["Task"].items()
    ]
    task_parser = ArgumentParser()
    task_parser = SpatioTemporalTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    task = SpatioTemporalTask(model=model, hparams=task_params)

    if checkpoint != None:
        task.load_model_weights_from_checkpoint(checkpoint_path=checkpoint)
    else:
        print("Testing the model with random weights init")

    # Trainer

    trainer_dict = setting_dict["Trainer"]
    trainer_dict["logger"] = False
    trainer = pl.Trainer(callbacks=TQDMProgressBar(refresh_rate=10), **trainer_dict)

    dm.setup("test")

    trainer.test(model=task, datamodule=dm, ckpt_path=None)

    score_path = task.pred_dir / "total_score.json"
    if not score_path.is_file():
        raise FileNotFoundError(f"Test score was not written to {score_path}")

    with score_path.open() as fp:
        score = json.load(fp)

    if "RMSE_Veg" not in score:
        raise KeyError(f"RMSE_Veg is missing from {score_path}")

    return float(score["RMSE_Veg"])

def main(args):
    setting_dict = parse_setting(args.setting)
    score = test_model(setting_dict, args.checkpoint)

    return score
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--setting_root",
        type=Path,
        metavar="path/to/seed/directories",
        default=Path(
            "./path/to/experiments"
        ),
        help="directory containing one subdirectory per seed",
    )

    args = parser.parse_args()

    for k in list(os.environ):
        if k.startswith("SLURM"):
            del os.environ[k]

    seed_dirs = sorted(
        path
        for path in args.setting_root.iterdir()
        if path.is_dir()
        and (path / "setting.yaml").is_file()
        and (path / "checkpoints" / "last.ckpt").is_file()
    )

    if not seed_dirs:
        raise RuntimeError(f"No valid seed directories found in {args.setting_root}")

    final_scores = []
    for seed_dir in seed_dirs:
        args.setting = str(seed_dir / "setting.yaml")
        args.checkpoint = str(seed_dir / "checkpoints" / "last.ckpt")
        score = main(args)
        final_scores.append(score)
        print(f"{seed_dir.name}: masked RMSE = {score:.6f}")

    final_scores = np.asarray(final_scores, dtype=float)

    if final_scores.size == 1:
        print(f"Masked RMSE: {final_scores[0]:.6f}")
    else:
        print(f"Mean masked RMSE: {np.mean(final_scores):.6f}")
        print(
            "Population standard deviation: "
            f"{np.std(final_scores, ddof=0):.6f}"
        )
    
