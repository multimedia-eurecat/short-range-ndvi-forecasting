import os, sys
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
import yaml
from data import EarthNet2021XCustomDataModule
from earthnet_models_pytorch.model import MODELS
from earthnet_models_pytorch.task import SpatioTemporalTask
from earthnet_models_pytorch.utils import parse_setting
from pytorch_lightning.callbacks import TQDMProgressBar
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy


def train_model(setting_dict: dict, setting_file: str = None, checkpoint: str = None):
    start = time.time()

    pl.seed_everything(setting_dict["Seed"])
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
        print(f"===> Loading from checkpoint {checkpoint}")
        task.load_model_weights_from_checkpoint(checkpoint_path=checkpoint)

    # Logger
    logger = pl.loggers.TensorBoardLogger(**setting_dict["Logger"])

    if (
        setting_file is not None
        and type(logger.experiment).__name__ != "DummyExperiment"
    ):
        print("Copying setting yaml.")
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(os.path.join(logger.log_dir, "setting.yaml"), "w") as fp:
            yaml.dump(setting_dict, fp)

    # Checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**setting_dict["Checkpointer"])

    # Trainer
    trainer_dict = setting_dict["Trainer"]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
        **trainer_dict,
    )

    trainer.fit(task, dm)

    print(
        f"Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}"
    )

    end = time.time()

    print(f"Calculation done in {end - start} seconds.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./short-range-ndvi-forecasting/model_configs/contextformer/contextformer6M/rep_seed=27.yaml",
        help="yaml with all settings",
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to checkpoint" 
    )
    args = parser.parse_args()
    setting_dict = parse_setting(args.config)

    print("Setting:", setting_dict["Setting"])
    print("Data dataset_name:", setting_dict["Data"]["dataset_name"])
    print("Model context_length:", setting_dict["Model"]["context_length"])
    print("Model target_length:", setting_dict["Model"]["target_length"])
    print("Task setting:", setting_dict["Task"]["setting"])
    print("Task loss:", setting_dict["Task"]["loss"])
    print("Task metric_kwargs:", setting_dict["Task"]["metric_kwargs"])
    print("Checkpointer:", setting_dict["Checkpointer"])

    train_model(setting_dict, args.config, args.checkpoint)
