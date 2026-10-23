# Based on the SpatioTemporalTask implementation from:
# https://github.com/earthnet2021/earthnet-models-pytorch
#
# Original implementation licensed under the MIT License.
#
# Substantially modified and extended by Cynthia Ugwu (@cugwu) for:
# https://github.com/multimedia-eurecat/short-range-ndvi-forecasting
#
# Adapted the training and evaluation workflow for short-range NDVI forecasting.
# Changes include configurable input handling and prediction visualization,
# cloud-channel handling, and validation metrics exported to JSON and CSV.
# Reworked test-time evaluation to save per-sample and aggregate metrics,
# support multi-GPU result aggregation, and provide optional score sorting.
# Added partial checkpoint loading for compatible pretrained model weights.

import argparse
import ast
import copy
import json
import csv
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from earthnet_models_pytorch.metric import METRICS
from earthnet_models_pytorch.task import SCHEDULERS, setup_loss
from earthnet_models_pytorch.utils import log_or_save_ndvi_residual_viz, str2bool
from torch import nn


class SpatioTemporalTask(pl.LightningModule):
    def __init__(self, model: nn.Module, hparams: argparse.Namespace):
        super().__init__()

        if hasattr(self, "save_hyperparameters"):
            # SpatioTemporalTask herite from the LightningModule, save the parameters in a file
            self.save_hyperparameters(copy.deepcopy(hparams))
        else:
            self.hparams = copy.deepcopy(hparams)

        if hparams.pred_dir is None:
            self.pred_dir = (
                Path(self.logger.log_dir) / "predictions"
                if self.logger is not None
                else Path.cwd() / "experiments" / "predictions"
            )  # logger: hyperparameter of LightningModule for the Trainer
        else:
            self.pred_dir = Path(self.hparams.pred_dir)

        self.model = model
        
        self.loss = setup_loss(hparams.loss)

        self.context_length = hparams.context_length
        self.target_length = hparams.target_length
        self.input_mode = hparams.input_mode

        self.lc_min = hparams.lc_min
        self.lc_max = hparams.lc_max

        self.n_stochastic_preds = hparams.n_stochastic_preds

        self.schedulers = []
        for schedule in self.hparams.schedulers:
            self.schedulers.append(
                (schedule["call_name"], SCHEDULERS[schedule["name"]](**schedule["args"]))
            )
            
        self.metric = METRICS[self.hparams.metric](**self.hparams.metric_kwargs)

        # Storage for outputs to support PyTorch Lightning 2.0 migration
        self.test_step_outputs = []

    @staticmethod
    def add_task_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None,
    ):  # Optional[X] is equivalent to Union[X, None].
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(
            parents=parent_parser, add_help=False
        )  # parents - A list of ArgumentParser objects whose arguments should also be included

        # Path of the directory to save the prediction
        parser.add_argument("--pred_dir", type=str, default=None)

        # Name of the dataset, involves major differences in the variables available and the tasks to be performed.
        parser.add_argument("--setting", type=str, default="en21-std")

        # Dictionnary of the loss name and the distance norm used.
        parser.add_argument(
            "--loss",
            type=ast.literal_eval,
            default='{"name": "L2NDVILoss"}',
        )
        # Metric used for the test set and the validation set.
        parser.add_argument("--metric", type=str, default="NNSE")
        parser.add_argument("--metric_kwargs", type=ast.literal_eval, default="{}")

        # Context and target length for temporal model. A temporal model use a context period to learn the temporal dependencies and predict the target period.
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)
        parser.add_argument("--input_mode", type=str, default=None)

        # Landcover bounds. Used as mask on the non-vegetation pixel.
        parser.add_argument("--lc_min", type=int, default=10)
        parser.add_argument("--lc_max", type=int, default=20)

        # Number of stochastic prediction for statistical models.
        parser.add_argument("--n_stochastic_preds", type=int, default=1)

        # Number of batches to be displayed in the logger
        parser.add_argument("--viz_every_n_epochs", type=int, default=2)
        parser.add_argument("--save_viz_to_disk", type=str2bool, default=True)
        parser.add_argument("--log_viz_to_tensorboard", type=str2bool, default=True)
        parser.add_argument("--viz_residual_scale", type=float, default=0.5)
        parser.add_argument("--vmin", type=float, default=-1.0)
        parser.add_argument("--vmax", type=float, default=1.0)

        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)

        # optimizer: Function that adjusts the attributes of the neural network, such as weights and learning rates.
        parser.add_argument(
            "--optimization",
            type=ast.literal_eval,
            default='{"optimizer": [{"name": "Adam", "args:" {"lr": 0.0001, "betas": (0.9, 0.999)} }], "lr_schedule": [{"name": "multistep", "args": {"milestones": [25, 40], "gamma": 0.1} }]}',
        )
        # scheduler: methods to adjust the learning rate based on the number of epochs
        parser.add_argument("--schedulers", type=ast.literal_eval, default="[]")

        parser.add_argument("--compute_metric_on_test", type=str2bool, default=False)
        return parser

    def forward(
        self, data, pred_start: int = 0, preds_length: Optional[int] = None, kwargs={}
    ):
        """
        data is a dict with tensors
        pred_start is the first index that shall be predicted, defaults to zero.
        preds_length is the length of the prediction, could also be None.
        kwargs are optional keyword arguments parsed to the model, right now these are model schedulers.
        """
        data["global_step"] = self.global_step
        return self.model(
            data,
            pred_start=pred_start,
            preds_length=preds_length,
            **kwargs,
        )

    def configure_optimizers(self):
        "define and load optimizers and schedulers"
        optimizers = [
            getattr(torch.optim, o["name"])(self.parameters(), **o["args"])
            for o in self.hparams.optimization["optimizer"]
        ]

        # torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs.
        schedulers = [
            getattr(torch.optim.lr_scheduler, s["name"])(optimizers[i], **s["args"])
            for i, s in enumerate(self.hparams.optimization["lr_schedule"])
        ]
        return optimizers, schedulers
    
    def _jsonify_value(self, value):
        """
        Convert tensors and numpy values to JSON-safe Python values.
        """
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (np.integer, np.floating)):
            return value.item()

        return value

    def training_step(self, batch, batch_idx):
        """compute and return the training loss and some additional metrics for e.g. the progress bar or logger"""

        # Learning rate scheduling should be applied after optimizer’s update
        kwargs = {}
        for schedule_name, schedule in self.schedulers:
            kwargs[schedule_name] = schedule(self.global_step)
       
        # Predictions generation
        preds, aux = self(batch, kwargs=kwargs)
        loss, logs = self.loss(preds, batch, aux, current_step=self.global_step)

        # # Logs
        # for schedule_name in kwargs:
        #     if len(kwargs[schedule_name]) > 1:
        #         for i, shed_val in enumerate(kwargs[schedule_name]):
        #             logs[f"{schedule_name}_i"] = shed_val
        #     else:
        #         logs[schedule_name] = kwargs[schedule_name]
        # logs["batch_size"] = torch.tensor(
        #     self.hparams.train_batch_size, dtype=torch.float32
        # )
        # Metric logging method
        self.log_dict(logs)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform one evaluation epoch over the validation set. Operates on a single batch of data from the validation set. In this step you d might generate examples or calculate anything of interest like accuracy.
        Return  a List of dictionaries with metrics logged during the validation phase
        """

        data = copy.deepcopy(batch)
        if "nextcloud" in str(self.input_mode).lower():
            data["dynamic"][0][:, :, -1, :, :] = 0.0

        batch_size = torch.tensor(self.hparams.val_batch_size, dtype=torch.int64)

        # # Select only the context data for the model
        # data["dynamic"][0] = data["dynamic"][0][:, : self.context_length, ...]
        # if len(data["dynamic_mask"]) > 0:
        #     data["dynamic_mask"][0] = data["dynamic_mask"][0][
        #         :, : self.context_length, ...
        #     ]

        loss_logs = []  # list of loss values

        # nb of predictions for statistical models
        for i in range(self.n_stochastic_preds):
            # Predictions of the model
            preds, aux = self(
                data, pred_start=self.context_length, preds_length=self.target_length
            )

            # Loss computation
            mse_lc, logs = self.loss(preds, batch, aux)

            if np.isfinite(mse_lc.cpu().detach().numpy()):
                loss_logs.append(logs)

            # Update of the metric
            self.metric.update(preds, batch)

        mean_logs = {
            log_name: torch.tensor(
                [log[log_name] for log in loss_logs],
                dtype=torch.float32,
                device=self.device,
            ).mean()
            for log_name in loss_logs[0]
        }

        # loss_val
        self.log_dict(
            {log_name + "_val": mean_logs[log_name] for log_name in mean_logs},
            sync_dist=True,
            batch_size=batch_size,
        )

        # Visualisation of the prediction for the n first batches
        if ((self.current_epoch + 1) % self.hparams.viz_every_n_epochs)==0 and len(preds.shape) == 5:
            if self.logger is not None:
                scores = self.metric.compute_batch(batch)
                log_or_save_ndvi_residual_viz(
                    tensorboard_logger=self.logger.experiment,
                    batch=batch,
                    preds=preds,
                    batch_idx=batch_idx,
                    current_epoch=self.current_epoch,
                    context_length=self.context_length,
                    target_length=self.target_length,
                    setting=self.hparams.setting,
                    log_dir=self.logger.log_dir,
                    scores=scores,
                    save_to_disk=self.hparams.save_viz_to_disk,
                    log_to_tensorboard=self.hparams.log_viz_to_tensorboard,
                    residual_scale=self.hparams.viz_residual_scale,
                    vmin=self.hparams.vmin,
                    vmax=self.hparams.vmax,
                    split_name="val",
                )


    def on_validation_epoch_end(self):
        current_scores = self.metric.compute()
        self.log_dict(current_scores, sync_dist=True)
        self.metric.reset()  

        if (
            self.logger is not None
            and type(self.logger.experiment).__name__ != "DummyExperiment"
            and self.trainer.is_global_zero
            and (self.current_epoch + 1) % self.hparams.viz_every_n_epochs==0 
        ):
            current_scores["epoch"] = self.current_epoch
            current_scores = {
                k: (
                    str(v.detach().cpu().item())
                    if isinstance(v, torch.Tensor)
                    else str(v)
                )
                for k, v in current_scores.items()
            }

            print(f"\nValidation epoch {self.current_epoch}: {current_scores}\n")

            outpath = Path(self.logger.log_dir) / "validation_scores.json"
            if outpath.is_file():
                with open(outpath, "r") as fp:
                    past_scores = json.load(fp)
                scores = past_scores + [current_scores]
            else:
                scores = [current_scores]

            with open(outpath, "w") as fp:
                json.dump(scores, fp, indent=2)
            

            csv_path = Path(self.logger.log_dir) / "validation_scores.csv"

            fieldnames = ["epoch"] + [k for k in current_scores.keys() if k != "epoch"]

            with open(csv_path, "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(scores)


    def test_step(self, batch, batch_idx): #TODO DA SISTEMARE
        scores = []

        if "nextcloud" in str(self.input_mode).lower():
            batch["dynamic"][0][:, :, -1, :, :] = 0.0

        for i in range(self.n_stochastic_preds):
            preds, aux = self(
                batch,
                pred_start=self.context_length,
                preds_length=self.target_length,
            )

            batch_scores = None
             
            if self.hparams.compute_metric_on_test:
                self.metric.update(preds, batch)
                batch_scores = self.metric.compute_batch(batch)
                # Add useful metadata to each individual score.
                for sample_idx, score_dict in enumerate(batch_scores):
                    score_dict["batch_idx"] = batch_idx
                    score_dict["sample_idx"] = sample_idx
                    score_dict["stochastic_pred"] = i

                    if "cubename" in batch:
                        score_dict["name"] = str(batch["cubename"][sample_idx])
                    elif "filepath" in batch:
                        score_dict["name"] = str(batch["filepath"][sample_idx])

                scores.append(batch_scores)
                
                if self.logger is not None:
                    log_or_save_ndvi_residual_viz(
                        tensorboard_logger=self.logger.experiment,
                        batch=batch,
                        preds=preds,
                        batch_idx=batch_idx,
                        current_epoch=self.current_epoch,
                        context_length=self.context_length,
                        target_length=self.target_length,
                        setting=self.hparams.setting,
                        log_dir=self.pred_dir,
                        scores=batch_scores,
                        save_to_disk=self.hparams.save_viz_to_disk,
                        log_to_tensorboard=self.hparams.log_viz_to_tensorboard,
                        residual_scale=self.hparams.viz_residual_scale,
                        vmin=self.hparams.vmin,
                        vmax=self.hparams.vmax,
                        split_name="test",
                    )

        self.test_step_outputs.append(scores)

        return scores

    def on_test_epoch_end(self):  #TODO DA SISTEMARE
        """Called at the end of a test epoch with the output of all test steps."""
        
        if not self.hparams.compute_metric_on_test:
            self.test_step_outputs.clear()
            return
    
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        print(self.pred_dir)
        
        individual_scores = []

        for batch_output in self.test_step_outputs:
            for stochastic_output in batch_output:
                if stochastic_output is None:
                    continue

                for sample_score in stochastic_output:
                    individual_scores.append(
                        {
                            k: self._jsonify_value(v)
                            for k, v in sample_score.items()
                        }
                    )

        with open(self.pred_dir / f"individual_scores_{self.global_rank}.json", "w") as fp:
            json.dump(individual_scores, fp, indent=2)

        total_scores = self.metric.compute()

        total_scores = {
            k: self._jsonify_value(v)
            for k, v in total_scores.items()
        }

        if self.trainer.is_global_zero:
            with open(self.pred_dir / "total_score.json", "w") as fp:
                json.dump(total_scores, fp, indent=2)

            print("\nFinal test score:")
            print(total_scores)
            print(f"\nPredictions and scores saved in: {self.pred_dir}\n")

        self.metric.reset()
        self.test_step_outputs.clear()

    def teardown(self, stage):
        if stage != "test":
            return

        if not self.hparams.compute_metric_on_test:
            return

        # Important for multi-GPU: wait until every rank has written its file.
        if self.trainer is not None:
            self.trainer.strategy.barrier("test_scores_written")

        if self.global_rank != 0:
            return

        data = []

        # Use only rank score files, not any old sorted/merged files.
        for path in self.pred_dir.glob("individual_scores_[0-9]*.json"):
            with open(path, "r") as fp:
                data += json.load(fp)

            path.unlink(missing_ok=True)

        if len(data) == 0:
            return

        # Do not deduplicate by name.
        with open(self.pred_dir / "individual_scores.json", "w") as fp:
            json.dump(data, fp, indent=2)

        # Optional: save sorted version by local/main score if available.
        sort_key = None
        for candidate in ["rmse", "RMSE_Veg", "mse_ndvi", "loss", "veg_score"]:
            if candidate in data[0]:
                sort_key = candidate
                break

        if sort_key is not None:
            lower_is_better = any(
                token in sort_key.lower()
                for token in ["rmse", "mse", "loss"]
            )

            sorted_data = sorted(
                data,
                key=lambda d: float(d.get(sort_key, float("inf"))),
                reverse=not lower_is_better,
            )

            with open(self.pred_dir / f"individual_scores_sorted_by_{sort_key}.json", "w") as fp:
                json.dump(sorted_data, fp, indent=2)

        return
    
    # def load_model_weights_from_checkpoint(self, checkpoint_path, strict=True):
    #     device = next(self.model.parameters()).device

    #     checkpoint = torch.load(
    #         checkpoint_path,
    #         map_location=device,
    #         weights_only=False,  # safer for Lightning checkpoints
    #     )

    #     state_dict = checkpoint["state_dict"]

    #     # Lightning task checkpoint usually stores model weights as "model.xxx"
    #     model_state_dict = {}
    #     for key, value in state_dict.items():
    #         if key.startswith("model."):
    #             model_state_dict[key[len("model.") :]] = value

    #     if len(model_state_dict) == 0:
    #         raise ValueError(
    #             "No keys starting with 'model.' found in checkpoint. "
    #             "Check checkpoint['state_dict'].keys()."
    #         )

    #     missing, unexpected = self.model.load_state_dict(
    #         model_state_dict,
    #         strict=strict,
    #     )

    #     print("Loaded model weights from checkpoint.")
    #     print("Missing keys:", missing)
    #     print("Unexpected keys:", unexpected)
    
    def load_model_weights_from_checkpoint(self, checkpoint_path):
        device = next(self.model.parameters()).device

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint["state_dict"]

        # Remove Lightning's "model." prefix
        checkpoint_state_dict = {
            key[len("model."):]: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }

        if len(checkpoint_state_dict) == 0:
            raise ValueError(
                "No keys starting with 'model.' found in checkpoint. "
                "Check checkpoint['state_dict'].keys()."
            )

        # Current model parameters
        current_state_dict = self.model.state_dict()

        compatible_state_dict = {}
        skipped = []

        for key, checkpoint_value in checkpoint_state_dict.items():

            # Key doesn't exist in current model
            if key not in current_state_dict:
                skipped.append(
                    (key, tuple(checkpoint_value.shape), "key not in current model")
                )
                continue

            current_value = current_state_dict[key]

            # Same name but different shape
            if checkpoint_value.shape != current_value.shape:
                skipped.append(
                    (
                        key,
                        tuple(checkpoint_value.shape),
                        tuple(current_value.shape),
                    )
                )
                continue

            compatible_state_dict[key] = checkpoint_value

        # Load only compatible parameters
        missing, unexpected = self.model.load_state_dict(
            compatible_state_dict,
            strict=False,
        )

        print(f"Loaded {len(compatible_state_dict)} tensors from checkpoint.")

        if skipped:
            print("\nSkipped incompatible tensors:")
            for item in skipped:
                print("  ", item)

        print("\nMissing keys:", missing)
        print("Unexpected keys:", unexpected)
