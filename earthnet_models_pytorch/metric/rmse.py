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

import torch
from torchmetrics import Metric


class RootMeanSquaredError(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(
        self,
        lc_min: int,
        lc_max: int,
        context_length: int,
        target_length: int,
        n_pred_channels: int = 1,
        n_targ_channels: int = 1,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
    ):
        """
        Initialize the RootMeanSquaredError metric.

        Parameters
        ----------
        lc_min : int
            Minimum value bound for the landcover mask.
        lc_max : int
            Maximum value bound for the landcover mask.
        context_length : int
            Length of the context for predictions.
        target_length : int
            Length of the target sequence for predictions.
        dist_sync_on_step : bool, optional
            If True, synchronize metric computation across devices in a distributed environment (default is False).
        process_group : optional
            Process group for distributed training (default is None).
        dist_sync_fn : optional
            Function for synchronizing metric computation across devices (default is None).
        """
        super().__init__(
            # Advanced metric settings
            dist_sync_on_step=dist_sync_on_step,  # distributed environment, if the metric should synchronize between different devices every time forward is called
            process_group=process_group,  # distributed environment, by default we synchronize across the world i.e. all processes being computed on. Specify exactly what devices should be synchronized over
            dist_sync_fn=dist_sync_fn,  # distributed environment, by default we use torch.distributed.all_gather() to perform the synchronization between devices.
        )

        # State variables for the metric
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("current_scores", default=torch.tensor(0), dist_reduce_fx=None)

        self.context_length = context_length
        self.target_length = target_length
        self.n_pred_channels = n_pred_channels
        self.n_targ_channels = n_targ_channels

        self.lc_min = lc_min
        self.lc_max = lc_max
        print(
            f"Using Masked RootMeanSquaredError metric Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
        )

    def update(self, preds, targs):
        """
        Update the state variables of the metric given the input predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted tensor from the model.
        targs : dict
            Dictionary containing the target data.
        """

        # Targets for the RMSE computation
        targets = targs["dynamic"][2][
            :, self.context_length : self.context_length + self.target_length, :self.n_targ_channels] #.unsqueeze(2)

        # Prediction NDVI: [B,T,1,H,W]
        preds = preds[:, -targets.shape[1] :, :self.n_pred_channels]
        if preds.shape != targets.shape:
            raise RuntimeError(
                f"Prediction shape {preds.shape} does not match RMSE target "
                f"shape {targets.shape}."
            )

        # Masks on the non-vegetation pixels
        if len(targs["dynamic_mask"]) > 0:
            s2_mask = targs["dynamic_mask"][0][
                :,
                self.context_length : self.context_length + self.target_length,
                ...,
            ]
            if s2_mask.ndim == 4:
                s2_mask = s2_mask.unsqueeze(2)
            if s2_mask.ndim != 5:
                raise RuntimeError(
                    "Expected dynamic mask [B,T,H,W] or [B,T,1,H,W], got "
                    f"{s2_mask.shape}."
                )
            s2_mask = (s2_mask < 1.0).type_as(preds)
        else:
            s2_mask = torch.ones_like(targets)

        # Landcover mask
        lc = targs["landcover"]
        lc_mask = ((lc >= self.lc_min) & (lc <= self.lc_max)).type_as(s2_mask)
        if lc_mask.ndim == 3:
            lc_mask = lc_mask.unsqueeze(1)
        if lc_mask.ndim != 4:
            raise RuntimeError(
                f"Expected landcover [B,H,W] or [B,1,H,W], got {lc_mask.shape}."
            )
        lc_mask = lc_mask.unsqueeze(1).expand_as(s2_mask)

        mask = s2_mask * lc_mask

        # Compute the Mean Squared Error (MSE) for the masked regions
        sum_squared_error = torch.pow((preds - targets) * mask, 2).sum((1, 2, 3, 4))
        n_obs = (mask == 1).sum((1, 2, 3, 4))  # sum of pixel with vegetation

        # Update the states variables
        self.current_scores = torch.sqrt(sum_squared_error / n_obs.clamp_min(1.0))
        self.sum_squared_error += sum_squared_error.sum()
        self.total += n_obs.sum()

    def compute(self):
        """
        Compute the final Root Mean Squared Error (RMSE) over the state of the metric.

        Returns
        -------
        dict
            Dictionary containing the computed RMSE for vegetation pixels.
        """
        return {
            "RMSE_Veg": torch.sqrt(
                self.sum_squared_error / self.total.clamp_min(1)
            )
        }

    def compute_batch(self, targs):
        """
        Compute the final RMSE for each sample of a batch over the state of the metric.

        Parameters
        ----------
        targs : dict
            Dictionary containing the target data.

        Returns
        -------
        list
            List of dictionaries containing RMSE scores for each sample in the batch.
        """
        cubenames = targs["cubename"]
        return [
            {
                "name": cubenames[i],
                "rmse": self.current_scores[i],
            }
            for i in range(len(cubenames))
        ]
