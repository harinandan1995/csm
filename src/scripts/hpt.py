import json
import os

import torch
from ax.service.ax_client import AxClient

from src.estimators.csm_trainer import CSMTrainer
from src.utils.config import ConfigParser


def start_hpt(config_path, params, device):
    """Start hyperparameter tuning for loss weights and whether to only use the mask loss for the warmup phase."""

    config = ConfigParser(config_path, params).config

    ax_client = AxClient()

    ax_client.create_experiment(
        name="csm_loss_weights",
        parameters=[
            {
                "name": "visibility",
                "type": "range",
                "bounds": [0.1, 5.0],
            },
            {
                "name": "mask",
                "type": "range",
                "bounds": [1.0, 5.0],
            },
            {
                "name": "diverse",
                "type": "range",
                "bounds": [0.0, 0.1],
            },
            {
                "name": "quat",
                "type": "range",
                "bounds": [0.0, 5.0],
            },
            {
                "name": "mask_only",
                "type": "choice",
                "values": [False, True],
            },
        ],
        objective_name="loss",
        minimize=True  # Optional, defaults to False.
    )

    for i in range(20):
        parameters, trial_index = ax_client.get_next_trial()

        # update loss weights accordingly
        config.train.loss.update(parameters)

        print(json.dumps(config, indent=3))
        trainer = CSMTrainer(config, device)
        [geometric, visibility, mask, diverse, quat, arti] = trainer.train()

        visibility /= config.train.loss.visibility
        mask /= config.train.loss.mask
        quat /= config.train.loss.quat
        geometric /= config.train.loss.geometric
        arti /= config.train.loss.arti

        loss = sum([visibility, mask, quat, geometric, arti])
        print(f"overall loss:  {loss}")

        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data={
            "visibility": (visibility.item(), 0.0),
            "mask": (mask.item(), 0.0),
            "quat": (quat.item(), 0.0),
            "geometric": (geometric.item(), 0.0),
            "diverse": (diverse.item(), 0.0),
            "loss": (loss, 0, 0)
        }
        )
