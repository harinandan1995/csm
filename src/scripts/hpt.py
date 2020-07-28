import json
import os

import torch
from ax.service.ax_client import AxClient

from src.estimators.csm_trainer import CSMTrainer
from src.utils.config import ConfigParser


def start_hpt(config_path, params, device):

    config = ConfigParser(config_path, params).config

    ax_client = AxClient()

    ax_client.create_experiment(
        name="hartmann_test_experiment",
        parameters=[
            {
                "name": "visibility",
                "type": "range",
                "bounds": [0.1, 5.0],
                # Optional, defaults to inference from type of "bounds".
                "value_type": "float",
                "log_scale": False,  # Optional, defaults to False.
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
    # l = os.listdir(config.train.out_dir)
    # if not len(l):

    #   config.train.out_dir = os.path.join(
    #        config.train.out_dir, "0")
    # else:

    #   config.train.out_dir = os.path.join(
    #      config.train.out_dir, l[-1])

    for i in range(20):
        parameters, trial_index = ax_client.get_next_trial()
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
