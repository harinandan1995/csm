import json
import os

import torch
from ax.service.ax_client import AxClient

from src.estimators.csm_trainer import CSMTrainer
from src.utils.config import ConfigParser
from src.estimators.kp_tester import KPTransferTester


def start_hpt(config_path, params, device):
    """Start hyperparameter tuning for loss weights and whether to only use the mask loss for the warmup phase."""

    config = ConfigParser(config_path, params).config

    ax_client = AxClient()

    ax_client.create_experiment(
        name="csm_loss_weights",
        parameters=[
            {
                "name": "loss/visibility",
                "type": "range",
                "bounds": [0.1, 5.0],
            },
            {
                "name": "loss/mask",
                "type": "range",
                "bounds": [1.0, 4.0],
            },
            {
                "name": "loss/diverse",
                "type": "range",
                "bounds": [0.0, 0.5],
            },
            {
                "name": "loss/quat",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "loss/mask_only",
                "type": "choice",
                "values": [False, True],
            },
            {
                "name": "num_in_chans",
                "type": "choice",
                "values": [3, 4],
            },
        ],
        objective_name="loss",
        minimize=True  # defaults to False.
    )

    out_dir_old = config.train.out_dir

    for i in range(20):
        parameters, trial_index = ax_client.get_next_trial()

        config.train.out_dir = os.path.join(
            out_dir_old, f"trial{trial_index}")

        nested_params = split_params(parameters, {})
        # update loss weights accordingly
        config.train.loss.update(nested_params.pop("loss", {}))
        # update other train parameters, e.g. number of input channels
        config.train.update(nested_params)

        print(json.dumps(config, indent=3))
        trainer = CSMTrainer(config, device)
        trainer.train()

        [geometric, visibility, mask, diverse, quat, _] = trainer.running_loss

        visibility /= config.train.loss.visibility
        mask /= config.train.loss.mask
        quat /= config.train.loss.quat
        geometric /= config.train.loss.geometric
        # arti /= config.train.loss.arti

        losses = [visibility, mask, quat, geometric, _]
        loss = sum(losses)/len(losses)

        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data={
            "loss/visibility": (visibility.item(), 0.0),
            "loss/mask": (mask.item(), 0.0),
            "loss/quat": (quat.item(), 0.0),
            "loss/geometric": (geometric.item(), 0.0),
            "loss/diverse": (diverse.item(), 0.0),
            "loss": (loss, 0, 0)
        }
        )
        
    try:
        with open("report.html", "w+") as f:
            f.write(ax_client.get_report())
    except Exception as e:
        print("writing report didt work")

        print(e)
        
def split_params(params, out):

    for k, v in params.items():
        if len(k_split := k.split("/")) > 1:
            out[k_split[0]] = split_params(
                {"/".join(k_split[1:]): v}, out.get(k_split[0], {}))
        else:
            out[k] = v
    return out
