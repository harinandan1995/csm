import torch
import torch.utils.data

from src.nnutils.losses import *
from src.nnutils.geometry import *
from src.model.trainer import ITrainer


class CSMTrainer(ITrainer):

    def __init__(self, config):

        super(CSMTrainer, self).__init__(config)
        self.gt_2d_pos_grid = get_gt_positions_grid(config.image_size)

    def calculate_loss(self, batch):

        img = batch['img']
        mask = batch['mask']
        cam_pose = batch['sfm_pose']
        batch_size = img.shape[0]

        pred_out = self.model(img, mask)

        pred_positions = pred_out['pred_positions']
        pred_poses = pred_out['pred_poses']
        pred_masks = pred_out['pred_masks']
        pred_depths = pred_out['pred_depths']
        pred_z = pred_out['pred_z']

        loss = geometric_cycle_consistency_loss(self.gt_2d_pos_grid, pred_positions, mask)
        loss += visibility_constraint_loss(pred_depths, pred_z)
        loss += mask_reprojection_loss(mask, pred_masks)
        loss += diverse_loss(pred_poses)

        return loss

    def get_data_loader(self):

        # TODO: Add the corresponding dataset once its implemented

        return torch.utils.data.DataLoader(
            None, batch_size=self.config.batch_size,
            shuffle=self.config.shuffle, num_workers=self.config.workers)

    def get_model(self):

        # TODO: Write the code to get the actual model once the model is implemented
        return "None"
