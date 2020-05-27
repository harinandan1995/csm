import os.path as osp

import numpy as np
import torch.utils.data
from PIL import Image

from src.data.cub_dataset import CubDataset
from src.estimators.trainer import ITrainer
from src.model.csm import CSM
from src.nnutils.color_transform import sample_uv_contour
from src.nnutils.geometry import get_gt_positions_grid
from src.nnutils.losses import *
from src.utils.utils import get_date, get_time, create_dir_if_not_exists


class CSMTrainer(ITrainer):
    """
    A trainer for the Canonical Surface Mapping problem
    The following notations are used in the documentation
    W - Width of the image
    H - Height of the image
    B - Batch size of the image
    CP - Number of camera pose hypothesis. CP is 1 if only one pose is predicted or
    if the ground truth pose is used during the training

    """

    def __init__(self, config, device='cuda'):
        """
        :param config: A dictionary containing the following parameters
            template: Path to the mesh template for the data as an obj file
            epochs: Number of epochs for the training
            checkpoint: Path to a checkpoint to pre load a model. None if no weights are to be loaded.
            optim.type: Type of the optimizer to the used during the training. Allowed values are 'adam' and 'sgd'
            optim.lr: Learning rate for the optimizer
            optim.beta1: Beta1 value for the optimizer
            out_dir: Path to the directory where the summaries and the checkpoints should be stored
            batch_size: Batch size to be used in the dataloader
            shuffle: True or False. True if you want to shuffle the data during the training
            workers: Number of workers to be used for the data processing
        :param device: Device to store the tensors. Default: cuda
        """
        self.device = torch.device(device)
        self.dataset = CubDataset(config.dataset, self.device)
        self.gt_2d_pos_grid = get_gt_positions_grid(
            (config.dataset.img_size, config.dataset.img_size)).to(self.device).permute(2, 0, 1)
        self.gt_2d_pos_grid = self.gt_2d_pos_grid.unsqueeze(0).unsqueeze(0)
        self.texture_map = self._get_texture_map(config.dataset.dir.texture)

        super(CSMTrainer, self).__init__(config.train)

        self.template_mesh = self.dataset.template_mesh
        self.summary_writer.add_mesh('Template', self.template_mesh.verts_packed().unsqueeze(0),
                                     faces=self.template_mesh.faces_packed().unsqueeze(0))

        self.checkpoint_dir = osp.join(self.config.out_dir, "checkpoints",
                                       get_date(), get_time())
        create_dir_if_not_exists(self.checkpoint_dir)

    def _calculate_loss(self, step, batch, epoch):
        """
        Calculates the total loss for the batch which is a combination of the
        - Geometric cycle consistency loss
        - Visibility constraint loss
        - Mask re-projection loss
        - Camera pose diverse loss

        The mask re-projection and diverse loss are only calculated if the
        ground truth camera pose is not used

        :param batch: Batch data from the data loader which is a dict containing the following parameters
        img: The input image
        mask: The ground truth foreground mask
        sfm_pose: The ground truth camera pose
        use_gt_cam_pos: True or False. False if you want to model to predict the camera poses as well.

        :return: The total loss calculated for the batch
        """
        img = batch['img'].to(self.device, dtype=torch.float)
        mask = batch['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        scale = batch['scale'].to(self.device, dtype=torch.float)
        trans = batch['trans'].to(self.device, dtype=torch.float)
        quat = batch['quat'].to(self.device, dtype=torch.float)

        pred_out = self.model(img, mask, scale, trans, quat)

        uv = pred_out["uv"]
        uv_3d = pred_out["uv_3d"]
        pred_positions = pred_out['pred_positions']
        pred_depths = pred_out['pred_depths']
        pred_z = pred_out['pred_z']
        # pred_poses = pred_out['pred_poses']
        pred_masks = pred_out['pred_masks']

        loss = geometric_cycle_consistency_loss(self.gt_2d_pos_grid, pred_positions, mask)
        loss += 0.001 * visibility_constraint_loss(pred_depths, pred_z, mask)
        # loss += mask_reprojection_loss(mask, pred_masks)
        # loss += diverse_loss(pred_poses)

        self._add_summaries(step, epoch, uv, uv_3d, img, mask, pred_depths, pred_masks)

        return loss

    def _epoch_end_call(self, current_epoch, total_epochs):
        # Save checkpoint after every 10 epochs
        if current_epoch % 10 == 0:
            self._save_model(osp.join(self.checkpoint_dir,
                                      'model_%s_%d' % (get_time(), current_epoch)))

    def _batch_end_call(self, batch, loss, step, total_steps, epoch, total_epochs):
        # Print the loss at the end of each batch
        print('%d:%d/%d loss %f' % (epoch, step, total_steps, loss))

    def _get_data_loader(self):

        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config.batch_size,
            shuffle=self.config.shuffle, num_workers=self.config.workers)

    def _get_model(self):
        """
        Returns a torch model which takes image(B X W X H) and mask (B X W X H) and returns a
        dictionary containing the following parameters

        pred_positions: A (B X CP X W X H X 2) tensor with the final projected positions in camera frame
        after performing 2D to 3D to 2D transformations
        pred_depths: A (B X CP X W X H) tensor with the depths rendered either using the predicted camera poses
        or the ground truth pose
        pred_z: A (B X CP X W X H) tensor with the z values in the camera frame for the positions predicted by the model
        pred_poses: A (B X CP X 6) tensor containing the predicted camera poses if config.use_gt_cam_pos is False.
        pred_masks: A (B X CP X W X H) tensor with masks rendered using the predicted camera poses.
        Is used only if config.use_gt_cam_pos is False

        :return: A torch model satisfying the above input output structure
        """

        model = CSM(self.dataset.template_mesh,
                    self.dataset.mean_shape, self.device).to(self.device)

        return model

    def _add_summaries(self, step, epoch, uv, uv_3d, img, mask, pred_depths, pred_masks):
        """
        Adds image summaries to the summary writer

        :param step: Current optimization step number (Batch number)
        :param epoch: Current epoch
        :param uv: A (B X 2 X H X W) tensor of UV values for the batch
        :param uv_3d: A (B X None X 3) tensor of 3D coordinates for the UV values
        :param img: A (B X 3 X H X W) tensor of input image
        :param mask: A (B X 1 X H X W) tensor of input foreground mask
        :param pred_depths: A (B X 3 X H X W) tensor of depths rendered with the
            gt. camera pose /predicted camera poses (if config.use_gt_cam_pose) is true
        :param pred_masks: A (B X 3 X H X W) tensor of masks  rendered with the
            gt. camera pose /predicted camera poses
        """

        if step % 20 == 0:

            uv_color, uv_blend = sample_uv_contour(img, uv.permute(0, 2, 3, 1), self.texture_map, mask)
            self.summary_writer.add_images('%d/out/img' % epoch, img, step % 20)
            self.summary_writer.add_images('%d/out/mask' % epoch, mask, step % 20)
            self.summary_writer.add_images('%d/out/uv_blend' % epoch, uv_blend, step % 20)
            self.summary_writer.add_images('%d/out/uv' % epoch, uv_color * mask, step % 20)
            self.summary_writer.add_images('%d/out/depth' % epoch, pred_depths, step % 20)
            self.summary_writer.add_images('%d/out/pred_mask' % epoch, pred_masks, step % 20)

    def _get_texture_map(self, img_path):

        texture_map = Image.open(img_path).resize((256, 256), Image.ANTIALIAS)
        texture_map = torch.from_numpy(np.asarray(texture_map)).permute(2, 0, 1)

        return texture_map.to(self.device, dtype=torch.float) / 255
