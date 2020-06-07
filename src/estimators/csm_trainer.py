import os.path as osp

import torch.utils.data

from src.data.cub_dataset import CubDataset
from src.data.imnet_dataset import ImnetDataset
from src.data.p3d_dataset import P3DDataset
from src.estimators.trainer import ITrainer
from src.model.csm import CSM
from src.nnutils.color_transform import sample_uv_contour
from src.nnutils.geometry import get_gt_positions_grid, convert_3d_to_uv_coordinates
from src.nnutils.losses import *
from src.utils.config import ConfigParser
from src.utils.utils import get_time


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

    def __init__(self, config: ConfigParser.ConfigObject, device='cuda'):
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
        self.data_cfg = config.dataset

        super(CSMTrainer, self).__init__(config.train)

        self.gt_2d_pos_grid = get_gt_positions_grid(
            (self.data_cfg.img_size, self.data_cfg.img_size)).to(self.device).permute(2, 0, 1)
        self.gt_2d_pos_grid = self.gt_2d_pos_grid.unsqueeze(0).unsqueeze(0)

        self.texture_map = self.dataset.texture_map
        self.template_mesh = self.dataset.template_mesh
        template_mesh_colors = self._get_template_mesh_colors()
        self.summary_writer.add_mesh('Template', self.template_mesh.verts_packed().unsqueeze(0),
                                     faces=self.template_mesh.faces_packed().unsqueeze(0),
                                     colors=template_mesh_colors)
        self.running_loss_1 = 0
        self.running_loss_2 = 0
        self.running_loss_3 = 0
        self.running_loss_4 = 0

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

        pred_z = pred_out['pred_z']
        pred_masks = pred_out['pred_masks']
        pred_depths = pred_out['pred_depths']
        pred_positions = pred_out['pred_positions']

        loss = self._calculate_loss_for_predictions(mask, pred_positions, pred_masks, pred_depths, pred_z)

        return loss, pred_out

    def _calculate_loss_for_predictions(self, mask, pred_positions, pred_masks, pred_depths, pred_z):

        loss_1 = geometric_cycle_consistency_loss(self.gt_2d_pos_grid, pred_positions, mask)
        self.running_loss_1 += loss_1
        loss = self.config.loss.geometric * loss_1

        loss_2 = visibility_constraint_loss(pred_depths, pred_z, mask)
        self.running_loss_2 += loss_2
        loss += self.config.loss.visibility * loss_2

        if not self.config.use_gt_cam:
            loss_3 = mask_reprojection_loss(mask, pred_masks)
            self.running_loss_3 += loss_3
            loss += self.config.loss.mask * loss_3
            # loss_4 = diverse_loss(pred_poses)

        return loss

    def _epoch_end_call(self, current_epoch, total_epochs, total_steps):
        # Save checkpoint after every 10 epochs
        if current_epoch % 5 == 0:
            self._save_model(osp.join(self.checkpoint_dir,
                                      'model_%s_%d' % (get_time(), current_epoch)))
        self.summary_writer.add_scalar('loss/geometric', self.running_loss_1 / total_steps, current_epoch)
        self.summary_writer.add_scalar('loss/visibility', self.running_loss_2 / total_steps, current_epoch)
        self.running_loss_1 = 0
        self.running_loss_2 = 0

        if not self.config.use_gt_cam:

            self.summary_writer.add_scalar('loss/mask', self.running_loss_3 / total_steps, current_epoch)
            self.running_loss_3 = 0
            self.running_loss_4 = 0

    def _batch_end_call(self, batch, loss, out, step, total_steps, epoch, total_epochs):
        # Print the loss at the end of each batch
        if step % 10 == 0:
            print('%d:%d/%d loss %f' % (epoch, step, total_steps, loss))

        uv = out["uv"]
        uv_3d = out["uv_3d"]
        pred_z = out['pred_z']
        pred_masks = out['pred_masks']
        pred_depths = out['pred_depths']
        pred_positions = out['pred_positions']

        img = batch['img'].to(self.device, dtype=torch.float)
        mask = batch['mask'].unsqueeze(1).to(self.device, dtype=torch.float)

        self._add_summaries(step, epoch, uv, uv_3d, img, mask,
                            pred_positions, pred_depths, pred_z, pred_masks)

    def _load_dataset(self):

        if self.data_cfg.category == 'car':
            self.dataset = P3DDataset(self.data_cfg, self.device)
        elif self.data_cfg.category == 'bird':
            self.dataset = CubDataset(self.data_cfg, self.device)
        else:
            self.dataset = ImnetDataset(self.data_cfg, self.device)

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
                    self.dataset.mean_shape,
                    self.config.use_gt_cam,
                    self.device).to(self.device)

        return model

    def _add_summaries(self, step, epoch, uv, uv_3d, img, mask, pred_positions,
                       pred_depths, pred_z, pred_masks):
        """
        Adds image summaries to the summary writer

        :param step: Current optimization step number (Batch number)
        :param epoch: Current epoch
        :param uv: A (B X 2 X H X W) tensor of UV values for the batch
        :param uv_3d: A (B X H X W X 3) tensor of 3D coordinates for the UV values
        :param img: A (B X 3 X H X W) tensor of input image
        :param mask: A (B X 1 X H X W) tensor of input foreground mask
        :param pred_depths: A (B X CP X 1 X H X W) tensor of depths rendered with the
            gt. camera pose /predicted camera poses (if config.use_gt_cam_pose) is true
        :param pred_z: A (B X CP X 1 X H X W) tensor of z values of the projected 3D points
            for the predicted UV values
        :param pred_masks: A (B X CP X 1 X H X W) tensor of masks  rendered with the
            gt. camera pose /predicted camera poses
        """

        if step % 20 == 0:

            sum_step = step % 20

            loss_values = torch.mean(geometric_cycle_consistency_loss(
                self.gt_2d_pos_grid, pred_positions, mask, reduction='none'), dim=2)
            loss_values = loss_values / 0.1
            self.summary_writer.add_images('loss/geometric', loss_values, sum_step)

            uv_color, uv_blend = sample_uv_contour(img, uv.permute(0, 2, 3, 1), self.texture_map, mask)
            self.summary_writer.add_images('data/img', img, sum_step)
            self.summary_writer.add_images('data/mask', mask, sum_step)

            self.summary_writer.add_images('pred/uv_blend', uv_blend, sum_step)
            self.summary_writer.add_images('pred/uv', uv_color * mask, sum_step)
            depth = (pred_depths - pred_depths.min())/(pred_depths.max()-pred_depths.min())
            z = (pred_z - pred_z.min()) / (pred_z.max() - pred_z.min())
            self.summary_writer.add_images('pred/depth', depth.view(-1, 1, depth.size(-2), depth.size(-1)), sum_step)
            self.summary_writer.add_images('pred/z', z.view(-1, 1, z.size(-2), z.size(-1)), sum_step)
            self.summary_writer.add_images('pred/mask', pred_masks.view(-1, 1, pred_masks.size(-2), pred_masks.size(-1)), sum_step)

    def _get_template_mesh_colors(self):

        vertices = self.template_mesh.verts_packed()
        vertices_uv = convert_3d_to_uv_coordinates(vertices)
        colors = torch.nn.functional.grid_sample(self.texture_map.unsqueeze(0),
                                                 2*vertices_uv.unsqueeze(0).unsqueeze(0)-1)

        colors = colors.squeeze(2).permute(0, 2, 1) * 255

        return colors.to(torch.int)
