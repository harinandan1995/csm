import os.path as osp

import numpy as np
import torch.utils.data

from src.data.cub_dataset import CubDataset
from src.data.imnet_dataset import ImnetDataset
from src.data.p3d_dataset import P3DDataset
from src.estimators.trainer import ITrainer
from src.model.csm import CSM
from src.nnutils.color_transform import sample_uv_contour, draw_key_points
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

    def __init__(self, config: ConfigParser.ConfigObject, device):
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
        self.key_point_colors = np.random.uniform(0, 1, (len(self.dataset.kp_names), 3))

        # Running losses to calculate mean loss per epoch for all types of losses
        self.running_loss = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)
        if self.config.use_gt_cam:
            self.config.pose_warmup_epochs = 0

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

        pred_out = self.model(img, mask, scale, trans, quat, epoch)

        loss = self._calculate_loss_for_predictions(mask, pred_out, epoch < self.config.pose_warmup_epochs, epoch < self.config.arti_epochs)

        return loss, pred_out

    def _calculate_loss_for_predictions(self, mask: torch.tensor, pred_out: dict, epoch : int, pose_warmup: bool = False, not_arti: bool = True) -> torch.tensor:
        """Calculates the loss from the output

        :param mask: (B X 1 X H X W) foreground mask
        :param pred_out: A dictionary cotaining the output of the CSM model
        :param epoch : The number of epoch during the training
        :return: The computed loss from the output for the batch
        """

        pred_z = pred_out['pred_z']
        pred_masks = pred_out['pred_masks']
        pred_depths = pred_out['pred_depths']
        pred_positions = pred_out['pred_positions']

        loss = torch.zeros_like(self.running_loss)

        prob_coeffs = None
        if not self.config.use_gt_cam and not self.config.use_sampled_cam:
            _, _, _, prob_coeffs = pred_out['pred_poses']
            prob_coeffs = torch.add(prob_coeffs, 0.1)

        if self.config.loss.geometric > 0 and not pose_warmup:
            loss[0] = self.config.loss.geometric * geometric_cycle_consistency_loss(
                self.gt_2d_pos_grid, pred_positions, mask, coeffs=prob_coeffs)
        
        if self.config.loss.visibility > 0 and not pose_warmup:
            loss[1] = self.config.loss.visibility * visibility_constraint_loss(
                pred_depths, pred_z, mask, coeffs=prob_coeffs)

        if not self.config.use_gt_cam:
            _, _, pred_quat, pred_prob = pred_out['pred_poses']
            if self.config.loss.mask > 0: 
                loss[2] = self.config.loss.mask * mask_reprojection_loss(mask, pred_masks, coeffs=prob_coeffs)
            if self.config.loss.diverse > 0 and not pose_warmup:
                loss[3] = self.config.loss.diverse * diverse_loss(pred_prob)
            if self.config.loss.quat > 0 and not pose_warmup:
                loss[4] = self.config.loss.quat * quaternion_regularization_loss(pred_quat)

        #TODO:add config: use_arti & loss.arti & arti_threshold
        if self.config.use_arti and not not_arti:
            pred_arti_translation = pred_out["pred_arti_translation"]
            loss[5] = self.config.loss.arti * articulation_loss(pred_arti_translation)

        self.running_loss = torch.add(self.running_loss, loss)

        return loss.sum()

    def _epoch_end_call(self, current_epoch, total_epochs, total_steps):
        # Save checkpoint after every 10 epochs
        if current_epoch % 5 == 0:
            self._save_model(osp.join(self.checkpoint_dir,
                                      'model_%s_%d' % (get_time(), current_epoch)))
	
        self.running_loss = torch.true_divide(self.running_loss, total_steps)

        # Add loss summaries & reset the running losses
        self.summary_writer.add_scalar('loss/geometric', self.running_loss[0], current_epoch)
        self.summary_writer.add_scalar('loss/visibility', self.running_loss[1], current_epoch)

        if not self.config.use_gt_cam:

            self.summary_writer.add_scalar('loss/mask', self.running_loss[2], current_epoch)
            self.summary_writer.add_scalar('loss/diverse', self.running_loss[3], current_epoch)
            self.summary_writer.add_scalar('loss/quat', self.running_loss[4], current_epoch)

        self.running_loss = torch.zeros_like(self.running_loss)

    def _batch_end_call(self, batch, loss, out, step, total_steps, epoch, total_epochs):

        self._add_summaries(step, epoch, out, batch)

    def _load_dataset(self) -> torch.utils.data.Dataset:
        """
        Returns the dataset based on the input category
        """

        if self.data_cfg.category == 'car':
            return P3DDataset(self.data_cfg, self.device)
        elif self.data_cfg.category == 'bird':
            return CubDataset(self.data_cfg, self.device)
        else:
            return ImnetDataset(self.data_cfg, self.device)

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

        model = CSM(self.dataset.template_mesh, self.dataset.mean_shape, self.config.use_gt_cam, 
                    self.config.num_cam_poses, self.config.use_sampled_cam, self.config.use_arti,
                    self.config.arti_epochs, self.dataset.arti_info_mesh).to(self.device)

        return model

    def _add_summaries(self, step, epoch, out, batch):
        """
        Adds image summaries to the summary writer

        :param step: Current optimization step number (Batch number)
        :param epoch: Current epoch
        :param out: A dictionary containing the output from the model
        :param batch: A dictionary containing the batched inputs to the model
        """

        uv = out["uv"]
        pred_z = out['pred_z']
        pred_masks = out['pred_masks']
        pred_depths = out['pred_depths']
        pred_positions = out['pred_positions']

        img = batch['img'].to(self.device, dtype=torch.float)
        mask = batch['mask'].unsqueeze(1).to(self.device, dtype=torch.float)

        sum_step = int(step / self.config.log.image_summary_step)

        if step % self.config.log.image_summary_step == 0 and epoch % self.config.log.image_epoch == 0:

            self._add_loss_vis(pred_positions, mask, epoch, sum_step)
            self._add_input_vis(img, mask, epoch, sum_step)
            self._add_pred_vis(uv, pred_z, pred_depths, pred_masks, img, mask, epoch, sum_step)
            
    def _add_loss_vis(self, pred_positions, mask, epoch, sum_step):
        """
        Add loss visualizations to the tensorboard summaries
        """

        loss_values = torch.mean(geometric_cycle_consistency_loss(
            self.gt_2d_pos_grid, pred_positions, mask, reduction='none'), dim=2, keepdim=True)
        
        loss_values = loss_values.view(-1, 1, loss_values.size(-2), loss_values.size(-2))

        loss_values = (loss_values - loss_values.min())/(loss_values.max()-loss_values.min())
        self.summary_writer.add_images('%d/pred/geometric' % epoch, loss_values, sum_step)

    def _add_input_vis(self, img, mask, epoch, sum_step):
        """
        Add input data (img, mask) visualizations to the tensorboard summaries
        """

        self.summary_writer.add_images('%d/data/img' % epoch, img, sum_step)
        self.summary_writer.add_images('%d/data/mask' % epoch, mask, sum_step)
    
    def _add_pred_vis(self, uv, pred_z, pred_depths, pred_masks, img, mask, epoch, sum_step):
        """
        Add predicted output (depth, uv, masks) visualizations to the tensorboard summaries
        """

        uv_color, uv_blend = sample_uv_contour(img, uv.permute(0, 2, 3, 1), self.texture_map, mask)
        self.summary_writer.add_images('%d/pred/uv_blend' % epoch, uv_blend, sum_step)
        self.summary_writer.add_images('%d/pred/uv' % epoch, uv_color * mask, sum_step)
        
        depth = (pred_depths - pred_depths.min()) / (pred_depths.max()-pred_depths.min())
        self.summary_writer.add_images('%d/pred/depth' % epoch, depth.view(-1, 1, depth.size(-2), depth.size(-1)), sum_step)
        
        z = (pred_z - pred_z.min()) / (pred_z.max() - pred_z.min())
        self.summary_writer.add_images('%d/pred/z' % epoch, z.view(-1, 1, z.size(-2), z.size(-1)), sum_step)

        self.summary_writer.add_images(
            '%d/pred/mask' % epoch, pred_masks.view(-1, 1, pred_masks.size(-2), pred_masks.size(-1)), sum_step)

    def _get_template_mesh_colors(self):
        """
        Creates the colors for the template mesh using the texture map. These colors can be used for tensorboard mesh summary.
        """

        vertices = self.template_mesh.verts_packed()
        vertices_uv = convert_3d_to_uv_coordinates(vertices)
        colors = torch.nn.functional.grid_sample(self.texture_map.unsqueeze(0),
                                                 2*vertices_uv.unsqueeze(0).unsqueeze(0)-1)

        colors = colors.squeeze(2).permute(0, 2, 1) * 255

        return colors.to(torch.int)
