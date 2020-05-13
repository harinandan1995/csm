import torch.utils.data

from src.model.trainer import ITrainer
from src.nnutils.geometry import *
from src.nnutils.losses import *


class CSMTrainer(ITrainer):

    """
    A trainer for the Canonical Surface Mapping problem
    The following notations are used in the documentation
    W - Width of the image
    H - Height of the image
    B - Batch size of the image
    CP - Number of camera poses that are predicted

    """

    def __init__(self, config):

        """
        :param config: A dictionary containing the following parameters

        - For the parent class
        epochs: Number of epochs for the training
        checkpoint: Path to a checkpoint to pre load a model. None if no weights are to be loaded.
        optim.type: Type of the optimizer to the used during the training. Allowed values are 'adam' and 'sgd'
        optim.lr: Learning rate for the optimizer
        optim.beta1: Beta1 value for the optimizer
        - For the CSMTrainer
        img_size: A tuple (W, H) containing the width and height of the image
        batch_size: Batch size to be used in the dataloader
        shuffle: True or False. True if you want to shuffle the data during the training
        workers: Number of workers to be used for the data processing
        """

        super(CSMTrainer, self).__init__(config)
        self.gt_2d_pos_grid = get_gt_positions_grid(config.image_size)

    def calculate_loss(self, batch):

        """
        Calculates the total loss for the batch which is a combination of the
        - Geometric cycle consistency loss
        - Visibility constraint loss
        - Mask reprojection loss
        - Camera pose diverse loss

        The mask reprojection and diverse loss are only calculated if the
        ground truth camera pose is not used

        :param batch: Batch data from the data loader which is a dict containing the following parameters
        img: The input image
        mask: The ground truth foreground mask
        sfm_pose: The ground truth camera pose
        :return:
        """

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
