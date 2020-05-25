# TODO: loss functions that will be used for the training

import torch


def geometric_cycle_consistency_loss(gt_2d_pos_grid, pred_positions, mask):

    """
    Calculates the 2D l2 loss between the predicted 2D positions and ground truth positions

    :param gt_2d_pos_grid: The ground truth positions with values between -
    1 to 1 along both the axis. (W X H X 2)
    :param pred_positions: The predicted 2D positions (B X CP X W X H X 2) in camera frame
    :param mask: The ground truth fore ground mask
    :return: The geometric cycle consistency loss
    """

    gt = torch.mul(mask.unsqueeze(1), gt_2d_pos_grid)
    pred = torch.mul(mask.unsqueeze(1), pred_positions)

    return torch.nn.functional.mse_loss(gt, pred)


def visibility_constraint_loss(pred_depths, pred_z, mask):

    """
    Calculates the visibility constraint loss between the z values for the predicted positions
    and the depth values rendered for the camera poses.

    loss = max(0, z-depth)

    :param pred_depths: Depths rendered either by the predicted camera poses
    or the ground truth camera pose (B X CP X W X H)
    :param pred_z: Z values for the predicted positions in camera frame (B X CP X W X H)
    :param mask: Ground truth foreground mask (B X W X H)
    :return: The visibility constraint loss
    """

    extended_mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2])
    loss = torch.mul(torch.nn.functional.relu(torch.sub(pred_z, pred_depths)).pow(2), extended_mask)

    return loss.mean()


def mask_reprojection_loss(mask, pred_masks):

    """
    Calculates the mask re-projection loss (L2) between the ground truth mask and the
    masks rendered for the predicted camera poses.

    :param mask: The ground truth mask (B X W X H)
    :param pred_masks: The rendered masks (B X CP X W X H)

    :return: The mask re-projection loss
    """

    extended_mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2])

    return torch.nn.functional.mse_loss(extended_mask, pred_masks)


def diverse_loss(pred_poses):

    return

