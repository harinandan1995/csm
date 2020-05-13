# TODO: loss functions that will be used for the training

import torch

MSE_loss = torch.nn.MSELoss()


def geometric_cycle_consistency_loss(gt_2d_pos_grid, pred_positions, mask):

    """
    Calculates the 2D l2 loss between the predicted 2D positions and ground truth positions

    :param gt_2d_pos_grid: The ground truth positions with values between -
    1 to 1 along both the axis. (W X H X 2)
    :param pred_positions: The predicted 2D positions (B X CP X W X H X 2)
    :param mask: The ground truth fore ground mask
    :return: The geometric cycle consistency loss
    """

    gt = torch.mul(mask, gt_2d_pos_grid)
    pred = torch.mul(mask, pred_positions)

    return MSE_loss(gt, pred)


def visibility_constraint_loss(pred_depths, pred_z):

    return


def mask_reprojection_loss(mask, pred_masks):

    return


def diverse_loss(pred_poses):

    return

