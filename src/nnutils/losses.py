# TODO: loss functions that will be used for the training

import torch

MSE_loss = torch.nn.MSELoss()


def geometric_cycle_consistency_loss(gt_2d_pos_grid, pred_positions, mask):

    gt = torch.mul(mask, gt_2d_pos_grid)
    pred = torch.mul(mask, pred_positions)

    return MSE_loss(gt, pred)


def visibility_constraint_loss(pred_depths, pred_z):

    return


def mask_reprojection_loss(mask, pred_masks):

    return


def diverse_loss(pred_poses):

    return

