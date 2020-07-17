import itertools

import numpy as np
import torch

from src.nnutils.geometry import hamilton_product, quat2ang, quat_conj


def geometric_cycle_consistency_loss(gt_2d_pos_grid, pred_positions, mask, coeffs=None, reduction='mean'):
    """
    Calculates the 2D l2 loss between the predicted 2D positions and ground truth positions

    :param gt_2d_pos_grid: (1 x 1 x 2 x H X W) - The ground truth positions with values between -
    1 to 1 along both the axis.
    :param pred_positions: (B X CP X 2 X H X W) - The predicted 2D positions in camera frame
    :param mask: (B X 1 X H X W) -The ground truth fore ground mask
    :param coeffs: (B X CP) containing coefficients for the camera poses. If None all the coefficients are set to 1
    :param reduction: 'mean' - mean value is returned.
        'none' - tensor of shape same as the input is returned
    :return: The geometric cycle consistency loss
    """

    device = gt_2d_pos_grid.device

    if coeffs is None:
        coeffs = torch.ones((1, 1, 1, 1, 1)).to(device)
    else:
        coeffs = coeffs.view(coeffs.size(0), -1, 1, 1, 1)

    gt = torch.mul(mask.unsqueeze(1), gt_2d_pos_grid)
    pred = torch.mul(mask.unsqueeze(1), pred_positions)

    loss = coeffs * torch.sub(gt, pred).pow(2)
    
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss


def visibility_constraint_loss(pred_depths, pred_z, mask, coeffs=None, reduction='mean'):
    """
    Calculates the visibility constraint loss between the z values for the predicted positions
    and the depth values rendered for the camera poses.

    loss = max(0, z-depth)

    :param pred_depths: Depths rendered either by the predicted camera poses
    or the ground truth camera pose (B X CP X 1 X W X H)
    :param pred_z: Z values for the predicted positions in camera frame (B X CP X 1 X W X H)
    :param mask: Ground truth foreground mask (B X 1 X W X H)
    :param reduction: 'mean' - mean value is returned.
        'none' - tensor of shape same as the input is returned
    :return: The visibility constraint loss
    """

    device = pred_depths.device

    if coeffs is None:
        coeffs = torch.ones((1, 1, 1, 1, 1)).to(device)
    else:
        coeffs = coeffs.view(coeffs.size(0), -1, 1, 1, 1)

    extended_mask = mask.unsqueeze(1)
    loss = torch.nn.functional.relu(torch.sub(pred_z, pred_depths)).pow(2) * extended_mask * coeffs

    if reduction == 'mean':
        return loss.mean()
    else:
        return loss


def mask_reprojection_loss(mask, pred_masks, coeffs=None, reduction='mean'):
    """
    Calculates the mask re-projection loss (L2) between the ground truth mask and the
    masks rendered for the predicted camera poses.

    :param mask: The ground truth mask (B X 1 X W X H)
    :param pred_masks: The rendered masks (B X CP X 1 X W X H)

    :return: The mask re-projection loss
    """

    device = mask.device

    if coeffs is None:
        coeffs = torch.ones((1, 1, 1, 1, 1)).to(device)
    else:
        coeffs = coeffs.view(coeffs.size(0), -1, 1, 1, 1)

    extended_mask = mask.unsqueeze(1)
    loss = coeffs * torch.sub(extended_mask, pred_masks).pow(2)

    if reduction == 'mean':
        return loss.mean()
    else:
        return loss


def quaternion_regularization_loss(quats: torch.tensor):
    """
    A regularization loss for the quaternions. Only if number of camera poses per batch is > 1
    :param device: Torch device. Default: cuda
    :param quats: B X CP X 4
    :return: The quaternion regularization loss
    """

    device = quats.device
    num_cam_poses = quats.size(1)

    if num_cam_poses == 0:
        return 0

    quat_perm = list(itertools.permutations(range(num_cam_poses), 2))
    quat_perm = torch.tensor(quat_perm, dtype=torch.long).to(device).permute(1, 0)

    quats_x = torch.gather(quats, dim=1, index=quat_perm[0].view(1, -1, 1).repeat(len(quats), 1, 4))
    quats_y = torch.gather(quats, dim=1, index=quat_perm[1].view(1, -1, 1).repeat(len(quats), 1, 4))
    inter_quats = hamilton_product(quats_x, quat_conj(quats_y))
    
    quatAng = quat2ang(inter_quats).view(len(inter_quats), num_cam_poses - 1, -1)
    quatAng = torch.abs(quatAng - np.pi)
    quatAng = torch.nn.functional.max_pool1d(quatAng.permute(0, 2, 1), num_cam_poses - 1, stride=1)
    
    # quatAng = -1 * torch.nn.functional.max_pool1d(
    #     -1 * quatAng.permute(0, 2, 1), num_cam_poses - 1, stride=1).squeeze()

    return quatAng.mean()


def diverse_loss(probs):
    """
    Diverse loss for the poses
    :param probs: B X CP probabilities for each camera pose
    :return: Sum of entropy of the probabilities
    """

    entropy = torch.log(probs + 1E-9) * probs
    entropy = entropy.sum(1).mean()

    return entropy
