from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (euler_angles_to_matrix,
                                  matrix_to_euler_angles, matrix_to_quaternion,
                                  quaternion_raw_multiply,
                                  quaternion_to_matrix)

from src.nnutils.blocks import net_init


class CameraPredictor(nn.Module):
    """
    Camera predictor for camera pose prediction. It predicts a camera pose in the form of quaternions, scale scalar
    and translation vector based on an image.
    """

    def __init__(self, num_feats: int = 100, scale_bias: float = 1., scale_lr: float = 0.2):
        """
        :param encoder: An feature extractor of an image. If None, resnet18 will bes used.
        :param num_feats: The number of extracted features from the encoder.
        :param scale_bias: bias term in R, which is added to the scale prediction.
        :param scale_lr: learning rate in R, scalar which is multiplied with the predicted scale factor,
                        regularizes the amount of adjustment applied to the scale bias
        """
        super(CameraPredictor, self).__init__()

        self._num_feats = num_feats

        self.fc = nn.Sequential(
            nn.Linear(num_feats, num_feats),
            nn.LeakyReLU(0.1),
            nn.Linear(num_feats, num_feats),
            nn.LeakyReLU(0.1),
        )
        self.fc2 = nn.Linear(num_feats, 4)
        self.quat_layer = nn.Linear(num_feats, 4)

        self.quat_layer.bias = nn.Parameter(torch.FloatTensor(
            [1, 0, 0, 0]).type(self.quat_layer.bias.data.type()))

        self.scale_bias = scale_bias
        self.scale_lr = scale_lr

        net_init(self)

    def forward(self, img_feats: torch.FloatTensor, as_vec: bool = False) -> torch.FloatTensor:
        """
        Predicts the camera pose represented by a 3-tuple of scale factor, translation vector and quaternions
        representing the rotation to an input image features x.

        :param img_feats: [N x F] input tensor containing batch_size number of encoded features from the input images.
            N = batch size
            F = number of features
        :param as_vec: Returns the results as tensors instead of tuples.
        :return: 
            if as_vec :
                A 3-tuple containing the following tensors. (N = batch size)
                scale: N X 1 vector, containing the scale factors.
                translate: N X 2 tensor, containing the translation vectors for each sample
                quat: N X 4 tensor, containing the quaternions representing the rotation for each sample.
                    the quaternions are mapped to the rotation matrix outside of this forward pass.
                prob: [N x 1]: probability logit for a certain camera pose. only used in multi-hypotheses setting.
            else:
                [N x 8] tensor containing the above mentioned camera parameters in a tensor.
        """

        # convert NXCx1x1 tensor to a NXC vector
        img_feats = img_feats.view(-1, self._num_feats)
        preds = self.fc(img_feats)

        quat_preds = self.quat_layer(img_feats)

        # apply scale parameters
        scale_raw = self.scale_lr * preds[..., 0] + self.scale_bias
        scale = F.relu(scale_raw) + 1E-12  # minimum scale is 0.0

        # normalize quaternions
        norm_quats = F.normalize(quat_preds)

        z = torch.cat(
            (scale.unsqueeze(-1), preds[..., 1:3], norm_quats, preds[..., -1:]), dim=-1)

        return z if as_vec else vec_to_tuple(z)


class MultiCameraPredictor(nn.Module):
    """Module for predicting a set of camera poses and a corresponding probabilities."""

    def __init__(self, num_hypotheses=8, device="cuda", **kwargs):
        """
        :param num_hypotheses: number of camera poses which should be predicted.
        :param device: Device where the operations take place
        :param kwargs: arguments which are passed through to the single camera predictors
        """
        super(MultiCameraPredictor, self).__init__()

        self.num_hypotheses = num_hypotheses
        self.device = device
        self.cam_preds = nn.ModuleList(
            [CameraPredictor(**kwargs) for _ in range(num_hypotheses)])

        # taken from the original repo
        base_rotation = matrix_to_quaternion(
            euler_angles_to_matrix(torch.FloatTensor([np.pi / 2, 0, 0]), "XYZ")).unsqueeze(
            0)  # rotation by PI/2 around the x-axis
        base_bias = matrix_to_quaternion(
            euler_angles_to_matrix(torch.FloatTensor([0, np.pi / 4, 0]), "XYZ")).unsqueeze(
            0)  # rotation by PI/4 around the y-axis

        # base_rotation = torch.FloatTensor(
        #     [0.9239, 0, 0.3827, 0]).unsqueeze(0).unsqueeze(0)  # pi/4 (45° )
        # # base_rotation = torch.FloatTensor([ 0.7071,  0, 0.7071,  0]).unsqueeze(0).unsqueeze(0) ## pi/2
        # base_bias = torch.FloatTensor(
        #     [0.7071, 0.7071, 0, 0]).unsqueeze(0).unsqueeze(0)  # 90° by x-axis

        # taken from the original repo
        self.cam_biases = [base_bias]
        for i in range(1, self.num_hypotheses):
            self.cam_biases.append(quaternion_raw_multiply(
                base_rotation, self.cam_biases[i - 1]))

        self.cam_biases = torch.stack(
            self.cam_biases).squeeze().to(self.device)

    def forward(self, img_feats: torch.FloatTensor) -> Tuple:
        """
        Predict a certain number of camera poses. Samples one of the poses according to a predicted probability.
        :param img_feats: [N x F] input tensor containing batch_size number of encoded features from the input images.
            N = batch size
            F = number of features
        :return: 3-tuple containing
            - a sampled camera pose from the predicted camera poses, see CameraPredictor.forward for more documentation
            - the index of the camera which has been sampled
            - all predicted camera pose hypotheses, shape [N x H x 8 ]
             N = batch size, H = number of hypotheses, 8 = number of outputs from the pose predictor
        """

        # make camera pose predictions
        pred_pose = [cpp(img_feats, as_vec=True) for cpp in self.cam_preds]
        pred_pose = torch.stack(pred_pose, dim=1)

        # apply softmax to probabilities
        prob_logits = pred_pose[..., 7]
        # prob_logits = self.classification_head(img_feats)
        probs = F.softmax(prob_logits, dim=1)

        quats = pred_pose[..., 3:7]

        bias_quats = self.cam_biases.unsqueeze(0).repeat(
            quats.size(0), 1, 1)
        new_quats = quaternion_raw_multiply(quats, bias_quats)
        pred_pose_new = torch.cat(
            (pred_pose[..., :3], new_quats, probs.unsqueeze(-1)), dim=-1)

        # sample a camera according to multinomial distribution

        # taken from the original repo
        dist = torch.distributions.multinomial.Multinomial(probs=probs)
        # get index of hot-one in one-hot encoded vector. Delivers the index of the camera which should be sampled.
        sample_idx = dist.sample().argmax(dim=1)
        indices = sample_idx.unsqueeze(-1).repeat(1, 8).unsqueeze(1)
        sampled_cam = torch.gather(
            pred_pose, dim=1, index=indices).view(-1, pred_pose.size(-1))
        sampled_cam = vec_to_tuple(sampled_cam)

        return sampled_cam, sample_idx, vec_to_tuple(pred_pose_new)


def vec_to_tuple(x):
    """
    Converts an input tensor
    :param x: prediction output of the pose predictor.
     [Nx8]/[N x H x 8 ];  N = batch size, H = number of hypotheses, 8 = number of outputs from the pose predictor.
    :return: 4 tuples containing the
     - scale tensor [N x 1], predicted scale factors
     - translate tensor [N x 2], predicted translation vector
     - quat tensor [N X 4], predicted quaternions representing the rotation
     - prob tensor [N x 1], predicted probability logit for each camera pose
    """
    scale = x[..., 0]
    translate = x[..., 1:3]
    quat = x[..., 3:7]
    prob = x[..., 7]

    return scale, translate, quat, prob
