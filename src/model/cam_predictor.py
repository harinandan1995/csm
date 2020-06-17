from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (euler_angles_to_matrix,
                                  matrix_to_euler_angles, matrix_to_quaternion,
                                  quaternion_multiply, quaternion_to_matrix)


class CameraPredictor(nn.Module):
    """
    Camera predictor for camera pose prediction. It predicts a camera pose in the form of quaternions, scale scalar
    and translation vector based on an image.
    """

    def __init__(self, num_in_chans=3, num_feats=512, encoder=None, scale_bias=1.0, scale_lr=0.05):
        """
        :param num_in_chans: Number of input channels. E.g. 3 for an RGB image, 4 for image + mask etc.
        :param encoder: An feature extractor of an image. If None, resnet18 will bes used.
        :param num_feats: The number of extracted features from the encoder.
        :param scale_bias: bias term in R, which is added to the scale prediction.
        :param scale_lr: learning rate in R, scalar which is multiplied with the predicted scale factor,
                        regularizes the amount of adjustment applied to the scale bias
        """
        super(CameraPredictor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_in_chans,
                               out_channels=3,
                               kernel_size=1)

        # allows us to use only one shared encoder for all camera hypotheses in the multi-cam predictor.
        if not encoder:
            self.encoder = get_encoder(trainable=False)
        else:
            self.encoder = encoder

        self._num_feats = num_feats

        self.fc = nn.Sequential(
            nn.Linear(num_feats, num_feats),
            nn.LeakyReLU(),
            nn.Linear(num_feats, num_feats),
            nn.LeakyReLU(),
            nn.Linear(num_feats, 8)
        )

        self.scale_bias = scale_bias
        self.scale_lr = scale_lr

    def forward(self, x: torch.Tensor, as_vec: bool = False) -> Union[torch.Tensor, Tuple[Any, Any, Any, Any]]:
        """
        Predicts the camera pose represented by a 3-tuple of scale factor, translation vector and quaternions
        representing the rotation to an input image :param x.
        :param x: The input image, for which the camera pose should be predicted.
        :param as_vec: Returns the results as tensors instead of tuples.
        :return: if as_vec :
            A 3-tuple containing the following tensors. (N = batch size)
            scale: N X 1 vector, containing the scale factors.
            translate: N X 2 tensor, containing the translation vectors for each sample
            quat: N X 4 tensor, containing the quaternions representing the rotation for each sample.
                the quaternions are mapped to the rotation matrix outside of this forward pass.
            prob: [N x 1]: probability logit for a certain camera pose. only used in multi-hypotheses setting.
            else:
                [N x 8] tensor containing the above mentioned camera parameters in a tensor.
        """
        x = self.conv1(x)
        x = self.encoder(x)
        # convert NXCx1x1 tensor to a NXC vector
        x = x.view(-1, self._num_feats)
        x = self.fc(x)

        # apply scale parameters
        scale_raw = self.scale_lr * x[..., 0] + self.scale_bias
        scale = F.relu(scale_raw) + 1E-12  # minimum scale is 0.0

        # normalize quaternions
        norm_quats = F.normalize(x[..., 3:7])

        z = torch.cat(
            (scale.unsqueeze(-1), x[..., 1:3], norm_quats, x[..., 7:]), dim=-1)

        return z if as_vec else vec_to_tuple(z)


class MultiCameraPredictor(nn.Module):

    """Module for predicting a set of camera poses and a corresponding probabilities."""

    def __init__(self, num_hypotheses=8, device="cuda", **kwargs):
        """

        :param num_hypotheses: number of camera poses which should be predicted.
        :param kwargs: arguments which are passed through to the single camera predictors
        """
        super(MultiCameraPredictor, self).__init__()
        _encoder = get_encoder(trainable=False)

        self.num_hypotheses = num_hypotheses

        self.cam_preds = nn.ModuleList(
            [CameraPredictor(encoder=_encoder, **kwargs) for _ in range(num_hypotheses)])

        # taken from the original repo
        base_rotation = matrix_to_quaternion(
            euler_angles_to_matrix(torch.FloatTensor([0.5, 0, 0])*np.pi, "XYZ")).unsqueeze(0)  # rotation by PI/2 around the x-axis
        base_bias = matrix_to_quaternion(
            euler_angles_to_matrix(torch.FloatTensor([0, 0.25, 0])*np.pi, "XYZ")).unsqueeze(0)  # rotation by PI/4 around the y-axis

        # base_rotation = torch.FloatTensor(
        #     [0.9239, 0, 0.3827, 0]).unsqueeze(0).unsqueeze(0)  # pi/4 (45° )
        # # base_rotation = torch.FloatTensor([ 0.7071,  0 , 0.7071,   0]).unsqueeze(0).unsqueeze(0) ## pi/2
        # base_bias = torch.FloatTensor(
        #     [0.7071, 0.7071, 0, 0]).unsqueeze(0).unsqueeze(0)  # 90° by x-axis

        # taken from the original repo
        self.cam_biases = [base_bias]
        for i in range(1, self.num_hypotheses):
            self.cam_biases.append(quaternion_multiply(
                base_rotation, self.cam_biases[i - 1]))

        self.cam_biases = torch.stack(self.cam_biases).squeeze().to(device)

    def forward(self, x):
        """
        Predict a certain number of camera poses. Samples one of the poses according to a predicted probability.
        :param x: [N x C X H X W] input tensor containing  batch_size number of rgb images.
            N = batch size
            C = input channels (e.g. 3 for a rgb image)
            H = W = size of the input image. e.g 255 x 255
        :return: 3-tuple containing
            - a sampled camera pose from the predicted camera poses, see CameraPredictor.forward for more documentation
            - the index of the camera which has been sampled
            - all predicted camera pose hypotheses, shape [N x H x 8 ]
             N = batch size, H = number of hypotheses, 8 = number of outputs from the pose predictor
        """
        # make camera pose predictions
        pred_pose = [cpp(x, as_vec=True) for cpp in self.cam_preds]
        pred_pose = torch.stack(pred_pose, dim=1)

        # apply softmax to probabilities
        prob_logits = pred_pose[..., 7]
        probs = F.softmax(prob_logits, dim=1)

        quats = pred_pose[..., 3:7]

        bias_quats = self.cam_biases.unsqueeze(0).repeat(quats.size(0), 1, 1)
        new_quats = quaternion_multiply(quats, bias_quats)
        pred_pose_new = torch.cat(
            (pred_pose[..., :3], new_quats, probs.unsqueeze(-1)), dim=-1)

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


def get_encoder(trainable=False):
    """
    Loads resnet18 and extracts the pre-trained convolutional layers for feature extraction.
    Pre-trained layers are frozen.
    :param trainable: bool. whether to train the resnet layers  
    :return: Feature extractor from resnet18
    """

    resnet = torch.hub.load(
        'pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    encoder = nn.Sequential(*([*resnet.children()][:-1]))
    if not trainable:
        for param in encoder.parameters():
            param.requires_grad = True
    return encoder
