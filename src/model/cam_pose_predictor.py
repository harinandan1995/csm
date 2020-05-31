# model input: bottleneck layer of resnet 18+ CONV(FILTERS=64, kernel=4x4, S=2,) + 2 FC Layers
# model architecture:
# camera predictor:
#  - 2 FC Layer with leaky ReLU activation (see below) (different from the one in the encoder)
# - 100 input features (per default)
# - nn.Sequential(
#            nn.Linear(nc_inp, nc_out),
#            nn.LeakyReLU(0.1,inplace=True)
#        )
# - separate predictors for probability, scale, translation and quaternions (zero initialized)
# - biases: init
#   -         base_rotation = torch.FloatTensor([0.9239, 0, 0.3827 , 0]).unsqueeze(0).unsqueeze(0) ##pi/4
#   -        base_bias = torch.FloatTensor([ 0.7071,  0.7071,   0,   0]).unsqueeze(0).unsqueeze(0)

# model output: set of cameras with associated probabilities (also single camera output possible)
# - set of cameras
# loss function: re-projection loss (input mask - projected_mask squared); mask is a label


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: check whether encoded features are passed to the model


class CameraPredictor(nn.Module):
    """
    Camera predictor for camera pose prediction. It predicts a camera pose in the form of quaternions, scale scalar
    and translation vector based on an image.
    """

    def __init__(self, input_shape, feature_extractor=None):
        """

        :param feature_extractor: An feature extractor of an image. If None, resnet18 will bes used.
        :param input_shape: Shape of an input image. It is used to calculate the number of neurons
                            in the FC layer.
        """
        super(CameraPredictor, self).__init__()
        if not feature_extractor:
            _resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            self.encoder = nn.Sequential(*([*_resnet.children()][:-1]))

            for param in self.encoder.parameters():
                param.requires_grad = False
            self.avg_pool = None
        else:
            self.encoder = feature_extractor

        num_feats = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)

        self.fc = nn.Linear(num_feats, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the camera pose represented by a 3-tuple of scale factor, translation vector and quaternions
        representing the rotation to an input image :param x.
        :param x: The input image, for which the camera pose should be predicted.
        :return: A 3-tuple containing the following tensors. (N = batch size)
                scale: N X 1 vector, containing the scale factors.
                translate: N X 3 matrix, containing the translation vectors for each sample
                rotate: N X 4 matrix, containing the quaternions representing the rotation for each sample.
                    the quaternions are mapped to the rotation matrix outside of this forward pass.
        """
        x = self.encoder(x)
        if self.avg_pool:
            x = self.avg_pool(x)

        x = x.squeeze(-1).squeeze(-1)  # convert NXCx1x1 tensor to a NXC vector
        x = self.fc(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        # predicted cam_params
        scale = x[..., 0]
        translate = x[..., 1:3 + 1]
        rotate = x[..., 4:6 + 1]
        prob = x[..., 7]
        prob = F.softmax(prob)

        return scale, translate, rotate, prob


class MultiCameraPredictor(nn.Module):
    """Module for predicting a set of camera poses and a corresponding probabilities."""

    # TODO: implement

    def __init__(self, num_hypotheses, input_shape):
        self.cams = nn.ModuleList([CameraPredictor(input_shape) for _ in range(num_hypotheses)])

    def forward(self):
        ...
