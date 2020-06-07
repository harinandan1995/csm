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

    def __init__(self, device, num_feats=512, feature_extractor=None):
        """

        :param device: The device on which the computation is performed. Usually CUDA.
        :param feature_extractor: An feature extractor of an image. If None, resnet18 will bes used.
        :param num_feats: The number of extracted features from the encoder.
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

        self.fc = nn.Linear(num_feats, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the camera pose represented by a 3-tuple of scale factor, translation vector and quaternions
        representing the rotation to an input image :param x.
        :param x: The input image, for which the camera pose should be predicted.
        :return: A 3-tuple containing the following tensors. (N = batch size)
                scale: N X 1 vector, containing the scale factors.
                translate: N X 3 matrix, containing the translation vectors for each sample
                quat: N X 4 matrix, containing the quaternions representing the rotation for each sample.
                    the quaternions are mapped to the rotation matrix outside of this forward pass.
        """
        x = self.encoder(x)
        if self.avg_pool:
            x = self.avg_pool(x)

        x = x.squeeze()  # convert NXCx1x1 tensor to a NXC vector
        x = self.fc(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        # predicted cam_params
        scale = x[..., 0]
        translate = x[..., 1:3]
        quat = x[..., 3:]

        return scale, translate, quat


class MultiCameraPredictor(nn.Module):
    """Module for predicting a set of camera poses and a corresponding probabilities."""

    # TODO: implement

    def __init__(self):
        pass

    def forward(self):
        pass
