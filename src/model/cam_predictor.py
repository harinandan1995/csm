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


class CameraPredictor(nn.Module):
    """
    Camera predictor for camera pose prediction. It predicts a camera pose in the form of quaternions, scale scalar
    and translation vector based on an image.
    """

    def __init__(self, num_feats=512, encoder=None):
        """

        :param device: The device on which the computation is performed. Usually CUDA.
        :param feature_extractor: An feature extractor of an image. If None, resnet18 will bes used.
        :param num_feats: The number of extracted features from the encoder.
        """
        super(CameraPredictor, self).__init__()
        if not encoder:
            _resnet = torch.hub.load(
                'pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            self.encoder = nn.Sequential(*([*_resnet.children()][:-1]))

            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder = encoder
        self._num_feats = num_feats
        self.fc = nn.Linear(num_feats, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x: torch.Tensor, as_vec: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the camera pose represented by a 3-tuple of scale factor, translation vector and quaternions
        representing the rotation to an input image :param x.
        :param x: The input image, for which the camera pose should be predicted.
        :param as_vec: Returns the results as tensors instead of tuples. 
        :return: A 3-tuple containing the following tensors. (N = batch size)
                scale: N X 1 vector, containing the scale factors.
                translate: N X 3 matrix, containing the translation vectors for each sample
                quat: N X 4 matrix, containing the quaternions representing the rotation for each sample.
                    the quaternions are mapped to the rotation matrix outside of this forward pass.
        """
        x = self.encoder(x)

        # convert NXCx1x1 tensor to a NXC vector
        x = x.view(-1, self._num_feats)
        x = self.fc(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x if as_vec else vec_to_tuple(x)



class MultiCameraPredictor(nn.Module):
    """Module for predicting a set of camera poses and a corresponding probabilities."""

    def __init__(self, num_hypotheses=8, num_feats=512):
        super(MultiCameraPredictor, self).__init__()
        _resnet = torch.hub.load(
            'pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        _encoder = nn.Sequential(*([*_resnet.children()][:-1]))

        for param in _encoder.parameters():
            param.requires_grad = False

        self.num_hypotheses = num_hypotheses
        self.cam_preds = nn.ModuleList(
            [CameraPredictor(num_feats=num_feats, encoder=_encoder) for _ in range(num_hypotheses)])

    def forward(self, x):
        # make n camera pose predictions
        cam_preds = [cpp(x, as_vec=True) for cpp in self.cam_preds]
        cam_preds = torch.stack(cam_preds, dim=1)

        prob_logits = cam_preds[..., -1]
        probs = F.softmax(prob_logits, dim=1)
        new_cam_preds = torch.cat(
            (cam_preds[..., :-1, ], probs.unsqueeze(-1)), dim=-1)

        dist = torch.distributions.multinomial.Multinomial(probs=probs)

        # get index of hot-one in one-hot encoded vector
        sample_idx = dist.sample().argmax(dim=1)
        indices = sample_idx.unsqueeze(-1).repeat(1, 7).unsqueeze(1)
        sampled_cam = torch.gather(
            input=cam_preds, dim=1, index=indices).view(-1, cam_preds.size(-1))
        sampled_cam = vec_to_tuple(sampled_cam)
        # sampled_cam = vec_to_tuple(cam_preds[sample_idx])

        return sampled_cam, sample_idx, vec_to_tuple(new_cam_preds)


def vec_to_tuple(x):
    scale = x[..., 0]
    translate = x[..., 1:2 + 1]
    quat = x[..., 2:5 + 1]
    prob = x[..., 6]

    return scale, translate, quat, prob
