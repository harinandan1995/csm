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

        :param encoder: An feature extractor of an image. If None, resnet18 will bes used.
        :param num_feats: The number of extracted features from the encoder.
        """
        super(CameraPredictor, self).__init__()
        # TODO: add 1x1 conv to work with arbitrary num of in_channels
        # allows us to use only one encoder for all camera hypotheses in the multi-cam predictor.
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

    def forward(self, x: torch.Tensor, as_vec: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        x = self.encoder(x)
        # convert NXCx1x1 tensor to a NXC vector
        x = x.view(-1, self._num_feats)
        x = self.fc(x)

        return x if as_vec else vec_to_tuple(x)

# TODO: have one network to predict all hypotheses


class MultiCameraPredictor(nn.Module):

    """Module for predicting a set of camera poses and a corresponding probabilities."""

    def __init__(self, num_hypotheses=8, num_feats=512):
        """

        :param num_hypotheses: number of camera poses which should be predicted.
        :param num_feats: number of features extracted from the image by the encoder
        """
        super(MultiCameraPredictor, self).__init__()
        _encoder = get_encoder(trainable=False)

        self.num_hypotheses = num_hypotheses
        self.cam_preds = nn.ModuleList(
            [CameraPredictor(num_feats=num_feats, encoder=_encoder) for _ in range(num_hypotheses)])

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
        cam_preds = [cpp(x, as_vec=True) for cpp in self.cam_preds]
        cam_preds = torch.stack(cam_preds, dim=1)

        # apply softmax to probabilities
        prob_logits = cam_preds[..., 7]
        probs = F.softmax(prob_logits, dim=1)

        #  only positive scales
        scale_logits = cam_preds[..., 0]
        scale = F.relu(scale_logits)

        new_cam_preds = torch.cat((scale.unsqueeze(-1),
                                   cam_preds[..., 1:-1, ], probs.unsqueeze(-1)), dim=-1)

        dist = torch.distributions.multinomial.Multinomial(probs=probs)

        # get index of hot-one in one-hot encoded vector. Delivers the index of the camera which should be sampled.
        sample_idx = dist.sample().argmax(dim=1)
        indices = sample_idx.unsqueeze(-1).repeat(1, 8).unsqueeze(1)
        sampled_cam = torch.gather(
            input=cam_preds, dim=1, index=indices).view(-1, cam_preds.size(-1))
        sampled_cam = vec_to_tuple(sampled_cam)

        return sampled_cam, sample_idx, vec_to_tuple(new_cam_preds)


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
            param.requires_grad = False
    return encoder
