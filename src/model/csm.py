import torch

from src.model.uv_to_3d import UVto3D
from src.model.unet import UNet


class CSM(torch.nn.Module):

    def __init__(self, template, mean_shape, use_gt_cam):

        super(CSM, self).__init__()

        self.use_gt_cam = use_gt_cam

        self.unet = UNet(3, 2)
        self.uv_to_3d = UVto3D(mean_shape)
        self.renderer = None

    def forward(self, input):

        # TODO: 
        # Call unet
        # Call cam pose net if use_gt_cam is False
        # Convert UV to 3D points using UVto3D
        # https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html 
        # transform to 2D to get camera frame projection
        # call renderer if to get depth and mask
        

        return


