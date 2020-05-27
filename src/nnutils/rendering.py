from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, OpenGLOrthographicCameras,
    PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import so3_exponential_map

"""
1. Option: distance, elevation, azimuth
2. option: camera_position = (0, 0, 0) in world coordinates, 
3. option: quaternions 
4. option: (euler angles etc.)
5. option:  https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.so3_exponential_map
"""


# TODO: retrieve mask and depth in one rendering step to speed up training.
# TODO: check rasterization settings

class CameraParams(object):
    """Class for representing the camera projection."""

    def __init__(self, device: str = "cpu", n: int = 1):
        """
        Initialization of camera params.
        s is a scalar for scaling the translation vector t.
        r is a vector in the lie algebra to the corresponding lie group. it represents the rotation.

        :param device: The device, on which the computation is done.
        :param n: number of cameras necessary. n = batch size
        """
        # TODO: check for random initialization
        self.r = torch.zeros(n, 3, device=device)
        self.s = torch.ones(n, 1, device=device)  # scale
        self.t = torch.zeros(n, 3, device=device)  # translation vector


#
# class CombinedRenderer(nn.Module):
#     """Pytorch Module combining the mask and the depth renderer."""
#
#     def __init__(self, device, image_size=256):
#         """
#         Initialization of the Renderer Class. Instances of the mask and depth renderer are create on corresponding device.
#
#         :param device: The device, on which the computation is done.
#         :param image_size: Image size for the rasterization. Default is 256.
#         """
#         self.mask_renderer = MaskRenderer(device, image_size)
#         self.depth_renderer = DepthRenderer(device, image_size)
#
#     def forward(self, meshes: Meshes, pi: CameraParams) -> Tuple[
#         torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Combines the forward functions of mask and depth renderer. Argument are equal to both renderers parameters.
#
#         :param pi:
#         :param meshes: A batch of meshes. pytorch3d.structures.Meshes. Dimension meshes \in R^N.
#                         View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
#                         for additional information.
#             reference vector at (1, 0, 0) on the reference plane (the horizontal plane).
#         :return: Tuple with [N X W X H] / [N X W X H X C] tensor.
#                 with N = batch size, W = width of image, H = height of image, C = Channels. usually W=H.
#         """
#         R = so3_exponential_map(pi.r)
#         T = pi.s * pi.t
#
#         fragments = self._rasterizer(meshes, R=R, T=T)
#         image = self.depth_shader(fragments, meshes)
#         depth_maps = fragments.zbuf
#
#         silhouettes = self.silhouette_shader(fragments, meshes)
#         masks = silhouettes[..., 3]  # extract masks from alpha channel
#         masks = torch.ceil(masks)  # converts silhouette to 1-0 masks
#
#         return image, depth_maps, masks
#

class MaskRenderer(nn.Module):
    """Pytorch Module for computing the projection mask of a 3D Mesh. It is used to compute the re-projection loss."""

    def __init__(self, device, image_size=256):
        """

        Initialization of MaskRenderer. Renderer is initialized with predefined rasterizer and shader.
        A soft silhouette shader is used to compute the projection mask.

        :param device: The device, on which the computation is done.
        :param image_size: Image size for the rasterization. Default is 256.
        """
        super(MaskRenderer, self).__init__()

        self.device = device

        cameras = OpenGLOrthographicCameras(device=device)

        # parameter settings as of Pytorch3D Tutorial
        # (https://pytorch3d.org/tutorials/camera_position_optimization_with_differentiable_rendering)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=10,
        )
        self._rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        self._shader = SoftSilhouetteShader(blend_params=blend_params)
        self.renderer = MeshRenderer(
            rasterizer=self._rasterizer,
            shader=self._shader
        )

    def forward(self, meshes: Meshes, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Computes the projection mask of the 3D meshes. A silhouette shader is used to compute the silhouette.
        A ceiling operation converts the silhouette to a 1-0 mask.
        The mask is dependent on the camera, which is passed is as rotation matrix R and
        translation vector T.

        :param T: Translation vector of the camera
        :param R: Rotation matrix of the camera
        :param meshes: A batch of meshes. pytorch3d.structures.Meshes. Dimension meshes \in R^N.
                        View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
                        for additional information.
        :return: [N X W X H] tensor. with N = batch size, W = width of image, H = height of image. usually W=H.
        """
        silhouette = self.renderer(meshes, R=R, T=T)  # dimensions N x W x H x C

        masks = silhouette[..., 3]  # extract masks from alpha channel
        masks = torch.ceil(masks)  # converts silhouette to 1-0 masks

        return masks  # [N x W x H]


class DepthRenderer(nn.Module):
    """Pytorch Module for computing the depth map of a 3D Mesh. Is used to computer the visibility loss."""

    def __init__(self, device, image_size=256):
        """
           Initialization of DepthRenderer. Initialization of rasterizer and shader.
           In addition a point light source for the PhongShader is used.
           :param device: The device, on which the computation is done, e.g. cpu or cuda.
           :param image_size: Image size for the rasterization. Default is 256.
       """
        super(DepthRenderer, self).__init__()
        self.device = device
        # TODO: check which one to choose. Maybe introduce flag as function parameter.
        cameras = OpenGLOrthographicCameras(device=self.device)
        # cameras = OpenGLPerspectiveCameras(device=self.device)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        raster_settings = RasterizationSettings(
            image_size=image_size,
            faces_per_pixel=100,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            bin_size=None,
            max_faces_per_bin=None
        )
        self._rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        _lights = PointLights(device=device, location=((1.0, 1.0, 2.0),))

        self._shader = SoftSilhouetteShader(blend_params)

    def forward(self, meshes: Meshes, R: torch.Tensor, T: torch.Tensor):
        """
        Compute the depth map of the 3D Mesh according to the camera, which is passed is as rotation matrix R and
        translation vector T.

        :param T: Translation vector of the camera
        :param R: Rotation matrix of the camera
        :param meshes: A batch of meshes. pytorch3d.structures.Meshes. Dimension meshes \in R^N.
                        View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
                        for additional information.
        :return: 2-Tuple of [N X W X H x C] (image) and [N X W X H](depth map ) tensors.
                with N = batch size,
                W = width of image,
                H = height of image, usually W=H.
                C = channels.
        """
        fragments = self._rasterizer(meshes, R=R, T=T)

        # TODO: do i need to forward the image?
        image = self._shader(fragments, meshes)
        depth_map = fragments.zbuf

        return image[..., 3], depth_map[..., 0]
