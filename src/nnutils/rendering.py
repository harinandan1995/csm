from typing import Tuple

import numpy as np
import pytorch3d
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    look_at_view_transform, RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, OpenGLOrthographicCameras,
    PointLights
)

# TODO: choose camera representation option
"""
1. Option: distance, elevation, azimuth
2. option: camera_position = (0, 0, 0) in world coordinates, 
3. option: quaternions 
4. option: (euler angles etc. )
"""


class Renderer(nn.Module):
    """Pytorch Module combining the mask and the depth renderer."""

    def __init__(self, device, image_size=256):
        """
        Initialization of the Renderer Class. Instances of the mask and depth renderer are create on corresponding device.


        :param device: The device, on which the computation is done.
        :param image_size: Image size for the rasterization. Default is 256.
        """
        self.mask_renderer = MaskRenderer(device, image_size)
        self.depth_renderer = DepthRenderer(device, image_size)

    def forward(self, meshes: pytorch3d.structures.Meshes, distance: float, elevation: float, azimuth: float,
                camera_params: object = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combines the forward functions of mask and depth renderer. Argument are equal to both renderers parameters.

        :param meshes: A batch of meshes. pytorch3d.structures.Meshes. Dimension meshes \in R^N.
                        View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
                        for additional information.
        :param distance: distance of the camera from the object
        :param elevation: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        :param azimuth: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azimuth is the angle between the projected vector and a
            reference vector at (1, 0, 0) on the reference plane (the horizontal plane).
        :param camera_params: placeholder for potential capsuling of the camera params.
        :return: Tuple with [N X W X H] / [N X W X H X C] tensor.
                with N = batch size, W = width of image, H = height of image, C = Channels. usually W=H.
        """
        image, depth_maps = self.depth_renderer(meshes, distance, elevation, azimuth)
        masks = self.mask_renderer(meshes, distance, elevation, azimuth)
        return image, depth_maps, masks


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

        # TODO: check which one to choose. Maybe introduce flag as function parameter.
        cameras = OpenGLOrthographicCameras(device=device)
        # cameras = OpenGLPerspectiveCameras(device=device)

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

    def forward(self, meshes: pytorch3d.structures.Meshes, distance: float, elevation: float, azimuth: float,
                camera_params: object = None) -> torch.Tensor:
        """
        Computes the projection mask of the 3D meshes. A silhouette shader is used to compute the silhouette.
        A ceiling operation converts the silhouette to a 1-0 mask.
        The mask is dependent on the camera, which parameters are passed as parameters to this function.

        :param meshes: A batch of meshes. pytorch3d.structures.Meshes. Dimension meshes \in R^N.
                        View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
                        for additional information.
        :param distance: distance of the camera from the object
        :param elevation: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        :param azimuth: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azimuth is the angle between the projected vector and a
            reference vector at (1, 0, 0) on the reference plane (the horizontal plane).
        :param camera_params: placeholder for potential capsuling of the camera params.
        :return: [N X W X H] tensor. with N = batch size, W = width of image, H = height of image. usually W=H.
        """
        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)

        # eye = (0, 1, 1),
        # R, T = look_at_view_transform(eye=eye, device=self.device)

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

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0,  # TODO: Remove because it is default setting
            faces_per_pixel=1,  # TODO: Remove because it is default setting
        )
        self._rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        _lights = PointLights(device=device, location=((1.0, 1.0, 2.0),))

        self._shader = HardPhongShader(device=device, lights=_lights)

    def forward(self, meshes, distance, elevation, azimuth):
        """
        Compute the depth map of the 3D Mesh according to the camera, which is passed is parametrised by distance,
        elevation and azimuth.

        :param meshes: A batch of meshes. pytorch3d.structures.Meshes. Dimension meshes \in R^N.
                        View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
                        for additional information.
        :param distance: distance of the camera from the object
        :param elevation: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        :param azimuth: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azimuth is the angle between the projected vector and a
            reference vector at (1, 0, 0) on the reference plane (the horizontal plane).
        :return: 2-Tuple of [N X W X H x C] tensors. with N = batch size, W = width of image, H = height of image, C=channels.
                usually W=H. The depth map
        """
        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)

        fragments = self._rasterizer(meshes, R=R, T=T)

        # TODO: do i need to forward the image?
        image = self._shader(fragments, meshes)
        depth_map = fragments.zbuf

        return image, depth_map

#
# class CameraPoseEstimator(nn.Module):
#     def __init__(self, meshes, renderer, image_ref):
#         super().__init__()
#         self.meshes = meshes
#         self.device = meshes.device
#         self.renderer = renderer
#
#         # Get the silhouette of the reference RGB image by finding all the non zero values.
#         image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 0).astype(np.float32))
#
#         self.register_buffer('image_ref', image_ref)
#
#         # Create an optimizable parameter for the x, y, z position of the camera.
#         # TODO: check whether to use fixed values or choose random
#         # TODO: will be inputed
#         self.camera_position = nn.Parameter(
#             torch.from_numpy(np.array([3.0, 6.9, +2.5], dtype=np.float32)).to(meshes.device))
#
#     def forward(self):
#         # Render the image using the updated camera position. Based on the new position of the
#         # camera we calculate the rotation and translation matrices
#         R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
#
#         T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)
#
#         image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
#
#         # Calculate the silhouette loss (from tutorial)
#         # loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
#
#         return image
