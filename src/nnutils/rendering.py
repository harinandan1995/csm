from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer,
                                OpenGLOrthographicCameras, PointLights,
                                RasterizationSettings, SoftSilhouetteShader,
                                TexturedSoftPhongShader)
from pytorch3d.structures import Meshes


class MaskAndDepthRenderer(nn.Module):
    """Pytorch Module combining the mask and the depth renderer."""

    def __init__(self, device="cuda", image_size=256):
        """
        Initialization of the Renderer Class. Instances of the mask and depth renderer are create on corresponding
        device.

        :param device: The device, on which the computation is done.
        :param image_size: Image size for the rasterization. Default is 256.
        """
        super().__init__()

        cameras = OpenGLOrthographicCameras(device=device)
        self._rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=image_size,
                faces_per_pixel=100)
        )

        self._shader = SoftSilhouetteShader(
            blend_params=(BlendParams(sigma=1e-4, gamma=1e-4)))

    def forward(self, R: torch.Tensor, T: torch.Tensor, meshes: Meshes) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combines the forward functions of mask and depth renderer. Argument are equal to both renderers parameters.

        :param T: Translation vector of the camera
        :param R: Rotation matrix of the camera
        :param meshes: Contains a batch of meshes which should be rendered.

        :return: Tuple with [N X W X H] / [N X W X H X C] tensor.
                with N = batch size, W = width of image, H = height of image, C = Channels. usually W=H.
        """

        # retrieve depth map
        fragments = self._rasterizer(meshes, R=R, T=T)
        silhouettes = self._shader(fragments, meshes)
        depth_maps = fragments.zbuf

        # extract masks from alpha channel of rgba image
        masks = silhouettes[..., 3]

        return masks, depth_maps[..., 0]


class MaskRenderer(nn.Module):
    """Pytorch Module for computing the projection mask of a 3D Mesh. It is used to compute the re-projection loss."""

    def __init__(self, image_size=256, device='cuda'):
        """
        Initialization of MaskRenderer. Renderer is initialized with predefined rasterizer and shader.
        A soft silhouette shader is used to compute the projection mask.

        :param meshes: A batch of meshes. pytorch3d.structures.Meshes. Dimension meshes \in R^N.
                        View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
                        for additional information.
                        In our case it is usually only one batch which is the template for a certain category.
        :param device: The device, on which the computation is done.
        :param image_size: Image size for the rasterization. Default is 256.
        """
        super(MaskRenderer, self).__init__()

        self.device = device
        sigma = 1e-4

        cameras = OpenGLOrthographicCameras(device=device)

        # parameter settings as of Pytorch3D Tutorial
        # (https://pytorch3d.org/tutorials/camera_position_optimization_with_differentiable_rendering)

        self._rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=image_size,
                faces_per_pixel=100,
                blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
                bin_size=None,  # this setting controls whether naive or coarse-to-fine rasterization is used
                max_faces_per_bin=None  # this setting is for coarse rasterization
            )
        )

        self._shader = SoftSilhouetteShader(
            blend_params=(BlendParams(sigma=1e-4, gamma=1e-4)))

    def forward(self, R: torch.Tensor, T: torch.Tensor, meshes: Meshes) -> torch.Tensor:
        """
        Computes the projection mask of the 3D meshes. A silhouette shader is used to compute the silhouette.
        A ceiling operation converts the silhouette to a 1-0 mask.
        The mask is dependent on the camera, which is passed is as rotation matrix R and
        translation vector T.

        :param T: Translation vector of the camera
        :param R: Rotation matrix of the camera
        :param meshes: A batch of meshes that will be rendered. pytorch3d.structures.Meshes. Dimension meshes in R^N.
                View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
                for additional information.
        :return: [N X W X H] tensor. with N = batch size, W = width of image, H = height of image. usually W=H.
        """

        # dimensions N x W x H x C
        fragments = self._rasterizer(meshes, R=R, T=T)
        silhouette = self._shader(fragments, meshes)

        # extract masks from alpha channel, [N x W x H]
        return silhouette[..., 3]


class DepthRenderer(nn.Module):
    """Pytorch Module for computing the depth map of a 3D Mesh. Is used to computer the visibility loss."""

    def __init__(self, device: str, image_size: int = 256):
        """
           Initialization of DepthRenderer. Initialization of the default mesh rasterizer and silhouette shader which is
           used because of simplicity.

           :param device: The device, on which the computation is done, e.g. cpu or cuda.
           :param image_size: Image size for the rasterization. Default is 256.

       """
        super(DepthRenderer, self).__init__()

        cameras = OpenGLOrthographicCameras(device=device)

        self._rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=image_size,
                faces_per_pixel=100
            )
        )

        self._shader = SoftSilhouetteShader(
            blend_params=(BlendParams(sigma=1e-4, gamma=1e-4)))

    def forward(self, R: torch.Tensor, T: torch.Tensor, meshes: Meshes):
        """
        Compute the depth map of the 3D Mesh according to the camera, which is passed is as rotation matrix R and
        translation vector T.

        :param T: Translation vector of the camera
        :param R: Rotation matrix of the camera
        :param meshes: A batch of meshes that will be rendered. pytorch3d.structures.Meshes. Dimension meshes in R^N.
                View https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/structures/meshes.py
                for additional information.

        :return: 2-Tuple of [N X W X H x C] (image) and [N X W X H](depth map ) tensors.
                with N = batch size,
                W = width of image,
                H = height of image, usually W=H.
                C = channels.
        """

        fragments = self._rasterizer(meshes, R=R, T=T)

        # needs to be rendered through the shader to obtain proper depth values
        _ = self._shader(fragments, meshes)
        depth_map = fragments.zbuf

        return depth_map[..., 0]


class ColorRenderer(nn.Module):

    def __init__(self, meshes, image_size=256, device='cuda'):
        super(ColorRenderer, self).__init__()

        self.meshes = meshes
        cameras = OpenGLOrthographicCameras(device=device)

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=TexturedSoftPhongShader(device=device, lights=lights)
        )

    def forward(self, rotation, translation):
        color_image = self.renderer(self.meshes.extend(
            rotation.size(0)), R=rotation, T=translation)

        return color_image
