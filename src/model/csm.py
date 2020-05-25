import torch
from pytorch3d.renderer.cameras import OpenGLOrthographicCameras

from src.model.unet import UNet
from src.model.uv_to_3d import UVto3D
from src.nnutils.geometry import get_scaled_orthographic_projection
from src.nnutils.rendering import MaskRenderer, DepthRenderer


class CSM(torch.nn.Module):

    def __init__(self, template_mesh, mean_shape, device='cuda:0'):
        super(CSM, self).__init__()

        self.device = device

        self.unet = UNet(4, 2).to(self.device)
        self.uv_to_3d = UVto3D(mean_shape).to(self.device)
        self.mask_render = MaskRenderer(device=self.device)
        self.depth_render = DepthRenderer(device=self.device)
        self.template_mesh = template_mesh

    def forward(self, img, mask, scale, trans, quat):

        rotation, translation = get_scaled_orthographic_projection(
            scale, trans, quat, self.device)
        rotation = rotation.unsqueeze(0)
        translation = translation.unsqueeze(0)

        uv = self.unet(torch.cat((img, mask.unsqueeze(1)), 1))
        print(uv.shape)
        uv_3d = self._convert_uv_to_world(uv)
        cameras = OpenGLOrthographicCameras(
            R=rotation, T=translation, device=self.device)

        pred_pos, pred_z = self._project_world_to_image(uv_3d, cameras)
        depth = self.depth_render(self.template_mesh, rotation, translation)
        pred_mask = self.mask_render(self.template_mesh, rotation, translation)

        out = {
            "pred_positions": pred_pos,
            "pred_depths": depth,
            "pred_z": pred_z,
            "pred_masks": pred_mask
        }

        return out

    def _convert_uv_to_world(self, uv):
        batch_size = uv.size(0)
        height = uv.size(1)
        width = uv.size(2)

        uv_flatten = uv.view(-1, 2)
        uv_3d = self.uv_to_3d(uv_flatten)
        uv_3d = uv_3d.view(batch_size, height, width, 3)

        return uv_3d

    def _project_world_to_image(self, points, cameras):
        xy = cameras.transform_points(points)[:, :, :2]
        xyz_cam = cameras.get_world_to_view_transform().transform_points(points)
        z = xyz_cam[:, :, 2:]

        return xy, z
