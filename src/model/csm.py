import torch
from pytorch3d.renderer.cameras import OpenGLOrthographicCameras

from src.model.unet import UNet
from src.model.uv_to_3d import UVto3D
from src.nnutils.geometry import get_scaled_orthographic_projection, convert_3d_to_uv_coordinates
from src.nnutils.rendering import MaskRenderer, DepthRenderer


class CSM(torch.nn.Module):

    def __init__(self, template_mesh, mean_shape, device='cuda:0'):
        super(CSM, self).__init__()

        self.device = device

        self.unet = UNet(4, 3).to(self.device)
        self.uv_to_3d = UVto3D(mean_shape).to(self.device)
        self.mask_render = MaskRenderer(device=self.device)
        self.depth_render = DepthRenderer(device=self.device)
        self.template_mesh = template_mesh

    def forward(self, img, mask, scale, trans, quat):

        rotation, translation = get_scaled_orthographic_projection(
            scale, trans, quat, self.device)

        cameras = OpenGLOrthographicCameras(
            R=rotation, T=translation, device=self.device)

        sphere_points = self.unet(torch.cat((img, mask), 1))
        sphere_points = torch.tanh(sphere_points)
        sphere_points = torch.nn.functional.normalize(sphere_points, dim=1)

        pred_pos, pred_z, uv = self._get_projected_positions_image(sphere_points, cameras)
        pred_mask, depth = self.depth_render(self.template_mesh.extend(img.size(0)), rotation, translation)

        out = {
            "pred_positions": pred_pos,
            "pred_depths": depth.unsqueeze(1),
            "pred_z": pred_z,
            "pred_masks": pred_mask.unsqueeze(1),
            "uv": uv
        }

        return out

    def _get_projected_positions_image(self, sphere_points, cameras):

        uv = convert_3d_to_uv_coordinates(sphere_points.permute(0, 2, 3, 1))
        batch_size = uv.size(0)
        height = uv.size(1)
        width = uv.size(2)

        uv_flatten = uv.view(-1, 2)
        uv_3d = self.uv_to_3d(uv_flatten).view(batch_size, -1, 3)

        xy = cameras.transform_points(uv_3d)[:, :, :2]
        xyz_cam = cameras.get_world_to_view_transform().transform_points(uv_3d)
        z = xyz_cam[:, :, 2]

        xy = xy.view(batch_size, 1, height, width, 2)
        z = z.view(batch_size, 1, height, width, 1)
        uv = uv.view(batch_size, 1, height, width, 2)

        return xy.permute(0, 1, 4, 2, 3), z.permute(0, 1, 4, 2, 3), uv.permute(0, 1, 4, 2, 3)
