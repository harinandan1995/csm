import torch

from src.model.unet import UNet
from src.model.uv_to_3d import UVto3D
from src.nnutils.geometry import get_scaled_orthographic_projection, convert_3d_to_uv_coordinates
from src.nnutils.rendering import MaskRenderer, DepthRenderer


class CSM(torch.nn.Module):
    """
    Model for the task of Canonical Surface Mapping.
    This model is a combination of 4 sub models
    - UNet - to predict the sphere coordinates which will be used to generate uv values
    from a given input image and foreground mask
    - UVto3D - to convert UV values to its corresponding 3D points on the mesh template
    - renderer - to render depths and masks for the camera pose
    - CamPredictor - to predict camera poses. Used only if use_gt_cam is True

    B - batch size
    CP - number of camera poses used/predicted
    H - height of the image
    W - with of the image
    """

    def __init__(self, template_mesh, mean_shape, use_gt_cam=False, device='cuda'):
        """
        :param template_mesh: A pytorch3d.structures.Meshes object which will used for
        rendering depth and mask for a given camera pose
        :param mean_shape: A dictionary containing the following attributes
            - uv_map: A R X R tensor of defining the UV steps. Where R is the resolution of the UV map.
            - verts: A (None, 3) tensor of vertex coordinates of the mean shape
            - faces: A (None, 3) tensor of faces of the mean shape
            - face_inds: A R X R tensor where each value is the index of the face for
        the corresponding UV value in uv_map.
        :param use_gt_cam: True or False. True if you want to use the ground truth camera pose. False if you want to
            use the camer predictor to predict the camera poses
        :param device: Device to store the tensor. Default: cuda
        """
        super(CSM, self).__init__()

        self.device = device

        self.unet = UNet(4, 3).to(self.device)
        self.uv_to_3d = UVto3D(mean_shape).to(self.device)
        self.mask_render = MaskRenderer(device=self.device)
        self.depth_render = DepthRenderer(device=self.device)
        self.template_mesh = template_mesh

    def forward(self, img, mask, scale, trans, quat):
        """
        For the given img and mask
        - uses the unet to predict sphere coordinates
        - converts sphere coordinates to UV values
        - uses UVto3D to get the corresponding 3D points for the UV values
        - use CamPredictor to predict camera poses (if use_gt_cam is False)
        - perform scaled orthographic projection onto the 3D points to
            project them back to image plane
        - use the renderer to render depth and mask for the predicted/gt camera poses

        :param img: A (B X 3 X H X W) tensor of input image
        :param mask: A (B X 1 X H X W) tensor of input image
        :param scale: A (B X 1) tensor of input image
        :param trans: A (B X 2) tensor of translations (tx, ty)
        :param quat: A (B X 3) tensor of quaternions
        :return: A dictionary containing following values
            - pred_positions: A (B X CP X 2 X H X W) tensor containing predicted position of each pixel
                after transforming them to 3D and then projecting back to image plane
            - uv: A (B X 2 X H X W) tensor containing predicted UV values
            - uv_3d: A (B X H*W X 3) tensor containing the 3D coordinates for the corresponding UV values
            - pred_z: A (B X CP X 1 X H X W) tensor containing the z values for the UV values after
                projecting them to the image plane
            - pred_masks: A (B X CP X 1 X H X W) tensor containing the rendered masks of the mesh
                template for the camera poses
            - pred_depths: A (B X CP X 1 X H X W) tensor containing the rendered depths of the mesh
                template for the camera poses
        """

        rotation, translation = get_scaled_orthographic_projection(
            scale, trans, quat, self.device)

        sphere_points = self.unet(torch.cat((img, mask), 1))
        sphere_points = torch.tanh(sphere_points)
        sphere_points = torch.nn.functional.normalize(sphere_points, dim=1)

        pred_pos, pred_z, uv, uv_3d = self._get_projected_positions_of_sphere_points(
            sphere_points, rotation, translation)
        pred_mask, depth = self.depth_render(self.template_mesh.extend(img.size(0)), rotation, translation)

        out = {
            "pred_positions": pred_pos,
            "pred_depths": torch.flip(depth.unsqueeze(1), (-1, -2)),
            "pred_masks": torch.flip(pred_mask.unsqueeze(1), (-1, -2)),
            "pred_z": pred_z,
            "uv_3d": uv_3d,
            "uv": uv
        }

        return out

    def _get_projected_positions_of_sphere_points(self, sphere_points, rotation, translation):
        """
        For the given points on unit sphere calculates the 3D coordinates on the mesh template
        and projects them back to image plane

        :param sphere_points: A (B X 3 X H X W) tensor containing the predicted points on the sphere
        :param rotation: A (B X 3 X 3) camera rotation tensor
        :param translation: A (B X 3) camera translation tensor
        :return: A tuple(xy, z, uv, uv_3d)
            - xy - (B X 1 X 2 X H X W) x,y values of the 3D points after projecting onto image plane
            - z - (B X 1 X 1 X H X W) z value of the projection
            - uv - (B X H X W) UV values of the sphere coordinates
            - uv_3d - (B X H*W X 3) tensor with the 3D coordinates on the mesh
                template for the given sphere coordinates
        """

        uv = convert_3d_to_uv_coordinates(sphere_points.permute(0, 2, 3, 1))
        batch_size = uv.size(0)
        height = uv.size(1)
        width = uv.size(2)

        uv_flatten = uv.view(-1, 2)
        uv_3d = self.uv_to_3d(uv_flatten).view(batch_size, -1, 3)

        xyz = torch.bmm(uv_3d, rotation) + translation[:, None, :]

        xy = xyz[:, :, :2]
        z = xyz[:, :, 2]

        xy = xy.view(batch_size, 1, height, width, 2).permute(0, 1, 4, 2, 3)
        z = z.view(batch_size, 1, height, width, 1).permute(0, 1, 4, 2, 3)
        uv = uv.view(batch_size, height, width, 2).permute(0, 3, 1, 2)

        return xy, z, uv, uv_3d
