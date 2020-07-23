import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import OpenGLOrthographicCameras

from src.model.cam_predictor import CameraPredictor, MultiCameraPredictor
from src.model.unet import UNet
from src.model.uv_to_3d import UVto3D
from src.nnutils.geometry import get_scaled_orthographic_projection, convert_3d_to_uv_coordinates
from src.nnutils.rendering import MaskRenderer, DepthRenderer, MaskAndDepthRenderer


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

    def __init__(self, template_mesh: Meshes, mean_shape: dict,
                 use_gt_cam: bool = False, num_cam_poses: int = 8, 
                 use_sampled_cam=False):
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
            use the camera predictor to predict the camera poses
        :param num_cam_poses: Number of camera hypothesis to be used. Should be used in with use_gt_cam=False
        :use_sampled_cam: True of False. True if you want the output from the camera pose sampled according 
            to the probabilities. False if you want output for all the predicted camera poses. 
            Should be used in with use_gt_cam=False
        :param device: Device to store the tensor. Default: cuda
        """
        super(CSM, self).__init__()

        self.unet = UNet(4, 3, num_downs=5)
        self.uv_to_3d = UVto3D(mean_shape)
        self.renderer = MaskAndDepthRenderer(meshes=template_mesh)

        self.use_gt_cam = use_gt_cam
        self.use_sampled_cam = use_sampled_cam

        if not self.use_gt_cam:
            self.multi_cam_pred = MultiCameraPredictor(num_hypotheses=num_cam_poses,device=template_mesh.device)

    def forward(self, img: torch.Tensor, mask: torch.Tensor,
                scale: torch.Tensor, trans: torch.Tensor, quat: torch.Tensor):
        """
        For the given img and mask
        - uses the unet to predict sphere coordinates
        - converts sphere coordinates to UV values
        - uses UVto3D to get the corresponding 3D points for the UV values
        - use CamPredictor to predict camera poses (if use_gt_cam is False)
        - perform scaled orthographic projection onto the 3D points to
            project them back to image plane
        - use the renderer to render pred_depth and mask for the predicted/gt camera poses

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

        sphere_points = self.unet(torch.cat((img, mask), 1))
        sphere_points = torch.tanh(sphere_points)
        sphere_points = torch.nn.functional.normalize(sphere_points, dim=1)

        rotation, translation, pred_poses = self._get_camera_extrinsics(img, scale, trans, quat)

        # Project the sphere points onto the template and project them back to image plane
        pred_pos, pred_z, uv, uv_3d = self._get_projected_positions_of_sphere_points(
            sphere_points, rotation, translation)

        # Render depth and mask of the template for the cam pose
        pred_mask, pred_depth = self._render(rotation, translation)

        out = {
            "pred_positions": pred_pos,
            "pred_depths": torch.flip(pred_depth, (-1, -2)),
            "pred_masks": torch.flip(pred_mask, (-1, -2)),
            "pred_z": pred_z,
            "rotation": rotation,
            "translation": translation,
            "uv_3d": uv_3d,
            "uv": uv
        }

        if not self.use_gt_cam:
            out['pred_poses'] = pred_poses

        return out

    def _get_camera_extrinsics(self, img, scale, trans, quat):

        batch_size = img.size(0)
        pred_poses = None
        
        if self.use_gt_cam:
            rotation, translation = get_scaled_orthographic_projection(
                scale, trans, quat, True)
        else:
            cam_pred, sample_idx, pred_poses = self.multi_cam_pred(img)
            
            if self.use_sampled_cam:
                pred_scale, pred_trans, pred_quat, _ = cam_pred
                rotation, translation = get_scaled_orthographic_projection(
                    pred_scale, pred_trans, pred_quat)
            else:
                pred_scale, pred_trans, pred_quat, _ = pred_poses
                rotation, translation = get_scaled_orthographic_projection(
                    pred_scale.view(-1), pred_trans.view(-1, 2), pred_quat.view(-1, 4)
                )
                rotation = rotation.view(batch_size, -1, 3, 3)
                translation = translation.view(batch_size, -1, 3)

        if self.use_gt_cam or self.use_sampled_cam:
            rotation = rotation.unsqueeze(1)
            translation = translation.unsqueeze(1)
        
        return rotation, translation, pred_poses

    def _get_projected_positions_of_sphere_points(self, sphere_points, rotation, translation):
        """
        For the given points on unit sphere calculates the 3D coordinates on the mesh template
        and projects them back to image plane

        :param sphere_points: A (B X 3 X H X W) tensor containing the predicted points on the sphere
        :param rotation: A (B X CP X 3 X 3) camera rotation tensor
        :param translation: A (B X CP X 3) camera translation tensor
        :return: A tuple(xy, z, uv, uv_3d)
            - xy - (B X CP X 2 X H X W) x,y values of the 3D points after projecting onto image plane
            - z - (B X CP X 1 X H X W) z value of the projection
            - uv - (B X 2 X H X W) UV values of the sphere coordinates
            - uv_3d - (B X H X W X 3) tensor with the 3D coordinates on the mesh
                template for the given sphere coordinates
        """

        uv = convert_3d_to_uv_coordinates(sphere_points.permute(0, 2, 3, 1))
        batch_size = uv.size(0)
        height = uv.size(1)
        width = uv.size(2)
        num_poses = rotation.size(1)

        uv_flatten = uv.view(-1, 2)
        uv_3d = self.uv_to_3d(uv_flatten).view(batch_size, 1, -1, 3)
        uv_3d = uv_3d.repeat(1, num_poses, 1, 1).view(batch_size*num_poses, -1, 3)

        cameras = OpenGLOrthographicCameras(device=sphere_points.device, R=rotation.view(-1, 3, 3), T=translation.view(-1, 3))
        xyz_cam = cameras.get_world_to_view_transform().transform_points(uv_3d)
        z = xyz_cam[:, :, 2:].view(batch_size, num_poses, height, width, 1)
        xy = cameras.transform_points(uv_3d)[:, :, :2].view(batch_size, num_poses, height, width, 2)

        xy = xy.permute(0, 1, 4, 2, 3).flip(2)
        z = z.permute(0, 1, 4, 2, 3)
        uv = uv.permute(0, 3, 1, 2)
        uv_3d = uv_3d.view(batch_size, num_poses, height, width, 3)[:, 0, :, :, :].squeeze()

        return xy, z, uv, uv_3d

    def _render(self, rotation: torch.Tensor, translation: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        batch_size = rotation.size(0)
        cam_poses = rotation.size(1)

        pred_mask, pred_depth = self.renderer(
            rotation.view(-1, 3, 3),
            translation.view(-1, 3))

        height = pred_mask.size(1)
        width = pred_mask.size(2)

        pred_mask = pred_mask.view(batch_size, cam_poses, 1, height, width)
        pred_depth = pred_depth.view(batch_size, cam_poses, 1, height, width)

        # Pytorch renderer returns -1 values for the empty pixels which
        # when directly used results in wrong loss calculation so changing the values to the max + 1
        pred_depth = pred_depth * torch.ceil(pred_mask) + (1 - torch.ceil(pred_mask)) * pred_depth.max()

        return pred_mask, pred_depth
