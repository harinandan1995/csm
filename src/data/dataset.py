import cv2
import imageio
import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset

from src.data.utils import image, transformations
from src.data.utils.image import get_texture_map, get_template_texture
from src.nnutils.geometry import convert_3d_to_uv_coordinates


class IDataset(Dataset):
    """
    Interface for the dataset
    """

    def __init__(self, config, device='cuda'):

        self.config = config
        self.img_size = config.img_size
        self.jitter_frac = config.jitter_frac
        self.padding_frac = config.padding_frac
        self.rngFlip = np.random.RandomState(0)
        self.transform = config.transform
        self.flip = config.flip
        self.device = device

        self.mean_shape = self._get_mean_shape()
        self.texture_map = get_texture_map(self.config.dir.texture)
        self.template_mesh = self._get_template_mesh()

        self.kp_3d, self.kp_uv, self.kp_names, self.kp_perm = self.load_key_points()

    def __len__(self):

        raise NotImplementedError('get_len method should be implemented in the child class')

    def __getitem__(self, index):
        """
        :param index: the index of image
        :return: a dict contains info of the given index image
        img: A np.ndarray 3*256*256, index given image after crop and mirror (if train)
        kp: A np.ndarray 15*3, key points
        kp_uv: A np.ndarray 15*2， key points in uv coordinate
        mask: A np.ndarray 256*256, mask after transformation
        sfm_pose: sfm_pose after transformation
        np.ndarray 1, scale,
        np.ndarray 1*2, trans
        np.ndarray 1*4, quaternion,
        inds: np.ndarray of given indexs

        if self.transform == 'flip'
        flip_img: A np.ndarray 3*256*256, img after flip
        flip_mask: A np.ndarray 256*256, mask after transformation
        """
        img, kp, kp_uv, mask, sfm_pose = self.forward_img(index)
        elem = {
            'img': img,
            'kp': kp,
            'kp_uv': kp_uv,
            'mask': mask,
            'scale': sfm_pose[0],  # scale (1), trans (2), quat(4)
            'trans': sfm_pose[1],
            'quat': sfm_pose[2],
            'inds': np.array([index]),
        }
        if self.transform == 'flip':
            flip_img = img[:, :, ::-1].copy()
            elem['flip_img'] = flip_img
            flip_mask = mask[:, ::-1].copy()
            elem['flip_mask'] = flip_mask
        return elem

    def _get_mean_shape(self) -> dict:
        """
        Gets the template information

        :return: A tuple (mean_shape, template_mesh)
        mean_shape is a dictionary containing the following values
            uv_map - A R X R tensor of defining the UV steps. Where R is the resolution of the UV map.
            uv_vertices - A (None, 2) tensor with UV values for the vertices
            verts - A (None, 3) tensor of vertex coordinates of the mean shape
            sphere_verts - A (None, 3) tensor with sphere coordinates for the vertices
                calculated by projecting the vertices onto a sphere
            face_inds - A R X R tensor where each value is the index of the face for
            faces - A (None, 3) tensor of faces of the mean shape
        template_mesh is a pytorch3D mesh object
        """

        return NotImplementedError('Must be implemented by a child class')

    def _get_template_mesh(self) -> Meshes:

        if 'template' in self.config.dir:
            mesh = trimesh.load(self.config.dir.template, 'obj')
            vertices = torch.from_numpy(np.asarray(mesh.vertices)).to(self.device, dtype=torch.float)
            faces = torch.from_numpy(np.asarray(mesh.faces)).to(self.device, dtype=torch.long)
        else:
            vertices = self.mean_shape['verts'].to(torch.float)
            faces = self.mean_shape['faces'].to(torch.long)

        template_texture = get_template_texture(vertices, faces, self.texture_map)
        template_mesh = Meshes(verts=[vertices], faces=[faces], textures=template_texture).to(self.device)

        return template_mesh

    def load_key_points(self):
        """
        :return: A tuple containing the below values in the same order
        3D key points (None, 3),
        UV values for the key points (None, 2)
        key point names (None),
        key point permutation (None),
        """

        return NotImplementedError('Must be implemented by a child class')

    def get_data(self, index: int) -> tuple:
        """
        For the given index the child class should return a

        :param index: index of the sample required
        :return: tuple (bbox, mask, parts, pose, img_path)
            bbox - A np array of shape [4] with the x1, y1, x2, y2 coordinates of the bounding box
            mask - (H X W) numpy array. Foreground mask
            parts - (KP X 3) numpy array containing the 2D positions of the keypoints on the image and
                a value showing whether or not the key point is visible in the image.
            sfm_pose - A list of 3 numpy arrays of shapes (), (2,), (4,) with scale, translation and quaternions
            img_path - Path to image
        """

        raise NotImplementedError('get_items method should be implemented in the child class')

    # Space to implement common functions that can be used across multiple datasets
    def forward_img(self, index):
        """
        :param index: list or int, the index of image
        :return:
        img: A np.ndarray 3*256*256, index given image after crop and mirror (if train)
        kp_norm: A np.ndarray 15*3 , key points after transformation and normalization
        kp_uv: A np.ndarray 15*2， key points in uv coordinate
        mask: A np.ndarray 256*256, mask after transformation
        sfm_pose: sfm_pose after transformation
        float, scale,
        np.ndarray 1*2, trans
        np.ndarray 1*4, quaternion
        """

        bbox, mask, parts, sfm_pose, img_path = self.get_data(index)

        img = imageio.imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(mask, 2)

        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1
        kp_uv = self.kp_uv.copy()
        # Peturb bbox

        if self.config.tight_crop:
            self.padding_frac = 0.0

        if self.config.split == 'train':
            bbox = image.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)

        if self.config.tight_crop:
            bbox = bbox
        else:
            bbox = image.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)

        # scale image, and mask. And scale kps.
        if self.config.tight_crop:
            img, mask, kp, sfm_pose = self.scale_image_tight(img, mask, kp, vis, sfm_pose)
        else:
            img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)

        # Mirror image on random.
        if self.config.split == 'train':
            img, mask, kp, kp_uv, sfm_pose = self.mirror_image(img, mask, kp, kp_uv, sfm_pose)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))
        return img, kp_norm, kp_uv, mask, sfm_pose

    @staticmethod
    def crop_image(img, mask, bbox, kp, vis, sfm_pose):
        """
        crop image and mask and translate kps

        :param img: A np.ndarray img_height*img_width*3, index given image after crop and mirror (if train)
        :param mask: A np.ndarray img_height*img_width, mask
        :param bbox: A np.ndarray 1*4, [x1,y1,x2,y2]
        :param kp: A np.ndarray 15*3 , key points
        :param vis: A np.ndarray 15*1 false if kp is [0,0,0], key_point visualble
        :param sfm_pose:
        float, scale,
        np.ndarray 1*2, trans
        np.ndarray 1*4, quaternion
        :return:
        img: A np.ndarray (y2-y1)*(x2-x1)*3
        mask A np.ndarray (y2-y1)*(x2-x1)
        kp: A np.ndarray 15*3 , key points after crop
        sfm_pose: sfm_pose after crop
        float, scale,
        np.ndarray 1*2, trans
        np.ndarray 1*4, quaternion
        """
        img = image.crop(img, bbox, bgval=1)
        mask = image.crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]

        kp[vis, 0] = np.clip(kp[vis, 0], a_min=0, a_max=bbox[2] - bbox[0])
        kp[vis, 1] = np.clip(kp[vis, 1], a_min=0, a_max=bbox[3] - bbox[1])

        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]

        return img, mask, kp, sfm_pose

    def scale_image_tight(self, img, mask, kp, vis, sfm_pose):
        """
        Scale image to 256*256 with xscale and yscale

        :param img: A np.ndarray (y2-y1)*(x2-x1)*3
        :param mask: A np.ndarray (y2-y1)*(x2-x1)
        :param kp: A np.ndarray 15*3 , key points after crop
        :param vis: A np.ndarray 15*1 false if kp is [0,0,0], visualble
        :param sfm_pose: sfm_pose after crop
        float, scale,
        np.ndarray 1*2, trans
         np.ndarray 1*4, quaternion
        :return:
        img_scale: A np.ndarray 256*256*3, scaled image
        mask_scale: A np.ndarray 256*256, scaled mask
        kp: A np.ndarray 15*3 , scaled key points
        sfm_pose
        sfm_pose: sfm_pose after scale
        float, scale,
        np.ndarray 1*2, trans
         np.ndarray 1*4, quaternion
        """
        bwidth = np.shape(img)[1]
        bheight = np.shape(img)[0]

        scale_x = self.img_size / bwidth
        scale_y = self.img_size / bheight

        img_scale = cv2.resize(img, (self.img_size, self.img_size))

        mask_scale = cv2.resize(mask, (self.img_size, self.img_size))

        kp[vis, 0:1] *= scale_x
        kp[vis, 1:2] *= scale_y
        sfm_pose[0] *= scale_x
        sfm_pose[1] *= scale_y

        return img_scale, mask_scale, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        """
        Scale image to 256*256 with max(xscale, yscale)

        :param img: A np.ndarray (y2-y1)*(x2-x1)*3
        :param mask: A np.ndarray (y2-y1)*(x2-x1)
        :param kp: A np.ndarray 15*3 , key points after crop
        :param vis: A np.ndarray 15*1 true if kp is not [0,0,0], key points visualble
        :param sfm_pose: sfm_pose after crop
        float, scale,
        np.ndarray 1*2, trans
         np.ndarray 1*4, quaternion,
        :return:
        img_scale: A np.ndarray 256*256*3, scaled image
        mask_scale: A np.ndarray 256*256, scaled mask
        kp: A np.ndarray 15*3 , scaled key points
        sfm_pose: sfm_pose after scale
        float, scale,
        np.ndarray 1*2, trans
         np.ndarray 1*4, quaternion
        """
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image.resize_img(img, scale)
        mask_scale, _ = image.resize_img(mask, scale)
        kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, kp_uv, sfm_pose):
        """
        half of img left-right

        :param img: A np.ndarray 256*256*3
        :param mask: A np.ndarray 256*256
        :param kp: A np.ndarray 15*3 , key points
        :param kp_uv: A np.ndarray 15*2, key points in uv_coordinate
        :param sfm_pose: sfm_pose after crop
        float, scale,
        np.ndarray 1*2, trans
         np.ndarray 1*4, quaternion,
        :return:
        img_scale: A np.ndarray 256*256*3, left-right image
        mask_scale: A np.ndarray 256*256, left-right mask
        kp: A np.ndarray 15*3 , left-right points
        kp_uv: A np.ndarray 15*2, left-right key points in uv_coordinate
        sfm_pose
        sfm_pose: sfm_pose after left-right
        float, scale,
        np.ndarray 1*2, trans
         np.ndarray 1*4, quaternion
        """
        kp_perm = self.kp_perm
        if self.rngFlip.rand(1) > 0.5 and self.config.flip:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            # Flip kps.
            new_x = img.shape[1] - kp[:, 0] - 1
            kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
            kp_flip = kp_flip[kp_perm, :]
            kp_uv_flip = kp_uv[kp_perm, :]

            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)

            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx

            return img_flip, mask_flip, kp_flip, kp_uv_flip, sfm_pose
        else:
            return img, mask, kp, kp_uv, sfm_pose

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        """
        :param kp: A np.ndarray 15*3 , key points
        :param sfm_pose:
        float, scale,
        np.ndarray 1*2, trans
         np.ndarray 1*4, quaternion
        :param img_h: int, height of image
        :param img_w: int, width of image
        :return:
        new_kp: A np.ndarray 15*3 , key points after normalization
        sfm_pose:scale, trans, quaternion after normalization
        float, scale,
        np.ndarray 1*2, trans
         np.ndarray 1*4, quaternion
        """
        vis = kp[:, 2, None] > 0
        new_kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                           2 * (kp[:, 1] / img_h) - 1,
                           kp[:, 2]]).T
        sfm_pose[0] *= (1.0 / img_w + 1.0 / img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp

        return new_kp, sfm_pose

    def preprocess_to_find_kp_uv(self, kp3d, faces, verts):
        """
        Project 3d key points to closest point on mesh and projected in uv_coordinate

        :param kp3d: A np.ndarray 15*3, 3d key points
        :param faces: A np.ndarray F*3 ,faces of mesh; F - number of faces
        :param verts: A np.ndarray V*3 ,vertexs of mesh; V - number of vertices
        :return:kp_uv: A np.ndarray 15*2, projected key points in uv coordinate
        """
        mesh = trimesh.Trimesh(verts, faces)
        closest_point = trimesh.proximity.closest_point(mesh, kp3d)
        kp_uv = convert_3d_to_uv_coordinates(closest_point[0])

        return kp_uv
