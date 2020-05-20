import os.path as osp

import cv2
import imageio
import numpy as np
import torch
from absl import flags
from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.utils.data import Dataset

from src.data import image
from src.data import transformations
from src.nnutils.geometry import convert_3d_to_uv_coordinates

# flags.DEFINE_integer('img_size', 256, 'image size')
# flags.DEFINE_integer('img_height', 320, 'image height')
# flags.DEFINE_integer('img_width', 512, 'image width')
# flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
# flags.DEFINE_enum('transform', 'flip', ['flip'], 'eval split')
# flags.DEFINE_float('padding_frac', 0.05, 'bbox is increased by this fraction of max_dim')
# flags.DEFINE_float('jitter_frac', 0.05, 'bbox is jittered by this fraction of max_dim')
# flags.DEFINE_boolean('flip', True, 'Allow flip bird left right')
# flags.DEFINE_boolean('tight_crop', False, 'Use Tight crops')
# flags.DEFINE_boolean('flip_train', True, 'Mirror Images while training')
# flags.DEFINE_integer('number_pairs', 10000,
#                      'N random pairs from the test to check if the correspondence transfers.')
# flags.DEFINE_integer('num_kps', 15, 'Number of keypoints')


class IDataset(Dataset):
    """
    Interface for the dataset
    """

    def __init__(self, config):

        self.config = config
        self.img_size = config.img_size
        self.jitter_frac = config.jitter_frac
        self.padding_frac = config.padding_frac
        self.rngFlip = np.random.RandomState(0)
        self.transform = config.transform
        self.device = torch.device('cuda')
        self.flip = config.flip

        self.kp_3d, self.kp_uv, self.kp_names, self.kp_perm = self.load_key_points()

    def __len__(self):

        raise NotImplementedError('get_len method should be implemented in the child class')

    def __getitem__(self, index):
        """
        :param index: the index of image
        :return: a dict contains info of the given index image
        img: A np.ndarray 3*256*256, index given image after crop and mirror (if train)
        kp_uv: A torch.Tensor 15*2， key points in uv coordinate
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

    def load_key_points(self):
        """

        :return: A tuple containing the below values in the same order
        3D key points (None, 3),
        UV values for the key points (None, 2)
        key point names (None),
        key point permutation (None),
        """

        return NotImplementedError('Must be implemented by a child class')

    def get_data(self, index):
        """
        Child class must implement this method, data should be a list of dictionaries with
        each containing the info of single image

        :return: list of data in expected format
        """

        raise NotImplementedError('get_items method should be implemented in the child class')

    # TODO: Write the augmentation functions if necessary, like vertical & horizontal flips, contrast adjustments etc.
    #  check https://pytorch.org/docs/stable/torchvision/transforms.html for transformations

    # Space to implement common functions that can be used across multiple datasets
    def forward_img(self, index):
        """
        :param index: list or int, the index of image
        :return:
        img: A np.ndarray 3*256*256, index given image after crop and mirror (if train)
        kp_norm: A np.ndarray 15*3 , key points after transformation and normalization
        kp_uv: A torch.Tensor 15*2， key points in uv coordinate
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

        # Adjust to 0 indexing
        bbox = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], float) - 1

        # parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1
        kp_uv = self.kp_uv.clone()

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
        :param kp_uv: A torch.Tensor 15*2, key points in uv_coordinate
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

    def preprocess_to_find_kp_uv(self, kp3d, faces, verts, verts_sphere):
        """
        Project 3d key points to closest point on mesh and projected in uv_coordinate

        :param kp3d: A torch.Tensor 15*3, 3d key points
        :param faces: A np.ndarray 1280*3 ,faces of mesh
        :param verts: A np.ndarray 642*3 ,vertexs of mesh
        :param verts_sphere: A np.ndarray 642*3, vertexs of mesh sphere
        :return:kp_uv: A torch.Tensor 15*2, projected key points in uv coordinate
        """
        vert = torch.tensor(verts).unsqueeze(dim=0).to(self.device)
        face = torch.tensor(faces).unsqueeze(dim=0).to(self.device)
        mesh = Meshes(vert, face)
        tp = torch.unsqueeze(kp3d, dim=0)
        ep = Pointclouds(tp)
        points = ep.points_packed()  # (P, 3)
        points_first_idx = ep.cloud_to_packed_first_idx()
        max_points = ep.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = mesh.verts_packed()
        faces_packed = mesh.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = mesh.mesh_to_faces_packed_first_idx()
        max_tris = mesh.num_faces_per_mesh().max().item()
        dists, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_tris
        )
        nfaces = verts_packed[faces_packed[idxs]]
        V = nfaces[:, 1] - nfaces[:, 0]
        W = nfaces[:, 2] - nfaces[:, 0]
        nx = V[:, 1] * W[:, 2] - V[:, 2] * W[:, 1]
        ny = V[:, 2] * W[:, 0] - V[:, 0] * W[:, 2]
        nz = V[:, 0] * W[:, 1] - V[:, 1] * W[:, 0]
        nz = V[:, 0] * W[:, 1] - V[:, 1] * W[:, 0]
        nx = nx.reshape(15, 1)
        ny = ny.reshape(15, 1)
        nz = nz.reshape(15, 1)
        n = torch.cat((nx, ny, nz), 1)
        np0 = kp3d + dists.reshape(15, 1) * n
        kp_uv = convert_3d_to_uv_coordinates(np0)
        kpc = kp3d.cpu().numpy()
        dist_to_verts = np.square(kpc[:, None, :] - verts[None, :, :]).sum(-1)
        min_inds = np.argmin(dist_to_verts, axis=1)
        kp_verts_sphere = verts_sphere[min_inds]
        return kp_uv
