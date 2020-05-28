import os.path as osp

import numpy as np
import scipy.io as sio
import torch
import trimesh
from PIL import Image
from pytorch3d.structures import Meshes, Textures

from src.data import transformations
from src.data.dataset import IDataset
from src.nnutils.geometry import load_mean_shape, convert_3d_to_uv_coordinates
from src.utils.utils import validate_paths


class CubDataset(IDataset):

    def __init__(self, config, device='cuda'):

        self.cache_dir = config.dir.cache_dir
        self.data_dir = config.dir.data_dir
        self.img_dir = osp.join(self.data_dir, 'images')

        super(CubDataset, self).__init__(config, device)

        self.anno = []
        self.anno_sfm = []
        self.num_samples = 0
        self.load_data()

    def __len__(self):
        """
        :return: number of images
        """

        return self.num_samples

    def _get_template_info(self):

        mean_shape = load_mean_shape(osp.join(self.cache_dir, 'uv', 'mean_shape.mat'), device=self.device)

        texture_map = self._get_texture_map(self.config.dir.texture)
        mesh = trimesh.load(self.config.dir.template, 'obj')
        vertices = torch.from_numpy(np.asarray(mesh.vertices)).to(torch.float)
        faces = torch.from_numpy(np.asarray(mesh.faces)).to(torch.long)
        template_texture = self._get_template_texture(vertices, faces, texture_map)

        template_mesh = Meshes(verts=[vertices], faces=[faces], textures=template_texture).to(self.device)

        return mean_shape, template_mesh, texture_map.to(self.device)

    @staticmethod
    def _get_template_texture(vertices, faces, texture_map):

        verts_uv = convert_3d_to_uv_coordinates(vertices)
        vertex_rgb = torch.nn.functional.grid_sample(texture_map.unsqueeze(0),
                                                     2 * verts_uv.unsqueeze(0).unsqueeze(0) - 1)
        vertex_rgb = vertex_rgb.squeeze(2).permute(0, 2, 1) * 255
        texture = Textures([texture_map.permute(1, 2, 0)],
                           faces_uvs=faces.unsqueeze(0),
                           verts_uvs=verts_uv.unsqueeze(0),
                           verts_rgb=vertex_rgb)

        return texture

    @staticmethod
    def _get_texture_map(img_path):

        texture_map = Image.open(img_path).resize((256, 256), Image.ANTIALIAS)
        texture_map = torch.from_numpy(np.asarray(texture_map)).permute(2, 0, 1)

        return texture_map.to(dtype=torch.float) / 255

    def load_key_points(self):

        kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
                    'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']

        anno_train_sfm_path = osp.join(self.cache_dir, 'sfm', 'anno_%s.mat' % 'train')
        validate_paths(anno_train_sfm_path)

        kp_3d = sio.loadmat(anno_train_sfm_path, struct_as_record=False, squeeze_me=True)['S'].transpose().copy()

        kp_uv = self.preprocess_to_find_kp_uv(
            kp_3d,
            self.mean_shape['faces'].cpu().numpy(),
            self.mean_shape['verts'].cpu().numpy())

        return kp_3d, kp_uv, kp_names, kp_perm

    def load_data(self):

        cache_dir = self.cache_dir

        anno_path = osp.join(cache_dir, 'data', '%s_cub_cleaned.mat' % self.config.split)
        anno_sfm_path = osp.join(cache_dir, 'sfm', 'anno_%s.mat' % self.config.split)

        validate_paths(anno_path, anno_sfm_path)

        # Load the annotation file.
        print('loading %s' % anno_path)
        self.anno = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_samples = len(self.anno)

    def get_data(self, index):

        data = self.anno[index]
        data_sfm = self.anno_sfm[index]
        img_path = osp.join(self.img_dir, str(data.rel_path))

        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]
        sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)
        parts = data.parts.T.astype(float)

        return data.bbox, data.mask, parts, sfm_pose, img_path

    def get_img_data(self, index):
        """
        :param index: the index of image
        :return: A list of dicts of  contains info of the given index image
        img: A np.ndarray 3*256*256, index given image after crop and mirror (if train)
        kp_uv: A np.ndarray  15*2， key points in uv coordinate
        mask: A np.ndarray 256*256, mask after transformation
        sfm_pose: sfm_pose after transformation
        float, scale,
        np.ndarray 1*2, trans
        np.ndarray 1*4, quaternion,
        inds: list of given indices

        if self.transform == 'flip'
        flip_img: A np.ndarray 3*256*256, img after flip
        flip_mask: A np.ndarray 256*256, mask after transformation
        """

        ty_idx = type(index)
        if ty_idx != int and ty_idx != list:
            raise TypeError("Invalid type of index")
        elif ty_idx == int:
            index = [index]
        res = []
        for idx in index:
            dic = self.__getitem__(idx)
            res.append(dic)
        return res

    def get_3d_data(self):
        """
        :return: a dict contains info of mean shape:
        A np.ndarray  15*3, 3d key_points
        A np.ndarray  15, key_point_indx
        A list 15, key_point names
        A np.ndarray 15*2,projected in uv-cordinate key_points
        A np.ndarray  1001*1001*3, bary_cord
        A np.ndarray  144*3, conv_tri
        A np.ndarray  1001*1001*2, uv_map
        A np.ndarray  15*3, (3d key_points)S
        A np.ndarray  V*3, verts
        A np.ndarray  F*3, faces
        A np.ndarray  1001*1001, face_inds
        A np.ndarray  V*2, uv_verts
        A np.ndarray  V*3, sphere_verts
        F - number of faces
        V - number of vertices
        """
        dic = {
            'kp3d': self.kp_3d,
            'kp_perm': self.kp_perm,
            'kp_names': self.kp_names,
            'kp_uv': self.kp_uv
        }
        dic.update(self.mean_shape)

        return dic
