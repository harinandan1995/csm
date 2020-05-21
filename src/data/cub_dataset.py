import os.path as osp

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data

from src.data import transformations
from src.data.dataset import IDataset
from src.utils.utils import validate_paths

# curr_path = osp.dirname(osp.abspath(__file__))
# cache_path = osp.join(curr_path, '..', 'cachedir')
# cub_path = osp.join(curr_path, '..', 'CUB_200_2011')
#
# flags.DEFINE_string('cub_cache_dir', osp.join(cache_path, 'cub'), 'CUB Data Directory')
# flags.DEFINE_string('cub_dir', cub_path, 'CUB Data Directory')


class CubDataset(IDataset):

    def __init__(self, config):

        self.cache_dir = config.dir.cache_dir
        self.data_dir = config.dir.data_dir
        self.img_dir = osp.join(self.data_dir, 'images')

        super(CubDataset, self).__init__(config)

        self.anno = []
        self.anno_sfm = []
        self.mean_shape = {}
        self.num_samples = 0

        self.load_data()

    def __len__(self):
        """
        :return: number of images
        """

        return self.num_samples

    def load_key_points(self):

        kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
                    'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']

        anno_train_sfm_path = osp.join(self.cache_dir, 'sfm', 'anno_%s.mat' % 'train')
        validate_paths(anno_train_sfm_path)

        kp_3d = sio.loadmat(anno_train_sfm_path, struct_as_record=False, squeeze_me=True)['S'].transpose().copy()
        kp_3d = torch.tensor(kp_3d)

        self.mean_shape = sio.loadmat(osp.join(self.cache_dir, 'uv', 'mean_shape.mat'))
        kp_uv = self.preprocess_to_find_kp_uv(
            kp_3d,
            self.mean_shape['faces'],
            self.mean_shape['verts'],
            self.mean_shape['sphere_verts'])

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
        self.d3_data = self.get_3d_data()

    def get_data(self, index):

        data = self.anno[index]
        data_sfm = self.anno_sfm[index]
        img_path = osp.join(self.img_dir, str(data.rel_path))

        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]
        sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)
        parts = data.parts.T.astype(float)

        return data, data.bbox, data.mask, data.parts.T.astype(float), sfm_pose, img_path

    def get_img_data(self, index):
        """
        :param index: the index of image
        :return: A list of dicts of  contains info of the given index image
        img: A np.ndarray 3*256*256, index given image after crop and mirror (if train)
        kp_uv: A torch.Tensor 15*2， key points in uv coordinate
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

    def get_3d_data(self,device = "cuda"):
        """
        :return: a dict contains info of mean shape:
        A torch.Tensor 15*3, 3d key_points
        A torch.Tensor 15, key_point_indx
        A list 15, key_point names
        A np.ndarray 15*2,projected in uv-cordinate key_points
        A torch.Tensor 1001*1001*3, bary_cord
        A torch.Tensor 144*3, conv_tri
        A torch.Tensor 1001*1001*2, uv_map
        A torch.Tensor 15*3, (3d key_points)S
        A torch.Tensor 642*3, verts
        A torch.Tensor 1280*3, faces
        A torch.Tensor 1001*1001, face_inds
        A torch.Tensor 642*2, uv_verts
        A torch.Tensor 642*3, sphere_verts
        """
        dic = {
            'kp3d': self.kp_3d,
            'kp_perm': self.kp_perm,
            'kp_names': self.kp_names,
            'kp_uv': self.kp_uv
        }
        for i in self.mean_shape:
            if type(self.mean_shape[i]) == np.ndarray:
                self.mean_shape[i] = torch.from_numpy(self.mean_shape[i]).float().to(device)
        dic.update(self.mean_shape)

        return dic
