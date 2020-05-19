import os.path as osp

import numpy as np
import scipy.io as sio
import torch
from absl import flags

from src.data.dataset import IDataset

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
cub_path = osp.join(curr_path, '..', 'CUB_200_2011')

flags.DEFINE_string('cub_cache_dir', osp.join(cache_path, 'cub'), 'CUB Data Directory')
flags.DEFINE_string('cub_dir', cub_path, 'CUB Data Directory')


class CubDataset(IDataset):

    def __init__(self, config):

        super(CubDataset, self).__init__(config)
        self.data_dir = config.cub_dir
        self.data_cache_dir = config.cub_cache_dir
        self.config = config
        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % config.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % config.split)
        self.anno_train_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % 'train')
        self.jitter_frac = config.jitter_frac
        self.padding_frac = config.padding_frac
        self.img_size = config.img_size
        if not osp.exists(self.anno_path):
            raise NotImplementedError('%s doesnt exist!' % self.anno_path)
        device = torch.device('cuda')
        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
        self.mean_shape = sio.loadmat(osp.join(config.cub_cache_dir, 'uv', 'mean_shape.mat'))
        self.kp3d = sio.loadmat(self.anno_train_sfm_path, struct_as_record=False,
                                squeeze_me=True)['S'].transpose().copy()
        self.kp3d = torch.Tensor(self.kp3d).to(device)

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        self.kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
                         'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']

        self.kp_uv = self.preprocess_to_find_kp_uv(self.kp3d, self.mean_shape['faces'], self.mean_shape[
            'verts'], self.mean_shape['sphere_verts'])
        self.flip = config.flip
        self.d3_data = self.get_3d_data()
        return

    def __len__(self):
        """
        :return: number of images
        """

        return self.num_imgs

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
        inds: list of given indexs

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
        A A torch.Tensor 15*3, 3d key_points
        A np.ndarray 15, key_point_indx
        A list 15, key_point names
        A torch.Tensor 15*2,projected in uv-cordinate key_points
        A np.ndarray 1001*1001*3, bary_cord
        A np.ndarray 144*3, conv_tri
        A np.ndarray 1001*1001*2, uv_map
        A np.ndarray 15*3, (3d key_points)S
        A np.ndarray 642*3, verts
        A np.ndarray 1280*3, faces
        A np.ndarray 1001*1001, face_inds
        A np.ndarray 642*2, uv_verts
        A np.array 642*3, sphere_verts
        """
        dic = {'kp3d': self.kp3d, 'kp_perm': self.kp_perm, 'kp_names': self.kp_names, 'kp_uv': self.kp_uv}
        dic.update(self.mean_shape)
        return dic
