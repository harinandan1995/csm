import os.path as osp

import numpy as np
import scipy.io as sio

from src.data.dataset import IDataset
from src.data.utils import transformations
from src.nnutils.geometry import load_mean_shape
from src.utils.utils import validate_paths


class P3DDataset(IDataset):

    def __init__(self, config, device='cuda'):

        super(P3DDataset, self).__init__(config, device)

        self.img_dir = osp.join(config.dir.data_dir, 'images')
        self.anno = []
        self.anno_sfm = []
        self.num_samples = 0
        self.load_data()

    def __len__(self):

        return self.num_samples

    def _get_mean_shape(self):

        mean_shape = load_mean_shape(
            osp.join(self.config.dir.cache_dir, 'uv', '%s_mean_shape.mat' % self.config.category), device=self.device)

        return mean_shape

    def load_key_points(self):

        kp_path = osp.join(self.config.dir.cache_dir, 'data', '%s_kps.mat' % self.config.category)
        anno_train_sfm_path = osp.join(self.config.dir.cache_dir, 'sfm', '%s_train.mat' % self.config.category)
        validate_paths(anno_train_sfm_path, kp_path)

        kp_perm = sio.loadmat(kp_path, struct_as_record=False, squeeze_me=True)['kp_perm_inds'] - 1
        kp_names = sio.loadmat(kp_path, struct_as_record=False, squeeze_me=True)['kp_names'].tolist()
        kp_3d = sio.loadmat(anno_train_sfm_path, struct_as_record=False, squeeze_me=True)['S'].transpose().copy()

        kp_uv = self.preprocess_to_find_kp_uv(
            kp_3d,
            self.mean_shape['faces'].cpu().numpy(),
            self.mean_shape['verts'].cpu().numpy())

        return kp_3d, kp_uv, kp_names, kp_perm

    def get_data(self, index):

        data = self.anno[index]
        data_sfm = self.anno_sfm[index]
        img_path = osp.join(self.img_dir, str(data.rel_path))

        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]
        sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)
        parts = data.parts.T.astype(float)

        bbox = np.array([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], float) - 1

        return bbox, data.mask, parts, sfm_pose, img_path

    def load_data(self):

        cache_dir = self.config.dir.cache_dir

        anno_path = osp.join(cache_dir, 'data', '%s_%s.mat' % (self.config.category, self.config.split))
        anno_sfm_path = osp.join(cache_dir, 'sfm', '%s_%s.mat' % (self.config.category, self.config.split))

        validate_paths(anno_path, anno_sfm_path)

        # Load the annotation file.
        print('Loading p3D annotations from %s' % anno_path)
        self.anno = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_samples = len(self.anno)

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
