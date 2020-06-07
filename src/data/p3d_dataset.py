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

        self.img_dir = osp.join(config.dir.data_dir, 'Images')
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

