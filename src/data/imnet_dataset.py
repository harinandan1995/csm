import os.path as osp

import numpy as np
import scipy.io as sio

from src.data.dataset import IDataset
from src.data.utils import transformations
from src.data.utils.imnet import get_sysnet_id_for_imnet_class
from src.nnutils.geometry import load_mean_shape
from src.utils.utils import validate_paths


class ImnetDataset(IDataset):

    def __init__(self, config, device='cuda'):
        self.cache_dir = config.dir.cache_dir

        super(ImnetDataset, self).__init__(config, device)

        self.sysnet_id = get_sysnet_id_for_imnet_class(self.config.category)
        self.img_dir = osp.join(config.dir.data_dir, 'ImageSets', self.sysnet_id)
        self.category = self.config.category
        self.anno = []
        self.anno_sfm = []
        self.num_samples = 0
        self.load_data()

    def __len__(self):

        return self.num_samples

    def _get_mean_shape(self):

        mean_shape = load_mean_shape(
            osp.join(self.config.dir.cache_dir, 'shapenet', self.category, 'shape.mat'),
            device=self.device)

        return mean_shape

    def load_key_points(self):

        kp_perm = np.linspace(0, 9, 10).astype(np.int)
        kp_names = ['lpsum' for _ in range(len(self.kp_perm))]
        kp_uv = np.random.uniform(0, 1, (len(self.kp_perm), 2))

        return None, kp_uv, kp_names, kp_perm

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

        anno_path = osp.join(cache_dir, 'data', '%s_%s.mat' % (self.sysnet_id, self.config.split))
        anno_sfm_path = osp.join(cache_dir, 'sfm', '%s_%s.mat' % (self.sysnet_id, self.config.split))

        validate_paths(anno_path, anno_sfm_path)

        # Load the annotation file.
        print('Loading imagenet annotations from %s' % anno_path)
        self.anno = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_samples = len(self.anno)
