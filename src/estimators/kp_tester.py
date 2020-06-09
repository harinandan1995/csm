import numpy as np
import torch
from pytorch3d.ops.cubify import unravel_index

from src.data.cub_dataset import CubDataset
from src.data.dataset import KPDataset
from src.data.imnet_dataset import ImnetDataset
from src.data.p3d_dataset import P3DDataset
from src.estimators.tester import ITester
from src.model.csm import CSM
from src.nnutils.color_transform import draw_key_points, sample_uv_contour
from src.utils.config import ConfigParser


class KPTransferTester(ITester):

    def __init__(self, config: ConfigParser.ConfigObject, device):

        self.device = device
        self.data_cfg = config.dataset
        super(KPTransferTester, self).__init__(config.test)
        self.key_point_colors = np.random.uniform(0, 1, (len(self.dataset.kp_names), 3))

    def _batch_call(self, step, batch_data):

        src, tar = batch_data

        src_img = src['img'].to(self.device, dtype=torch.float)
        tar_img = tar['img'].to(self.device, dtype=torch.float)
        tar_mask = tar['mask'].unsqueeze(1).to(self.device, dtype=torch.float)

        src_pred_out = self._call_model(src)
        tar_pred_out = self._call_model(tar)

        self._add_uv_summaries(src_pred_out, tar_pred_out, src, tar, step)

        src_kp = src['kp'].to(self.device, dtype=torch.float)
        tar_kp = tar['kp'].to(self.device, dtype=torch.float)

        tar_pred_kp = self._find_target_kps(src_kp, src_pred_out['uv'], tar_pred_out['uv'], tar_mask)
        tar_pred_kp = torch.cat((tar_pred_kp, self._convert_to_int_indices(src_kp)[:, :, 2:].to(torch.int64)), dim=2)

        self._add_kp_summaries(self._convert_to_int_indices(src_kp), self._convert_to_int_indices(tar_kp),
                               tar_pred_kp, src_img, tar_img, step)
    
    def _call_model(self, data):

        img = data['img'].to(self.device, dtype=torch.float)
        mask = data['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        scale = data['scale'].to(self.device, dtype=torch.float)
        trans = data['trans'].to(self.device, dtype=torch.float)
        quat = data['quat'].to(self.device, dtype=torch.float)

        pred_out = self.model(img, mask, scale, trans, quat)

        return pred_out

    def _convert_to_int_indices(self, float_inds):

        return (255 * (float_inds + 1) // 2).to(torch.int32)

    def _find_target_kps(self, src_kp, src_uv, tar_uv, tar_mask):

        batch_size = src_uv.size(0)
        height = src_uv.size(2)
        width = src_uv.size(3)

        src_kp_uv = torch.nn.functional.grid_sample(src_uv, src_kp[:, :, :2].view(batch_size, -1, 1, 2))
        src_kp_uv = src_kp_uv.permute(0, 2, 1, 3).squeeze()

        src_uv_3d = self.model.uv_to_3d(src_kp_uv.reshape(-1, 2)).view(batch_size, -1, 1, 3)
        tar_uv_3d = self.model.uv_to_3d(tar_uv.reshape(-1, 2)).view(batch_size, 1, -1, 3)

        tar_uv_3d = tar_uv_3d.repeat(1, src_uv_3d.size(1), 1, 1)
        src_uv_3d = src_uv_3d.repeat(1, 1, tar_uv_3d.size(2), 1)

        kp_dist = torch.pow((tar_uv_3d - src_uv_3d), 2)
        kp_dist = torch.sum(kp_dist, dim=-1)
        kp_dist = kp_dist + (1 - tar_mask.view(batch_size, 1, -1)) * torch.max(kp_dist) * 10

        tar_kp = unravel_index(torch.argmin(kp_dist, 2), (1, height, width, 1))
        tar_kp = tar_kp.permute(0, 2, 1)[:, :, 1:3]

        return tar_kp

    def _test_start_call(self):

        return

    def _test_end_call(self):

        return

    def _load_dataset(self):

        if self.data_cfg.category == 'car':
            dataset = P3DDataset(self.data_cfg, self.device)
        elif self.data_cfg.category == 'bird':
            dataset = CubDataset(self.data_cfg, self.device)
        else:
            dataset = ImnetDataset(self.data_cfg, self.device)

        return KPDataset(dataset, self.data_cfg.num_pairs)

    def _get_model(self) -> CSM:

        model = CSM(self.dataset.template_mesh,
                    self.dataset.mean_shape,
                    self.config.use_gt_cam,
                    self.device).to(self.device)

        return model

    def _add_kp_summaries(self, src_kp, tar_kp, tar_pred_kp, src_img, tar_img, step):

        self.summary_writer.add_images('src', draw_key_points(src_img, src_kp, self.key_point_colors), step)
        self.summary_writer.add_images('tar/orig', draw_key_points(tar_img, tar_kp, self.key_point_colors), step)
        self.summary_writer.add_images('tar/pred', draw_key_points(tar_img, tar_pred_kp, self.key_point_colors), step)

    def _add_uv_summaries(self, src_pred_out, tar_pred_out, src, tar, step):

        src_img = src['img'].to(self.device, dtype=torch.float)
        src_mask = src['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        src_uv = src_pred_out['uv']
        uv_color, uv_blend = sample_uv_contour(src_img, src_uv.permute(0, 2, 3, 1), self.dataset.texture_map, src_mask)
        self.summary_writer.add_images('src/uv_blend', uv_blend, step)
        self.summary_writer.add_images('src/uv', uv_color * src_mask, step)

        tar_img = tar['img'].to(self.device, dtype=torch.float)
        tar_mask = tar['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        tar_uv = tar_pred_out['uv']
        uv_color, uv_blend = sample_uv_contour(tar_img, tar_uv.permute(0, 2, 3, 1), self.dataset.texture_map, tar_mask)
        self.summary_writer.add_images('tar/uv_blend', uv_blend, step)
        self.summary_writer.add_images('tar/uv', uv_color * tar_mask, step)
