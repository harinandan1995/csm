import numpy as np
import torch
from pytorch3d.ops.cubify import unravel_index

from src.data.cub_dataset import CubDataset
from src.data.dataset import KPDataset
from src.data.imnet_dataset import ImnetDataset
from src.data.p3d_dataset import P3DDataset
from src.estimators.tester import ITester
from src.model.csm import CSM
from src.model.unet import UNet
from src.model.uv_to_3d import UVto3D
from src.nnutils.color_transform import draw_key_points, sample_uv_contour
from src.nnutils.metrics import calculate_correct_key_points
from src.nnutils.geometry import convert_3d_to_uv_coordinates
from src.utils.config import ConfigParser
from src.nnutils import pck


class KPTransferTester(ITester):

    def __init__(self, config: ConfigParser.ConfigObject, device):

        self.device = device
        self.data_cfg = config.dataset
        super(KPTransferTester, self).__init__(config.test)
        self.key_point_colors = np.random.uniform(0, 1, (len(self.dataset.kp_names), 3))
        self.num_kps = len(self.dataset.kp_names)
        self.kp_names = self.dataset.kp_names
        self.kp_uv =  torch.from_numpy(self.dataset.kp_uv).type(torch.float32).to(device)

        self.stats = {'kps1': [], 'kps2': [], 'transfer': [], 'kps_err': [], 'pair': [], }

        self.acc = []
        for alpha in self.config.alpha:
            self.acc.append(torch.zeros([3]))

    def _batch_call(self, step, batch_data):

        batch1, batch2 = batch_data

        transfer_kps12, error_kps12, transfer_kps21, error_kps21, kps1, kps2 = self._evaluate(batch1, batch2, step)

        kp_mask = kps1[:, 2:] * kps2[:, 2:]
        
        kp_12 = torch.cat((transfer_kps12, kp_mask), dim=1)
        kp_21 = torch.cat((transfer_kps21, kp_mask), dim=1)

        img1 = batch1['img'].to(self.device, dtype=torch.float)
        img2 = batch2['img'].to(self.device, dtype=torch.float)
        
        self._add_kp_summaries(kps1, kps2, kp_12, kp_21, img1, img2, step)

        self.stats['transfer'].append(self._to_numpy(transfer_kps12))
        self.stats['kps_err'].append(self._to_numpy(error_kps12))
        self.stats['kps1'].append(self._to_numpy(kps1))
        self.stats['kps2'].append(self._to_numpy(kps2))

        self.stats['transfer'].append(self._to_numpy(transfer_kps21))
        self.stats['kps_err'].append(self._to_numpy(error_kps21))
        self.stats['kps1'].append(self._to_numpy(kps2))
        self.stats['kps2'].append(self._to_numpy(kps1))

        return {}

    def _test_end_call(self):

        n_iter = len(self.dataset)

        self.stats['kps1'] = np.stack(self.stats['kps1'])
        self.stats['kps2'] = np.stack(self.stats['kps2'])
        self.stats['transfer'] = np.stack(self.stats['transfer'])
        self.stats['kps_err'] = np.stack(self.stats['kps_err'])

        dist_thresholds = [1e-4, 1e-3,0.25*1e-2, 0.5*1e-2, 0.75*1e-2, 1E-2, 1E-1, 0.2, 0.3, 0.4, 0.5, 0.6, 10]
        pck.run_evaluation(self.stats, n_iter, self.out_dir, self.data_cfg.img_size, self.kp_names, dist_thresholds)

    def _calculate_acc(self, src_kp, tar_kp, tar_pred_kp, height, width):

        out = {}
        for i, alpha in enumerate(self.config.alpha):
            self.acc[i] += calculate_correct_key_points(src_kp, tar_kp, tar_pred_kp, alpha * height, alpha * width)
            out['acc' + str(i)] = (self.acc[i][0] / self.acc[i][2]).item()
        
        return out
    
    def _call_model(self, data):
        """
        Calls the model with proper data and returns the output
        :param data: Batch data dictionary
        :return: Output from the model
        """

        img = data['img'].to(self.device, dtype=torch.float)
        mask = data['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        scale = data['scale'].to(self.device, dtype=torch.float)
        trans = data['trans'].to(self.device, dtype=torch.float)
        quat = data['quat'].to(self.device, dtype=torch.float)

        pred_out = self.model(img, mask, scale, trans, quat)

        return pred_out

    @staticmethod
    def _convert_to_int_indices(float_indices):
        """
        Converts float indices [-1, 1] to int indices [0, 255]
        :param float_indices: (B X KP X 3) containing the float indices [-1, 1] and indicator value
        :return: A tensor containing the corresponding integer indices and the indicator value
        """

        float_indices[:, :, :2] = (255 * (float_indices[:, :, :2] + 1) // 2).to(torch.int32)

        return float_indices

    def _evaluate(self, batch1, batch2, step):

        mask1 = batch1['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        mask2 = batch2['mask'].unsqueeze(1).to(self.device, dtype=torch.float)

        pred_out1 = self._call_model(batch1)
        pred_out2 = self._call_model(batch2)

        uv1 = pred_out1['uv']
        uv2 = pred_out2['uv']

        mesh1 = pred_out1["arti"]
        mesh2 = pred_out2["arti"]

        self._add_uv_summaries(pred_out1, pred_out2, batch1, batch2, step)

        kps1 = self._convert_to_int_indices(batch1['kp'].to(self.device, dtype=torch.float)).view(-1 , 3).long()
        kps2 = self._convert_to_int_indices(batch2['kp'].to(self.device, dtype=torch.float)).view(-1 , 3).long()

        transfer_kps12, error_kps12 = self.map_kp_img(kps1, uv1, mask1, mesh1)
        transfer_kps21, error_kps21 = self.map_kp_img(kps2, uv2, mask2, mesh2)
        
        return transfer_kps12, error_kps12, transfer_kps21, error_kps21, kps1, kps2

    def _to_numpy(self, tensor):

        return tensor.data.cpu().numpy()

    def map_kp_img(self, kps2, uv_map2, mask2, mesh2):

        kp_mask = kps2[:, 2]
        img_H = uv_map2.size(2)
        img_W = uv_map2.size(3)

        uv_map2 = uv_map2.reshape(-1, img_H, img_W).permute(1, 2, 0)

        kps1_3d = self.model.uv_to_3d(self.kp_uv, None).view(1, 1, -1 ,3)
        uv_points3d = self.model.uv_to_3d(uv_map2.reshape(-1, 2), None).view(1, img_H, img_W, 3)

        distances3d = torch.sum((kps1_3d.view(-1, 1, 3) - uv_points3d.view(1, -1, 3))**2, -1).sqrt()

        distances3d = distances3d + (1 - mask2.view(1, -1)) * 1000
        distances = distances3d
        min_dist, min_indices = torch.min(distances.view(len(self.kp_uv), -1), dim=1)
        min_dist = min_dist
        transfer_kps = torch.stack([min_indices % img_W, min_indices // img_W], dim=1)

        kp_transfer_error = torch.norm((transfer_kps.float() - kps2[:, 0:2]), dim=1)

        return transfer_kps, torch.stack([kp_transfer_error, kp_mask.float(), min_dist], dim=1)

    def _load_dataset(self) -> KPDataset:

        if self.data_cfg.category == 'car':
            dataset = P3DDataset(self.data_cfg, self.device)
        elif self.data_cfg.category == 'bird':
            dataset = CubDataset(self.data_cfg, self.device)
        else:
            dataset = ImnetDataset(self.data_cfg, self.device)

        return KPDataset(dataset, self.data_cfg.num_pairs)

    def _get_model(self) -> CSM:

        model = CSM(self.dataset.template_mesh, self.dataset.mean_shape, self.config.use_gt_cam,
                    self.config.num_cam_poses, self.config.use_sampled_cam, self.config.use_arti,
                    self.config.arti_epochs, self.dataset.arti_info_mesh, self.device, self.config.num_in_chans).to(self.device)

        return model

    def _add_kp_summaries(self, kps1, kps2, pred_kp12, pred_kp21, img1, img2, step, merge=True):
        """
        Add key point summaries to tensorboard

        :param src_kp: [B X KP X 2] indices of the key points on src image [0-255]
        :param tar_kp: [B X KP X 2] indices of the key points on target image [0-255]
        :param tar_pred_kp: [B X KP X 2] indices of the predicted key points corresponding
            to the src key points on target image [0-255]
        :param src_img: [B X 3 X H X W] source image
        :param tar_img: [B X 3 X H X W] target image
        :param step: Current batch number
        :param merge: True or False. True if you want to merge the src and tar outputs into one image
        :return:
        """
        if not self.config.add_summaries:
            return

        kps1 = kps1.unsqueeze(0)
        kps2 = kps2.unsqueeze(0)
        pred_kp12 = pred_kp12.unsqueeze(0)
        pred_kp21 = pred_kp21.unsqueeze(0)

        kp_img1 = draw_key_points(img1, kps1, self.key_point_colors)
        kp_img2 = draw_key_points(img2, kps2, self.key_point_colors)
        pred_kp_img1 = draw_key_points(img1, pred_kp21, self.key_point_colors)
        pred_kp_img2 = draw_key_points(img2, pred_kp12, self.key_point_colors)

        if merge:
            self.summary_writer.add_images('%d/merged/kp12' % step, torch.cat((kp_img1, pred_kp_img2), dim=2), step)
            self.summary_writer.add_images('%d/merged/kp21' % step, torch.cat((kp_img2, pred_kp_img1), dim=2), step)
            self.summary_writer.add_images('%d/merged/orig1'% step, kp_img1, step)
            self.summary_writer.add_images('%d/merged/orig2'% step, kp_img2, step)
        else:
            self.summary_writer.add_images('src', kp_img1, step)
            self.summary_writer.add_images('tar/orig', kp_img2, step)
            self.summary_writer.add_images('tar/pred', pred_kp_img2, step)

    def _add_uv_summaries(self, src_pred_out, tar_pred_out, src, tar, step, merge=True):
        
        if not self.config.add_summaries:
            return

        src_img = src['img'].to(self.device, dtype=torch.float)
        src_mask = src['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        src_uv = src_pred_out['uv']
        src_uv_color, src_uv_blend = sample_uv_contour(src_img, src_uv.permute(0, 2, 3, 1), self.dataset.texture_map, src_mask)

        tar_img = tar['img'].to(self.device, dtype=torch.float)
        tar_mask = tar['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        tar_uv = tar_pred_out['uv']
        tar_uv_color, tar_uv_blend = sample_uv_contour(tar_img, tar_uv.permute(0, 2, 3, 1), self.dataset.texture_map, tar_mask)

        if merge:
            self.summary_writer.add_images(
                '%d/merged/uv_blend' % step, torch.cat((src_uv_blend, tar_uv_blend), dim=2), step)
            self.summary_writer.add_images(
                '%d/merged/uv' % step, torch.cat((src_uv_color * src_mask, tar_uv_color * tar_mask), dim=2), step)
        else:
            self.summary_writer.add_images('src/uv_blend', src_uv_blend, step)
            self.summary_writer.add_images('src/uv', src_uv_color * src_mask, step)
            self.summary_writer.add_images('tar/uv_blend', tar_uv_blend, step)
            self.summary_writer.add_images('tar/uv', tar_uv_color * tar_mask, step)
