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
        self.uv_to_3d = UVto3D(self.dataset.mean_shape).to(self.device)
        self.kp_names = self.dataset.kp_names

        self.stats = {'kps1': [], 'kps2': [], 'transfer': [], 'kps_err': [], 'pair': [], }

        self.acc = []
        for alpha in self.config.alpha:
            self.acc.append(torch.zeros([3]))

        self.model1 = UNet(3, (4), 5).to(self.device)
        self.model2 = UNet(4, (3), 5).to(self.device)

        self._load_models()
    
    def _load_models(self):

        params1 = torch.load('/mnt/raid/csmteam/datasets/cachedir/snapshots/csm_bird_net/pred_net_200.pth')
        params2 = torch.load('/mnt/raid/csmteam/out/2020-07-23/112349/checkpoints/model_120401_10')
        
        for k, v in params1.items():
            if "unet_gen.model" in k:
                self.model1.state_dict()[k.replace('unet_gen.', '')].copy_(v)
        
        for k, v in params2.items():
            if "unet.model" in k:
                self.model2.state_dict()[k.replace('unet.', '')].copy_(v)

    def _batch_call(self, step, batch_data):

        batch, _ = batch_data

        uv1, _ = self._call_model1(batch)
        uv2, _ = self._call_model2(batch)

        self._add_uv_summaries(uv1, uv2, batch, batch, step)

        return {}

    def _call_model1(self, data):
        """
        Calls the model with proper data and returns the output
        :param data: Batch data dictionary
        :return: Output from the model
        """

        img = data['img'].to(self.device, dtype=torch.float)
        mask = data['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        
        unet_output = self.model1(img)
        uv_map  = unet_output[:, 0:3, :, :]
        uv_map = torch.tanh(uv_map) * (1 - 1E-6)
        uv_map = torch.nn.functional.normalize(uv_map, dim=1, eps=1E-6)
        uv_map = convert_3d_to_uv_coordinates(uv_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return uv_map, mask
    
    def _call_model2(self, data):
        """
        Calls the model with proper data and returns the output
        :param data: Batch data dictionary
        :return: Output from the model
        """

        img = data['img'].to(self.device, dtype=torch.float)
        mask = data['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        
        unet_output = self.model2(torch.cat([img, mask], 1))
        uv_map  = unet_output[:, 0:3, :, :]
        uv_map = torch.tanh(uv_map) * (1 - 1E-6)
        uv_map = torch.nn.functional.normalize(uv_map, dim=1, eps=1E-6)
        uv_map = convert_3d_to_uv_coordinates(uv_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return uv_map, mask

    @staticmethod
    def _convert_to_int_indices(float_indices):
        """
        Converts float indices [-1, 1] to int indices [0, 255]
        :param float_indices: (B X KP X 3) containing the float indices [-1, 1] and indicator value
        :return: A tensor containing the corresponding integer indices and the indicator value
        """

        float_indices[:, :, :2] = (255 * (float_indices[:, :, :2] + 1) // 2).to(torch.int32)

        return float_indices

    def _add_kp_summaries(self, src_kp, tar_kp, tar_pred_kp, src_img, tar_img, step, merge=True):
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

        src_kp_img = draw_key_points(src_img, src_kp, self.key_point_colors)
        tar_kp_img = draw_key_points(tar_img, tar_kp, self.key_point_colors)
        tar_pred_kp_img = draw_key_points(tar_img, tar_pred_kp, self.key_point_colors)

        if merge:
            self.summary_writer.add_images('merged/kp', torch.cat((src_kp_img, tar_pred_kp_img), dim=2), step)
            self.summary_writer.add_images('merged/tar_orig', tar_kp_img, step)
        else:
            self.summary_writer.add_images('src', src_kp_img, step)
            self.summary_writer.add_images('tar/orig', tar_kp_img, step)
            self.summary_writer.add_images('tar/pred', tar_pred_kp_img, step)

    def _add_uv_summaries(self, src_uv, tar_uv, src, tar, step, merge=True):

        src_img = src['img'].to(self.device, dtype=torch.float)
        src_mask = src['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        src_uv_color, src_uv_blend = sample_uv_contour(src_img, src_uv.permute(0, 2, 3, 1), self.dataset.texture_map, src_mask)

        tar_img = tar['img'].to(self.device, dtype=torch.float)
        tar_mask = tar['mask'].unsqueeze(1).to(self.device, dtype=torch.float)
        tar_uv_color, tar_uv_blend = sample_uv_contour(tar_img, tar_uv.permute(0, 2, 3, 1), self.dataset.texture_map, tar_mask)
        
        self.summary_writer.add_images('src/uv_blend', src_uv_blend, step)
        self.summary_writer.add_images('src/uv', src_uv_color, step)
        self.summary_writer.add_images('tar/uv_blend', tar_uv_blend, step)
        self.summary_writer.add_images('tar/uv', tar_uv_color, step)
    
    def _load_dataset(self) -> KPDataset:

        if self.data_cfg.category == 'car':
            dataset = P3DDataset(self.data_cfg, self.device)
        elif self.data_cfg.category == 'bird':
            dataset = CubDataset(self.data_cfg, self.device)
        else:
            dataset = ImnetDataset(self.data_cfg, self.device)

        return KPDataset(dataset, self.data_cfg.num_pairs)

    def _get_model(self) -> CSM:

        model = UNet(3, (4), 5).to(self.device)

        return model
