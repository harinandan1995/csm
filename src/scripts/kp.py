import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cub_dataset import CubDataset
from src.model.uv_to_3d import UVto3D
from src.nnutils.color_transform import draw_key_points
from src.nnutils.geometry import get_scaled_orthographic_projection, load_mean_shape
from src.utils.config import ConfigParser

if __name__ == '__main__':

    device = 'cuda:0'

    config = ConfigParser('config/bird_train.yml', None).config

    dataset = CubDataset(config.dataset)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=0)
    template_mesh = dataset.template_mesh
    mean_shape = load_mean_shape(
            osp.join(config.dataset.dir.cache_dir, 'uv', 'mean_shape.mat'), device=device)
    uv_to_3d = UVto3D(mean_shape, device)
    key_point_colors = np.random.uniform(0, 1, (len(dataset.kp_names), 3))

    for i, data in enumerate(data_loader):

        img = data['img'].to(device, dtype=torch.float)
        mask = data['mask'].unsqueeze(1).to(device, dtype=torch.float)
        scale = data['scale'].to(device, dtype=torch.float)
        trans = data['trans'].to(device, dtype=torch.float)
        quat = data['quat'].to(device, dtype=torch.float)
        kps = (((data['kp'].to(device, dtype=torch.float) + 1)/2) * 255).to(torch.int32)
        kp = draw_key_points(img, kps, key_point_colors)

        rotation, translation = get_scaled_orthographic_projection(
            scale, trans, quat, device)
        rotation = rotation.permute(0, 2, 1)

        kp_uv = data['kp_uv'].to(device, dtype=torch.float)
        uv_flatten = kp_uv.view(-1, 2)
        uv_3d = uv_to_3d(uv_flatten).view(1, -1, 3)

        xyz = torch.bmm(uv_3d, rotation.view(-1, 3, 3)) + translation.view(-1, 1, 3)
        xy = (((xyz[:, :, :2] + 1)/2) * 255).to(torch.int32)
        kp_xy = torch.cat((xy, kps[:, :, 2:]), dim=2)

        kp_pred = draw_key_points(img, kp_xy, key_point_colors)

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(kp[0].permute(1, 2, 0).cpu())
        plt.subplot(1, 2, 2)
        plt.imshow(kp_pred[0].permute(1, 2, 0).cpu())
        plt.show()

        if i == 0:
            break
