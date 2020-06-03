import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data.cub_dataset import CubDataset
from src.nnutils.geometry import get_scaled_orthographic_projection
from src.nnutils.rendering import DepthRenderer, ColorRenderer
from src.utils.config import ConfigParser

if __name__ == '__main__':

    device = 'cuda:0'

    config = ConfigParser('../../config/bird_train.yml', None).config

    dataset = CubDataset(config.dataset)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0)
    template_mesh = dataset.template_mesh

    renderer = DepthRenderer(device, image_size=256)
    color_renderer = ColorRenderer(template_mesh, image_size=1024)

    for i, data in enumerate(data_loader):

        img = data['img'].to(device, dtype=torch.float)
        mask = data['mask'].unsqueeze(1).to(device, dtype=torch.float)
        scale = data['scale'].to(device, dtype=torch.float)
        trans = data['trans'].to(device, dtype=torch.float)
        quat = data['quat'].to(device, dtype=torch.float)

        rotation, translation = get_scaled_orthographic_projection(
            scale, trans, quat, device)

        pred_mask, depth = renderer(template_mesh, rotation, translation)
        depth[depth < 0] = depth.max() + 1
        # depth = (depth - depth.min())/(depth.max()-depth.min())
        # depth_norm = (depth - depth.mean(dim=1)) / depth.std()

        color_image = color_renderer(rotation, translation)

        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img[0].permute(1, 2, 0).cpu())
        plt.subplot(2, 2, 2)
        plt.imshow(color_image[0].cpu())
        plt.subplot(2, 2, 3)
        plt.imshow(torch.flip(depth[0].cpu(), [0, 1]), cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(torch.flip(pred_mask[0].cpu(), [0, 1]), cmap='gray', vmin=0, vmax=1)

        plt.show()

        if i == 0:
            break
