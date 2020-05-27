import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data.cub_dataset import CubDataset
from src.nnutils.rendering import DepthRenderer
from src.utils.config import ConfigParser
from src.nnutils.geometry import get_scaled_orthographic_projection

if __name__ == '__main__':

    device = 'cuda:0'

    config = ConfigParser('config/train.yml', None).config
    renderer = DepthRenderer(device)

    dataset = CubDataset(config.dataset)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             shuffle=config.train.shuffle,
                             num_workers=config.train.workers)
    template_mesh = dataset.template_mesh

    for i, data in enumerate(data_loader):

        img = data['img'].to(device, dtype=torch.float)
        mask = data['mask'].unsqueeze(1).to(device, dtype=torch.float)
        scale = data['scale'].to(device, dtype=torch.float)
        trans = data['trans'].to(device, dtype=torch.float)
        quat = data['quat'].to(device, dtype=torch.float)

        rotation, translation = get_scaled_orthographic_projection(
            scale, trans, quat, device)

        pred_mask, depth = renderer(template_mesh, rotation, translation)
        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img[0].permute(1, 2, 0).cpu())
        plt.subplot(2, 2, 2)
        plt.imshow(mask[0][0].cpu(), cmap='gray', vmin=0, vmax=1)
        plt.subplot(2, 2, 3)
        plt.imshow(torch.flip(depth[0].cpu(), [0, 1]), cmap='gray', vmin=0, vmax=6)
        plt.subplot(2, 2, 4)
        plt.imshow(pred_mask[0].cpu(), cmap='gray', vmin=0, vmax=1)

        plt.show()

        if i == 1:
            break
