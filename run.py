import argparse

import torch.utils.data

from src.scripts.test import start_test
from src.scripts.train import start_train

from src.data.cub_dataset import CubDataset
from src.utils.config import ConfigParser
import numpy as np
from src.nnutils.color_transform import sample_UV_contour
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode',
                    help='Allowed values: train, test.'
                         ' train: start training the model.'
                         ' test: test the model.'
                         ' Default value is train',
                    default='train',
                    choices=['train', 'test'],
                    nargs=1)

parser.add_argument('-c', '--config',
                    help='Path to the config file. Default is config/train.yml.',
                    default='config/train.yml',
                    nargs=1)

args = parser.parse_args()

if __name__ == '__main__':

    config = ConfigParser('./config/train.yml', None).config
    cf_dataset = config.dataset
    if cf_dataset.dataloader.dataset == 'cub':
        dataset = CubDataset(cf_dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cf_dataset.dataloader.batch_size, shuffle=True)
        uv = Image.open("resources/color_maps/bird/map3.png").resize((256, 256), Image.ANTIALIAS)
        uv = np.asarray(uv).transpose((2, 0, 1))
        uv_img = torch.Tensor(uv)
        for i, data in enumerate(dataloader):
            img = data['img'][0]
            uv_map = torch.Tensor(np.random.random((256, 256, 2)))
            mask = data['mask'][0]
            uv_rendering = sample_UV_contour(img, uv_map, uv_img, mask)
            uv_rendering = uv_rendering.numpy()
            uv_rendering = uv_rendering.transpose(1, 2, 0)
            uv_rendering = uv_rendering.astype(int)
            plt.imshow(uv_rendering)
            plt.show()
            break

    print('Device: %s:%s' % (
        torch.cuda.get_device_name(),
        torch.cuda.current_device()))

    if args.mode == 'train':
        print('Starting the training........')
        start_train(args.config)
    else:
        print('Starting the testing........')
        start_test(args.config)
