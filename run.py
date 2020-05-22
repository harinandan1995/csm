import argparse

import torch.utils.data

from src.data.cub_dataset import CubDataset
from src.scripts.test import start_test
from src.scripts.train import start_train
from src.utils.config import ConfigParser

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

    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())
    config = ConfigParser('./config/train.yml', None).config
    dataset = CubDataset(config.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    for data in dataloader:
        print(data)
        break

    if args.mode == 'train':
        print('Starting the training........')
        # start_train(args.config)
    else:
        print('Starting the testing........')
        # start_test(args.config)
