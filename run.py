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
    cf_dataset = config.dataset
    if cf_dataset.dataloader.dataset == 'cub':
        dataset = CubDataset(cf_dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cf_dataset.dataloader.batch_size, shuffle=True)

        for i, data in enumerate(dataloader):
            for j in data:
                print(j)
                print(data[j].shape)
            break

        mean_shape = dataset.d3_data
        for i in mean_shape:
            print(i)
            print(type(mean_shape[i]))

    if args.mode == 'train':
        print('Starting the training........')
        # start_train(args.config)
    else:
        print('Starting the testing........')
        # start_test(args.config)
