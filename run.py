import argparse

import torch.utils.data

from src.scripts.test import start_test
from src.scripts.train import start_train

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

    print('Device: %s:%s' % (
        torch.cuda.get_device_name(),
        torch.cuda.current_device()))

    if args.mode == 'train':
        print('Starting the training........')
        start_train(args.config)
    else:
        print('Starting the testing........')
        start_test(args.config)
