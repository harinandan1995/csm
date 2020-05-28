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
                    choices=['train', 'test'])

parser.add_argument('-c', '--config',
                    help='Path to the config file. Default is config/train.yml.',
                    default='config/train.yml')

parser.add_argument('-d', '--device',
                    help='Device to be used by pytorch',
                    default='cuda:0')

args = parser.parse_args()

if __name__ == '__main__':

    print('Device: %s:%s' % (
        torch.cuda.get_device_name(),
        torch.cuda.current_device()))

    print(args)

    if args.mode == 'train':
        print('Starting the training........')
        start_train(args.config, args.device)
    else:
        print('Starting the testing........')
        start_test(args.config)
