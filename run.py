import argparse
import warnings

import torch.utils.data

from src.scripts.kp_test import start_test
from src.scripts.train import start_train
from src.utils.utils import add_train_arguments, add_kp_test_arguments

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    help='Path to the config file. Default is config/bird_train.yml.',
                    default='config/bird_train.yml', required=False)

parser.add_argument('-d', '--device',
                    help='Device to be used by pytorch',
                    default='cuda:0', required=False)

parser.add_argument('-sw', '--show_warnings',
                    help='Toggle this command if you want to show warnings',
                    action='store_true')

sub_parsers = parser.add_subparsers(help='Train or Test', dest='mode')

train_parser = sub_parsers.add_parser('train', help='Use this to start training a model')
add_train_arguments(train_parser)

test_parser = sub_parsers.add_parser('kp_test', help='Use this to start testing the model')
add_kp_test_arguments(test_parser)

args = parser.parse_args()

if not args.show_warnings:
    warnings.filterwarnings('ignore')

if __name__ == '__main__':

    print('Device: %s:%s' % (
        torch.cuda.get_device_name(),
        torch.cuda.current_device()))

    if args.mode == 'train':
        print('Starting the training........')
        start_train(args.config, args.__dict__, args.device)
    elif args.mode == 'kp_test':
        print('Starting the key point transfer testing........')
        start_test(args.config, args.__dict__, args.device)