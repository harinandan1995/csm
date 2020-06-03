import argparse
import os
import os.path as osp

from datetime import datetime


def validate_paths(*args):
    for arg in args:
        if not osp.exists(arg):
            raise FileNotFoundError('%s does not exist' % arg)

    return


def create_dir_if_not_exists(path):

    os.makedirs(path, exist_ok=True)


def get_date():
    date_time = datetime.now()
    return date_time.strftime("%Y-%m-%d")


def get_time():
    date_time = datetime.now()
    return date_time.strftime("%H%M%S")


def add_train_arguments(sub_parser: argparse.ArgumentParser):

    sub_parser.add_argument('-b', '--train.batch_size', required=False, type=int)
    sub_parser.add_argument('-e', '--train.epochs', required=False, type=int)
    sub_parser.add_argument('-s', '--train.shuffle', required=False, type=bool)
    sub_parser.add_argument('-w', '--train.workers', required=False, type=int)
    sub_parser.add_argument('-ck', '--train.checkpoint', required=False, type=str)
    sub_parser.add_argument('--train.use_gt_cam', required=False, type=str)

    sub_parser.add_argument('--train.loss.geometric', required=False, type=float)
    sub_parser.add_argument('--train.loss.visibility', required=False, type=float)
    sub_parser.add_argument('--train.loss.mask', required=False, type=float)

    sub_parser.add_argument('-lr', '--train.optim.lr', required=False, type=float)
    sub_parser.add_argument('-b1', '--train.optim.beta1', required=False, type=float)


def add_test_arguments(sub_parser: argparse.ArgumentParser):

    return
