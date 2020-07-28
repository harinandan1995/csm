import argparse
import os
import os.path as osp
from datetime import datetime

from tqdm import tqdm


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


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def str2bool(v):

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_train_arguments(sub_parser: argparse.ArgumentParser):

    sub_parser.add_argument('-b', '--train.batch_size', required=False, type=int)
    sub_parser.add_argument('-e', '--train.epochs', required=False, type=int)
    sub_parser.add_argument('-s', '--train.shuffle', required=False, type=str2bool)
    sub_parser.add_argument('-w', '--train.workers', required=False, type=int)
    sub_parser.add_argument('-ck', '--train.checkpoint', required=False, type=str)
    sub_parser.add_argument('--train.out_dir', required=False, type=str)

    sub_parser.add_argument('--train.use_gt_cam', required=False, type=str2bool)
    sub_parser.add_argument('--train.num_cam_poses', required=False, type=int)
    sub_parser.add_argument('--train.use_sampled_cam', required=False, type=str2bool)
    sub_parser.add_argument('--train.pose_warmup_epochs', required=False, type=int)

    sub_parser.add_argument('--train.loss.geometric', required=False, type=float)
    sub_parser.add_argument('--train.loss.visibility', required=False, type=float)
    sub_parser.add_argument('--train.loss.mask', required=False, type=float)
    sub_parser.add_argument('--train.loss.diverse', required=False, type=float)
    sub_parser.add_argument('--train.loss.quat', required=False, type=float)

    sub_parser.add_argument('-lr', '--train.optim.lr', required=False, type=float)
    sub_parser.add_argument('-b1', '--train.optim.beta1', required=False, type=float)


def add_kp_test_arguments(sub_parser: argparse.ArgumentParser):

    sub_parser.add_argument('-b', '--test.batch_size', required=False, type=int)
    sub_parser.add_argument('-s', '--test.shuffle', required=False, type=str2bool)
    sub_parser.add_argument('-w', '--test.workers', required=False, type=int)
    sub_parser.add_argument('--test.out_dir', required=False, type=str)
    sub_parser.add_argument('--test.use_gt_cam', required=False, type=str2bool)
    sub_parser.add_argument('--test.num_cam_poses', required=False, type=int)
    sub_parser.add_argument('--test.use_sampled_cam', required=False, type=str2bool)
    sub_parser.add_argument('-ck', '--test.checkpoint', required=False, type=str)
    sub_parser.add_argument('--test.alpha', required=False, type=float, nargs='+')
    sub_parser.add_argument('--test.add_summaries', required=False, type=str2bool)
    
    sub_parser.add_argument('--dataset.num_pairs', required=False, type=int)
