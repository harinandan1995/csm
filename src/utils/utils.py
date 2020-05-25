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
