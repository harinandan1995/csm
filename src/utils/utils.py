import os.path as osp


def validate_paths(*args):

    for arg in args:
        if not osp.exists(arg):
            raise FileNotFoundError('%s does not exist' % arg)

    return
