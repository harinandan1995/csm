import csv
import os.path as osp
import socket
import urllib.request

import scipy.io as sio

from src.data.utils.imnet import imnet_class2sysnet
from src.utils.utils import DownloadProgressBar, create_dir_if_not_exists

socket.setdefaulttimeout(3)


def load_csv(file):

    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        out = {}
        for row in reader:
            k, v = row
            out[k] = v
    
    return out


def download():

    path = 'datasets/cachedir/imnet/data'
    out_path = 'datasets/ImageNet/images'
    total = 0

    for id in imnet_class2sysnet.values():

        im_url_map = load_csv('resources/imnet/%s.csv' % id)
        success = 0
        fails = 0

        for split in ['val']:
            anno_path = osp.join(path, '%s_%s.mat' % (id, split))
            
            if osp.exists(anno_path):
                
                anno = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images']
                # print(split, id, len(anno))
                # total += len(anno)
                for data in anno:
                    
                    img_path = data.rel_path
                    img_id = img_path.replace('.JPEG', '')
                    
                    if img_id in im_url_map:
                        try:
                            create_dir_if_not_exists(osp.join(out_path, id))
                            write_image(im_url_map[img_id], osp.join(out_path, id, img_path))
                            success += 1
                        except Exception as e:
                            print(img_path, e)
                            fails += 1
                    else:
                        fails += 1
                        print(img_path)
                        
        print(success, fails, total)


def write_image(url, path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, path, reporthook=t.update_to)

if __name__ == '__main__':
    download()
