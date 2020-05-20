import os.path as osp

import numpy as np
import torch
from absl import app, flags

from src.data.cub_dataset import CubDataset

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, 'cachedir')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
flags.DEFINE_string('result_dir', osp.join(cache_path, 'results'),
                    'Directory where intermittent renderings are saved')
flags.DEFINE_string('dataset', 'cub', 'cub or imnet or p3d')
flags.DEFINE_integer('seed', 0, 'seed for randomness')


class Dataloder(CubDataset):
    def __init__(self, config):
        super(Dataloder, self).__init__(config)
        # this init is for the expanding the dataset choice (p3d, imnet)
        return


FLAGS = flags.FLAGS


def main(_):
    
    val_dataset = dataset_deform.DeformDataset(
        
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, shuffle=opt.shuffle, batch_size=opt.batch_size, num_workers=num_val_workers, collate_fn=collate_with_padding, pin_memory=True
    )
    
    for i, data in enumerate(train_dataloader):
        print(data)
        
    exit()
        

    seed = FLAGS.seed
    np.random.seed(seed)

    FLAGS.img_height = FLAGS.img_size
    FLAGS.img_width = FLAGS.img_size
    FLAGS.cache_dir = cache_path
    dataloader = Dataloder(FLAGS)

    index = 0
    # index = list(range(2))
    img_data = dataloader.get_img_data(index)
    for k, i in img_data[index].items():
        if type(i) == np.ndarray or type(i) == torch.Tensor:
            print(k)
            print(i.shape)
        else:
            print(k)
            print(type(i))
    print('...')
    d3_data = dataloader.get_3d_data()
    for k, i in d3_data.items():
        if type(i) == np.ndarray or type(i) == torch.Tensor:
            print(k)
            print(i.shape)
        else:
            print(k)
            print(type(i))


if __name__ == '__main__':
    app.run(main)