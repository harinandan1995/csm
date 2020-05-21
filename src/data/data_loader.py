import os.path as osp

import numpy as np
from torch.utils.data import DataLoader
from absl import app, flags

from src.data.cub_dataset import CubDataset

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, 'cachedir')
# flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
# flags.DEFINE_string('result_dir', osp.join(cache_path, 'results'),
#                     'Directory where intermittent renderings are saved')
# flags.DEFINE_string('dataset', 'cub', 'cub or imnet or p3d')
# flags.DEFINE_integer('seed', 0, 'seed for randomness')


class CSPTrainner():
    def __init__(self,config,shuffle):
        super(CSPTrainner, self).__init__()
        # this init is for the expanding the dataset choice (p3d, imnet)
        if config.dataset == 'cub':
            self.dataset = CubDataset(config)
            self.mean_shape = self.dataset.d3_data
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle = shuffle, num_workers= config.n_data_workers, pin_memory=True)
        return


FLAGS = flags.FLAGS
def main(_):
    seed = FLAGS.seed
    np.random.seed(seed)

    FLAGS.img_height = FLAGS.img_size
    FLAGS.img_width = FLAGS.img_size
    FLAGS.cache_dir = cache_path

    trainer = CSPTrainner(FLAGS,True)
    dataloader = trainer.dataloader
    for i, v in enumerate(dataloader):
        if i == 1:
            print(v['img'].shape)
            break
    mean_shape = trainer.mean_shape
    for i in mean_shape:
        print(i)
        print(type(mean_shape[i]))


if __name__ == '__main__':
    app.run(main)