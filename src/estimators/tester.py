import os.path as osp

from tqdm import tqdm
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import ConfigParser
from src.utils.utils import get_time, get_date


class ITester:

    """
    An interface for all the testers
    """

    def __init__(self, config: ConfigParser.ConfigObject):

        self.config = config
        self.dataset = self._load_dataset()

        self.model = self._get_model()
        self._load_model(config.checkpoint)
        self.model.eval()

        self.data_loader = self._get_data_loader()

        time = get_time()
        date = get_date()
        self.out_dir = osp.join(self.config.out_dir, date, time)
        self.summary_dir = osp.join(self.out_dir, 'summaries')
        self.summary_writer = SummaryWriter(self.summary_dir)

    def test(self, **kwargs):
        """
        Call this function to star the testing
        :param kwargs: Use this to override any config values
        """

        self.config.update(kwargs)

        self._test_start_call()
        batch_bar = tqdm(self.data_loader)

        for step, batch_data in enumerate(batch_bar):

            batch_bar.set_description('Testing %sth batch' % step)
            stats = self._batch_call(step, batch_data)
            batch_bar.set_postfix(stats)

        self._test_end_call()

    def _load_model(self, path: str):
        """
        Loads the model from the given path

        :param path: Path to the checkpoint file to preload the weights before the training starts
        """

        if path is not None and path != '' and osp.exists(path):
            self.model.load_state_dict(torch.load(path))
            print('Loaded model weights from %s' % path)

    def _get_data_loader(self) -> torch.utils.data.DataLoader:
        """
        Creates a torch.utils.data.DataLoader from the dataset
        """

        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config.batch_size,
            shuffle=self.config.shuffle, num_workers=self.config.workers)

    def _test_start_call(self):
        """
        This function is called before the start of the testing
        """

        return

    def _test_end_call(self):
        """
        This function is called after the end of the testing
        :return:
        """

        return {}

    def _batch_call(self, step, batch_data):
        """
        This function is called for every batch. Child class must implement the testing logic
        like calling the model etc. here
        :param step: Current batch number
        :param batch_data: Current batch data
        """

        return

    def _load_dataset(self) -> torch.utils.data.Dataset:
        """
        Use this to load any datasets which might be needed to load key points, template meshes etc.
        """

        return NotImplementedError

    def _get_model(self) -> torch.nn.Module:
        """
        Must be implemented by the child class
        Should return a torch model which will be optimized during the training
        """

        return NotImplementedError
