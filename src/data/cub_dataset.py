from src.data.dataset import IDataset


class CubDataset(IDataset):

    def __init__(self, config, transforms):

        super(CubDataset, self).__init__(config, transforms)

    def get_data(self):

        return None