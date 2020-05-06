from torch.utils.data import Dataset


class IDataset(Dataset):

    """
    Interface for the dataset
    """

    def __init__(self, config, transforms):

        self.transforms = transforms
        self.data = self.get_data()
        self.num_samples = config.num_samples

    def __len__(self):

        return self.num_samples

    def __getitem__(self, index):

        # TODO: Write the code to process the data to generate dictionary elements like in base.py from original repo
        #  and perform transformations provided via self.transforms

        return None

    def get_data(self):

        """
        Child class must implement this method, data should be a list of dicts with each dict containing the following elements
        img_path,
        mask,
        bbox.x1,
        bbox.y1,
        bbox.x2,
        bbox.y2,
        parts,
        scale,
        trans,
        rot

        :return: list of data in expected format
        """

        return NotImplementedError('get_items method should be implemented in the child class')

    # TODO: Write the augmentation functions if necessary, like vertical & horizontal flips, contrast adjustments etc.
    #  check https://pytorch.org/docs/stable/torchvision/transforms.html for transformations

    # Space to implement common functions that can be used across multiple datasets
