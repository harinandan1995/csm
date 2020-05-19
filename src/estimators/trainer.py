import os

import torch

from torch.utils.tensorboard import SummaryWriter


class ITrainer:
    """
    An interface for any trainer which performs training using the given model and data
    Any trainers that inherits this class should implement the get_model, get_dataloader
    and calculate_loss functions for the trainer to work.
    """

    # TODO: Visualizations
    # TODO: Saving checkpoints
    # TODO: More metrics

    def __init__(self, config):

        """
        :param config: A dictionary containing the following parameters.

        epochs: Number of epochs for the training
        checkpoint: Path to a checkpoint to pre load a model. None if no weights are to be loaded.
        optim.type: Type of the optimizer to the used during the training. Allowed values are 'adam' and 'sgd'
        optim.lr: Learning rate for the optimizer
        optim.beta1: Beta1 value for the optimizer
        summary_dir: Path to the directory where the summaries should be stored
        """

        self.config = config

        self.model = self.get_model()
        self._load_model(config.checkpoint)

        self.data_loader = self.get_data_loader()
        self.optimizer = self._get_optimizer(config)

        self.summary_writer = self._get_summary_writer()

    def train(self):

        """
        Call this function to start the training
        """

        for epoch in range(self.config.epochs):

            for step, batch in enumerate(self.data_loader):
                
                self._train_step(step, batch)

    def _save_model(self, path):

        """
        Saves the model at the path provided

        :param path: Path where the model weights should be stored
        """

        if path is not None:
            torch.save(self.model.state_dict(), path)
            print('Saving model at %s' % path)

    def _load_model(self, path):

        """
        Loads the model from the given path

        :param path: Path to the checkpoint file to preload the weights before the training starts
        """

        if path is not None and os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print('Loaded model from %s' % path)

    def _train_step(self, step, batch):

        """
        Optimization step for each batch of data
        Calculating the loss and perform gradient optimization for the batch

        :param batch: Batch data from the dataloader
        :return: The loss for the batch
        """

        self.model.zero_grad()
        loss = self.calculate_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss

    def _get_optimizer(self, config):

        """
        Returns the optimizer to be used for the training

        :param config: A dict containing the following parameters. optim.type, optim.lr, optim.beta1
        :return: The optimizer defined as per the config parameters
        """

        if config.optim.type == 'SGD':

            return torch.optim.SGD(self.model.parameters(), lr=config.optim.lr, momentum=config.optim.beta1)

        elif config.optim.type == 'adam':

            return torch.optim.Adam(self.model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999))

        return AttributeError('Invalid optimizer type %s' % config.optim.type)

    def get_data_loader(self):

        """
        Must be implemented by the child class.
        Should return a torch.utils.data.DataLoader which contains the data as a dictionary
        """

        return NotImplementedError

    def get_model(self):

        """
        Must be implemented by the child class
        Should return a torch model which will be optimized during the training
        """

        return NotImplementedError

    def calculate_loss(self, batch):

        """
        Must be implemented by the child class.

        :param batch: Batch data from the dataloader
        :return: The loss calculated for the batch as a torch.loss
        """

        return NotImplementedError

    def _get_summary_writer(self):

        """
        Create a summary writer to write the summaries like losses, meshes etc.

        :return: torch.utils.tensorboard.SummaryWriter
        """

        return SummaryWriter(self.config.summary_dir)
