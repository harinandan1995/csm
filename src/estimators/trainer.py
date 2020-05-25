import os.path as osp

import torch

from torch.utils.tensorboard import SummaryWriter
from src.utils.utils import get_date, get_time


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
        out_dir: Path to the directory where the summaries and the checkpoints should be stored
            the summaries are stored in out_dir/summaries/{date}/{timestamp}/.
        """

        self.config = config

        self.model = self._get_model()
        self._load_model(config.checkpoint)

        self.data_loader = self._get_data_loader()
        self.optimizer = self._get_optimizer(config)

        self.summary_dir = osp.join(self.config.out_dir, 'summaries', get_date(), get_time())
        self.summary_writer = SummaryWriter(self.summary_dir)

    def train(self):
        """
        Call this function to start training the model for the given number of epochs
        """

        for epoch in range(self.config.epochs):

            running_loss = 0

            self._epoch_start_call(epoch, self.config.epochs)

            for step, batch in enumerate(self.data_loader):
                
                self._batch_start_call(batch, step, len(self.data_loader), epoch, self.config.epochs)
                
                loss = self._train_step(step, batch, epoch)
                running_loss += loss.item()
                
                self._batch_end_call(batch, loss, step, len(self.data_loader), 
                                     epoch, self.config.epochs)
            
            epoch_loss = running_loss / len(self.data_loader)
            self.summary_writer.add_scalar('Loss/train', epoch_loss, epoch)
            
            self._epoch_end_call(epoch, self.config.epochs)            

    def _save_model(self, path):
        """
        Saves the model at the path provided

        :param path: Path where the model weights should be stored
        """

        if path is not None and path != '':
            torch.save(self.model.state_dict(), path)
            print('Saving model at %s' % path)

    def _load_model(self, path):
        """
        Loads the model from the given path

        :param path: Path to the checkpoint file to preload the weights before the training starts
        """

        if path is not None and path != '' and osp.exists(path):
            self.model.load_state_dict(torch.load(path))
            print('Loaded model from %s' % path)

    def _train_step(self, step, batch, epoch):
        """
        Optimization step for each batch of data
        Calculating the loss and perform gradient optimization for the batch

        :param batch: Batch data from the dataloader
        :return: The loss for the batch
        """

        self.model.zero_grad()
        loss = self._calculate_loss(step, batch, epoch)
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

    def _get_data_loader(self):
        """
        Must be implemented by the child class.
        Should return a torch.utils.data.DataLoader which contains the data as a dictionary
        """

        return NotImplementedError

    def _get_model(self):
        """
        Must be implemented by the child class
        Should return a torch model which will be optimized during the training
        """

        return NotImplementedError

    def _calculate_loss(self, step, batch, epoch):
        """
        Must be implemented by the child class.

        :param batch: Batch data from the dataloader
        :return: The loss calculated for the batch as a torch.loss
        """

        return NotImplementedError

    def _batch_start_call(self, batch, step, total_steps, epoch, total_epochs):
        """
        This function will be called before each optimizin the model for the batch

        :param batch: Batch data
        :param step: Current batch number
        :type step: Int
        :param total_steps: Total number of batches
        :type total_steps: Int
        :param epoch: Current epoch
        :type epoch: Int
        :param total_epochs: Total number of epochs
        :type total_epochs: Int
        """

        return
    
    def _batch_end_call(self, batch, loss, step, total_steps, epoch, total_epochs):
        """
        This function will be called after each batch has been used for optimization
        Use this function to print and/or load any metrics after each 
        batch has been processed

        :param batch: Batch data
        :param loss: Loss calcuated for the batch
        :param type: FloatTensor
        :param step: Current batch number
        :type step: Int
        :param total_steps: Total number of batches
        :type total_steps: Int
        :param epoch: Current epoch
        :type epoch: Int
        :param total_epochs: Total number of epochs
        :type total_epochs: Int
        """

        return
    
    def _epoch_start_call(self, epoch, total_epochs):
        """
        This function is called at the begining of the each epoch
        Use this function to do anything before the start of each epoch

        :param epoch: Current epoch
        :type epoch: Int
        :param total_epochs: Total epochs
        :type total_epochs: Int
        """
        
        return

    def _epoch_end_call(self, epoch, total_epochs):
        """
        This function is called at the end of the each epoch
        Use this function to do anything at the end of each epoch 
        Eg.
        - write any summaries to the summary writer (self.summary_writer)
        - save model checkpoints

        :param epoch: Current epoch
        :type epoch: Int
        :param total_epochs: Total epochs
        :type total_epochs: Int
        """
        
        return
