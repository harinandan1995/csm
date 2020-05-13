# TODO: Implement the training loops etc. assuming we have a model with expected output (Katam)
import os

import torch


class ITrainer:

    def __init__(self, config):

        self.config = config
        self.epochs = config.epochs

        self.model = self.get_model()
        self.load_model(config.checkpoint)

        self.data_loader = self.get_data_loader()
        self.optimizer = self.get_optimizer(config)

    def train(self):

        for epoch in range(self.epochs):

            for i, batch in enumerate(self.data_loader):

                self._train_step(batch)

    def save_model(self, path):

        if path is not None:

            torch.save(self.model.state_dict(), path)
            print('Saving model at %s' % path)

    def load_model(self, path):

        if path is not None and os.path.exists(path):

            self.model.load_state_dict(torch.load(path))
            print('Loaded model from %s' % path)

    def _train_step(self, batch):

        self.model.zero_grad()
        loss = self.calculate_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss

    def get_optimizer(self, config):

        if config.optim.type == 'SGD':

            return torch.optim.SGD(self.model.parameters(), lr=config.optim.lr, momentum=config.optim.beta1)

        elif config.optim.type == 'adam':

            return torch.optim.Adam(self.model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999))

        return AttributeError('Invalid optimizer type %s' % config.optim.type)

    def get_data_loader(self):

        return NotImplementedError

    def get_model(self):

        return NotImplementedError

    def calculate_loss(self, batch):

        return NotImplementedError
