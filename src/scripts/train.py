# TODO: Once the trainer is ready this script will be used to start the training
from src.utils.config import ConfigParser
from src.estimators.csm_trainer import CSMTrainer


def start_train(config_path):

    config = ConfigParser(config_path, None).config
    trainer = CSMTrainer(None, config)

    trainer.train()


if __name__ == '__main__':

    start_train('config/train.yml')
