from src.utils.config import ConfigParser
from src.estimators.csm_trainer import CSMTrainer


def start_train(config_path, device):
    config = ConfigParser(config_path, None).config
    trainer = CSMTrainer(config, device)

    trainer.train()


if __name__ == '__main__':
    start_train('config/train.yml')
