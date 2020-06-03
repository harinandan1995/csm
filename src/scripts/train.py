from src.estimators.csm_trainer import CSMTrainer
from src.utils.config import ConfigParser


def start_train(config_path, params, device):

    config = ConfigParser(config_path, params).config
    print(config)

    trainer = CSMTrainer(config, device)
    trainer.train()


if __name__ == '__main__':
    start_train('config/bird_train.yml')
