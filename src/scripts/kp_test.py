import json

from src.estimators.kp_tester import KPTransferTester
from src.utils.config import ConfigParser

def start_test(config_path, params, device):

    config = ConfigParser(config_path, params).config
    print(json.dumps(config, indent=3))

    return


if __name__ == '__main__':

    start_test('config/test.yml')