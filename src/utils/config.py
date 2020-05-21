import yaml
import os


class ConfigParser:

    """
    A parser to parse the config files. The parameters can be accessed as attributes
    Eg. For the below config.yml file
    batch_size: 8
    epochs: 100
    optim:
      lr: 0.0004
      type: 'adam'

    learning rate can be accessed as config.optim.lr which is not possible
    with a standard dictionary.
    Representations such as 1e-4 are considered as strings
    """

    class ConfigObject(dict):
        """
        Represents configuration options' group, works like a dict
        """

        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)

        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, val):
            self[name] = val

    def __init__(self, config_file, schema=None):

        """
        :param config_file: Path to the config file
        :param schema: Optional. Path to the schema file
        """

        self.config_file = config_file
        self.schema = schema

        self._check_files()

        with open(self.config_file) as file:
            self.config_dict = yaml.load(file, Loader=yaml.FullLoader)

        self.config = self._create_config_object(self.config_dict)

    def _create_config_object(self, config_dict):

        if isinstance(config_dict, dict):
            out = self.ConfigObject()
            for key in config_dict:
                out[key] = self._create_config_object(config_dict[key])
            return out
        else:
            return config_dict

    def _check_files(self):

        """
        Checks if the config and schema files exist
        """

        if self.config_file is not None and not os.path.exists(self.config_file):
            return FileNotFoundError('%s file does not exist' % self.config_file)

        if self.schema is not None and not os.path.exists(self.schema):
            return FileNotFoundError('%s file does not exist' % self.schema)


if __name__ == '__main__':

    config = ConfigParser('config/train.yml', 'config/train.yml').config
    print(config.optim)
    print(config.optim.lr + 1.0)
    print(config.batch_size)
