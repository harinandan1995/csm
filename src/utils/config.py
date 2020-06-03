import yaml
import os
import re


class ConfigParser:

    """
    A parser to parse the config files. The parameters can be accessed as attributes
    Eg. For the below config.yml file
    
    train:
        batch_size: 8
        epochs: 100
        optim:
            lr: 0.0004
            type: 'adam'

    config = ConfigParser('config/train.yml', None).config
    
    You can access the parameters form the config file as shown below
    config.train.optim
    config.train.optim.lr
    config.train.batch_size

    which is not possible with a standard python dictionary.
    
    Note: Values such as 1e-4 are considered as strings instead of float
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

    def __init__(self, config_file, args=None):

        """
        :param config_file: Path to the config file
        :param args: Optional. Arguments passed via command line
        """

        self.config_file = config_file
        self._check_files()

        self.pattern = re.compile(r"^([0-9a-zA-Z]+_*[0-9a-zA-Z]*.)*([0-9a-zA-Z]+_*[0-9a-zA-Z]*)+$")

        with open(self.config_file) as file:
            self.config_dict = yaml.load(file, Loader=yaml.FullLoader)

        self._update_with_args(args)

        self.config = self._create_config_object(self.config_dict)

    def _create_config_object(self, config_dict):

        if isinstance(config_dict, dict):
            out = self.ConfigObject()
            for key in config_dict:
                out[key] = self._create_config_object(config_dict[key])
            return out
        else:
            return config_dict

    def _update_dict_for_nested_key(self, dictionary, key: str, value):

        if "." not in key:
            dictionary[key] = value
        else:
            key_1, key_2 = key.split('.', 1)
            if key_1 not in dictionary:
                dictionary[key_1] = {}

            self._update_dict_for_nested_key(dictionary[key_1], key_2, value)

    def _update_with_args(self, args):

        if args is None:
            return

        for param, value in args.items():
            if value is None:
                continue
            if self.pattern.fullmatch(param):
                self._update_dict_for_nested_key(self.config_dict, param, value)

    def _check_files(self):
        """
        Checks if the config and schema files exist
        """

        if self.config_file is not None and not os.path.exists(self.config_file):
            return FileNotFoundError('%s file does not exist' % self.config_file)
