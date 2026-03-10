import importlib
from .base_config import BaseConfig


def find_config_using_name(config_name):
    """Import the module "config/[config_name]_config.py".

    The class called <ConfigName>Config() will be instantiated.
    It has to be a subclass of BaseConfig, and it is case-insensitive.
    """
    config_filename = "config." + config_name + "_config"
    modellib = importlib.import_module(config_filename)
    config = None
    target_config_name = config_name.replace('_', '') + 'config'
    for name, cls in modellib.__dict__.items():
        if name.replace('_', '').lower() == target_config_name.lower() \
           and issubclass(cls, BaseConfig):
            config = cls

    if config is None:
        print("In %s.py, there should be a subclass of BaseConfig with class name that matches %s in lowercase." % (
            config_filename, target_config_name))
        exit(0)

    return config
