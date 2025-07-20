import importlib

import yaml


def parse_yaml_config(config_path: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        raise IOError(f"Configuration file {config_path} does not exist")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file {config_path}: {e}")


def import_class_from_string(class_string: str):
    module_name, class_name = class_string.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name), class_name
    except ImportError as e:
        raise ImportError(f"Could not import {class_string}. Reason: {e}")
    except AttributeError:
        raise AttributeError(f"Class {class_name} not found in module {module_name}.")
