import yaml
from box import ConfigBox


def load_params(params_path):
    """
    Function used for loading the parameters from a yaml file.
    """
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params
