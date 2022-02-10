import os
import yaml
import numpy as np
import threading
from copy import deepcopy

def loadconfig(path):
    """
    :param path: path to a configuration file
    :return: configurations as a dictionary
    """
    with open(path) as f:
        try:
            return yaml.load(stream=f, Loader=yaml.FullLoader)
        except IOError as e:
            sys.exit("FAILED TO LOAD CONFIG {}: {}".format(path,e))


def loaddefaultconfig():
    """
    :return: configurations as a dictionary
    """
    path = "{}/configs/default.yaml".format(os.path.dirname(__file__))
    return loadconfig(path=path)

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y.copy())  # modifies z with y's keys and values & returns None
    return z

def merge_dicts(dicts):
    d = dicts[0]
    for i in range(1, len(dicts)):
        d = merge_two_dicts(d, dicts[i])
    return d
