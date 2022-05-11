import math as m
import os
import sys

import numpy as np
import yaml


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

def q2e(w, x, y, z):
    pitch = -m.asin(2.0 * (x * z - w * y))
    roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return (roll, pitch, yaw)

def e2q(roll, pitch, yaw):
    qx = m.sin(roll / 2) * m.cos(pitch / 2) * m.cos(yaw / 2) - m.cos(roll / 2) * m.sin(pitch / 2) * m.sin(
        yaw / 2)
    qy = m.cos(roll / 2) * m.sin(pitch / 2) * m.cos(yaw / 2) + m.sin(roll / 2) * m.cos(pitch / 2) * m.sin(
        yaw / 2)
    qz = m.cos(roll / 2) * m.cos(pitch / 2) * m.sin(yaw / 2) - m.sin(roll / 2) * m.sin(pitch / 2) * m.cos(
        yaw / 2)
    qw = m.cos(roll / 2) * m.cos(pitch / 2) * m.cos(yaw / 2) + m.sin(roll / 2) * m.sin(pitch / 2) * m.sin(
        yaw / 2)
    return (qw, qx, qy, qz)

def theta_to_quat(theta):
    qx = 0
    qy = 0
    qz = m.sin(theta / 2)
    qw = m.cos(theta / 2)
    return [qw, qx, qy, qz]

def dist_between_wps(wp_1, wp_2):
    return np.sqrt(np.square(wp_1[0] - wp_2[0]) + np.square(wp_1[1] - wp_2[1]))
