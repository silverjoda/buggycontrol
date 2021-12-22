import os
import yaml
import rospy
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
import numpy as np


def vectror3tonumpy(v: Vector3) -> np.ndarray:
    """
    :param v: 3d vector ros
    :return: numpy array shape (3,)
    """
    return np.array([v.x, v.y, v.z])


def invquaterniontonumpy(q: Quaternion) -> np.ndarray:
    """
    :param q: rotation as a ros quaternion
    :return: inversed quaternion x,y,z,-w as a numpy array
    """
    return np.array([q.x, q.y, q.z, -q.w])


def quaterniontonumpy(q: Quaternion) -> np.ndarray:
    """
    :param q: rotation as a ros quaternion
    :return: quaternion x,y,z,w as a numpy array
    """
    return np.array([q.x, q.y, q.z, q.w])


def numpytoquaternion(q: np.ndarray) -> Quaternion:
    """
    :param q: quaternion x,y,z,w as a numpy array
    :return: rotation as a ros quaternion
    """
    return Quaternion(*q)


def make_tf(frame: str, child: str, pos: np.ndarray, q: Quaternion) -> TransformStamped:
    """
    :param frame: frame id
    :param child: child frame id
    :param pos: x, y, z translation
    :param q: rotation represented by quaternion
    :return: transformation message
    """
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = frame
    t.child_frame_id = child
    t.transform.translation.x = pos[0]
    t.transform.translation.y = pos[1]
    t.transform.translation.z = pos[2]
    t.transform.rotation.x = q.x
    t.transform.rotation.y = q.y
    t.transform.rotation.z = q.z
    t.transform.rotation.w = q.w
    return t


def loadconfig(path: str) -> dict:
    """
    :param path: path to a configuration file
    :return: configurations as a dictionary
    """
    with open(path) as f:
        try:
            return yaml.load(stream=f, Loader=yaml.FullLoader)
        except IOError as e:
            sys.exit(f"FAILED TO LOAD CONFIG {path}: {e}")


def loaddefaultconfig() -> dict:
    """
    :return: configurations as a dictionary
    """
    path = f"{os.path.dirname(__file__)}/configs/default.yaml"
    return loadconfig(path=path)
