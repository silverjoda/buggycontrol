import os
import yaml
import rospy
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
import numpy as np
import threading
from copy import deepcopy

def subscriber_factory(topic_name, topic_type):
    class RosSubscriber:
        def __init__(self, topic_name=None):
            self.topic_name = topic_name
            self.lock = threading.Lock()
            self.cb = None
            self.msg = None
        def get_msg(self, copy_msg=False):
            with self.lock:
                if copy_msg:
                    return deepcopy(self.msg)
                return self.msg

    def msg_cb_wrapper(subscriber):
        def msg_cb(msg):
            with subscriber.lock:
                subscriber.msg = msg
        return msg_cb
    subscriber = RosSubscriber(topic_name)
    rospy.Subscriber(topic_name,
                     topic_type,
                     msg_cb_wrapper(subscriber), queue_size=1)
    return subscriber

def vector3tonumpy(v):
    """
    :param v: 3d vector ros
    :return: numpy array shape (3,)
    """
    return np.array([v.x, v.y, v.z])


def invquaterniontonumpy(q):
    """
    :param q: rotation as a ros quaternion
    :return: inversed quaternion x,y,z,-w as a numpy array
    """
    return np.array([q.x, q.y, q.z, -q.w])


def quaterniontonumpy(q):
    """
    :param q: rotation as a ros quaternion
    :return: quaternion x,y,z,w as a numpy array
    """
    return np.array([q.x, q.y, q.z, q.w])


def numpytoquaternion(q):
    """
    :param q: quaternion x,y,z,w as a numpy array
    :return: rotation as a ros quaternion
    """
    return Quaternion(*q)


def make_tf(frame, child, pos, q):
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
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t


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
