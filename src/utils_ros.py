import os
import sys
import threading
from copy import deepcopy

import numpy as np
import rospy
import tf
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3


def subscriber_factory(topic_name, topic_type):
    class RosSubscriber:
        def __init__(self, topic_name=None):
            self.topic_name = topic_name
            self.lock = threading.Lock()
            self.cb = None
            self.msg = None
            self.ctr = 0
        def get_msg(self, copy_msg=False):
            with self.lock:
                if copy_msg:
                    return deepcopy(self.msg)
                return self.msg
        def get_ctr(self):
            with self.lock:
                return deepcopy(self.ctr)

    def msg_cb_wrapper(subscriber):
        def msg_cb(msg):
            with subscriber.lock:
                subscriber.msg = msg
                subscriber.ctr += 1
                #if subscriber.ctr % 10 == 0 and subscriber.topic_name=="/camera/odom/sample":
                #    print(msg)
                #    print(subscriber.ctr)
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

def get_static_tf(source_frame, target_frame):
    tfBuffer = tf2_ros.Buffer()
    tflistener = tf2_ros.TransformListener(tfBuffer)
    while True:
        try:
            trans = tfBuffer.lookup_transform(target_frame,
                                              source_frame,
                                              rospy.Time(0),
                                              rospy.Duration(0))
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn_throttle(1, "ros_utils tf lookup could not lookup tf: {}".format(err))
            continue
    return trans

def rotate_vector_by_quat(v, q):
    qm = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    new_v = np.matmul(qm, np.array([v.x, v.y, v.z]))
    return Vector3(x=new_v[0], y=new_v[1], z=new_v[2])

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
