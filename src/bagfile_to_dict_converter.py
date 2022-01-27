#!/usr/bin/env python
import rospy
import tf2_ros
from tf.transformations import *
from geometry_msgs.msg import Vector3Stamped, QuaternionStamped, TransformStamped, Quaternion, Vector3
from std_msgs.msg import Float64
from utils_python2 import *
import numpy as np
import threading
import time

class BagfileConverter:
    def __init__(self):
        self.dataset_path = ""

    def init_ros(self):
        rospy.init_node("bagfile_converter")

        self.tfBuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfBuffer)

        self.wheel_speed_sub = subscriber_factory("/wheel_speed", Float64)
        self.dv_sub = subscriber_factory("/imu/dv", Vector3Stamped)
        self.quat_sub = subscriber_factory("/filter/quaternion", QuaternionStamped)

        self.ros_rate = rospy.Rate(200)
        time.sleep(0.5)

    def gather(self):
        pass
