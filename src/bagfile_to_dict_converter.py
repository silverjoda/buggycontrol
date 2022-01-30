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
import pickle

class BagfileConverter:
    def __init__(self):
        self.dataset_path = self.create_dataset_path()
        self.dataset_dict = {}

        self.init_ros()

    def create_dataset_path(self):
        # Save dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/dataset")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Find last indexed dataset
        for i in range(100):
            file_path = os.path.join(dataset_dir, "dataset_{}.pkl".format(i))
            if not os.path.exists(file_path):
                return os.path.join(dataset_dir, "dataset_{}.pkl".format(i))

    def save_dataset(self, data_dict_list, file_path):
        pickle.dump(data_dict_list, open(file_path, "wb"))

    def init_ros(self):
        rospy.init_node("bagfile_converter")

        self.tfBuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfBuffer)

        self.wheel_speed_sub = subscriber_factory("/wheel_speed", Float64)
        self.dv_sub = subscriber_factory("/imu/dv", Vector3Stamped)
        self.quat_sub = subscriber_factory("/filter/quaternion", QuaternionStamped)

        self.subscriber_list = []
        self.subscriber_list.append(self.wheel_speed_sub)
        self.subscriber_list.append(self.dv_sub)
        self.subscriber_list.append(self.quat_sub)

        self.ros_rate = rospy.Rate(200)
        time.sleep(0.5)

    def gather(self):
        # Wait until all subscribers have a message to begin
        while not rospy.is_shutdown():
            if np.all([s.get_msg() is not None for s in self.subscriber_list]): break

        # Do the gathering
        while not rospy.is_shutdown():
            # Get messages from all subscribers
            for s in self.subscriber_list:
                self.dataset_dict[s.topic_name] = s.get_msg(copy_msg=True)

            # Maintain that 200hz
            self.ros_rate.sleep()

        print("Saving dataset")
        self.save_dataset(self.dataset_dict, self.dataset_path)

if __name__=="__main__":
    bagfile_converter = BagfileConverter()
    bagfile_converter.gather()