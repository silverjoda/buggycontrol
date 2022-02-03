#!/usr/bin/env python3
import rospy
import tf2_ros
from tf.transformations import *
from geometry_msgs.msg import Vector3Stamped, QuaternionStamped, TransformStamped, Quaternion, Vector3
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from buggycontrol.msg import Actions
from utils import *
import numpy as np
import threading
import time
import pickle

class BagfileConverter:
    def __init__(self):
        self.dataset_path = self.create_dataset_path()
        self.init_ros()

        self.dataset_dict_list = []

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

        self.gt_odometry_sub = subscriber_factory("/gt/base_link_odom", Odometry)
        self.actions_sub = subscriber_factory("/actions", Actions)

        self.ros_rate = rospy.Rate(200)
        time.sleep(0.3)

    def process_dataset(self):
        pass

    def gather(self):
        # Wait until all subscribers have a message to begin
        while not rospy.is_shutdown():
            if np.all([s.get_msg() is not None for s in [self.gt_odometry_sub, self.actions_sub]]): break

        # Do the gathering
        print("Started gathering")
        while not rospy.is_shutdown():
            # Get messages from all subscribers
            dataset_dict = {}
            act_msg = self.actions_sub.get_msg(copy_msg=True)
            odom_msg = self.gt_odometry_sub.get_msg(copy_msg=True)
            dataset_dict["act_msg"] = act_msg
            dataset_dict["odom_msg"] = odom_msg
            self.dataset_dict_list.append(dataset_dict)

            # Maintain that 200hz
            self.ros_rate.sleep()

        X, Y = self.process_dataset()

        print("Saving dataset")
        self.save_dataset(X, Y, self.dataset_path)

if __name__=="__main__":
    bagfile_converter = BagfileConverter()
    bagfile_converter.gather()