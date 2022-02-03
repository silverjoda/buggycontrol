#!/usr/bin/env python3
import pickle
import time

import numpy as np
from buggycontrol.msg import Actions
from nav_msgs.msg import Odometry

from utils import *


class BagfileConverter:
    def __init__(self):
        self.xy_dataset_paths = self.create_dataset_path()
        self.init_ros()

    def create_dataset_path(self):
        # Save dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/dataset")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Find last indexed dataset
        for i in range(100):
            x_file_path = os.path.join(dataset_dir, "X_{}.pkl".format(i))
            y_file_path = os.path.join(dataset_dir, "Y_{}.pkl".format(i))
            if not os.path.exists(x_file_path):
                return x_file_path, y_file_path

    def save_dataset(self, X, Y, file_paths):
        pickle.dump(X, open(file_paths[0], "wb"))
        pickle.dump(Y, open(file_paths[1], "wb"))

    def init_ros(self):
        rospy.init_node("bagfile_converter")

        self.gt_odometry_sub = subscriber_factory("/gt/base_link_odom", Odometry)
        self.actions_sub = subscriber_factory("/actions", Actions)

        self.ros_rate = rospy.Rate(200)
        time.sleep(0.3)

    def get_delta(self, msg_cur, msg_next):
        return msg_next.header.stamp.secs - msg_cur.header.stamp.secs + \
                (msg_next.header.stamp.nsecs - msg_cur.header.stamp.nsecs) / 1000000000.

    def process_dataset(self, dataset_dict_list):
        dt = 0.005
        x_list = []
        y_list = []
        for i in range(len(dataset_dict_list) - 1):
            current_act_msg = dataset_dict_list[i]["act_msg"]
            current_odom_msg = dataset_dict_list[i]["odom_msg"]
            next_odom_msg = dataset_dict_list[i + 1]["odom_msg"]

            time_delta = self.get_delta(current_odom_msg, next_odom_msg)
            time_correction_factor = np.clip(dt / time_delta, -0.9, 1.1)

            # Calculate velocity delta from current to next odom
            current_lin_vel = current_odom_msg.twist.twist.linear
            next_lin_vel = next_odom_msg.twist.twist.linear

            lin_vel_diff_x = (next_lin_vel.x - current_lin_vel.x) * time_correction_factor
            lin_vel_diff_y = (next_lin_vel.y - current_lin_vel.y) * time_correction_factor

            ang_vel_diff_z = (next_odom_msg.twist.twist.angular.z - current_odom_msg.twist.twist.angular.z) * time_correction_factor

            # Assemble observation and label
            x = np.array([current_act_msg.throttle,
                          current_act_msg.turn,
                          current_lin_vel.x,
                          current_lin_vel.y,
                          current_odom_msg.twist.twist.angular.z], dtype=np.float32)
            y = np.array([lin_vel_diff_x,
                          lin_vel_diff_y,
                          ang_vel_diff_z])

            x_list.append(x)
            y_list.append(y)

        X = np.array(x_list)
        Y = np.array(y_list)

        return X, Y

    def gather(self):
        # Wait until all subscribers have a message to begin
        while not rospy.is_shutdown():
            if np.all([s.get_msg() is not None for s in [self.gt_odometry_sub, self.actions_sub]]): break

        dataset_dict_list = []
        # Do the gathering
        print("Started gathering")
        while not rospy.is_shutdown():
            # Get messages from all subscribers
            dataset_dict = {}
            act_msg = self.actions_sub.get_msg(copy_msg=True)
            odom_msg = self.gt_odometry_sub.get_msg(copy_msg=True)
            dataset_dict["act_msg"] = act_msg
            dataset_dict["odom_msg"] = odom_msg
            dataset_dict_list.append(dataset_dict)

            # Maintain that 200hz
            self.ros_rate.sleep()

        X, Y = self.process_dataset(dataset_dict_list)

        print("Saving dataset")
        self.save_dataset(X, Y, self.xy_dataset_paths)

if __name__=="__main__":
    bagfile_converter = BagfileConverter()
    bagfile_converter.gather()