#!/usr/bin/env python3
import pickle
import time

import rospy
from buggycontrol.msg import ActionsStamped
from nav_msgs.msg import Odometry

from src.utils_ros import *
from threading import Lock

import message_filters

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

    def init_ros(self):
        rospy.init_node("bagfile_converter")

        self.gt_odometry_list = []
        self.gt_odometry_lock = Lock()
        self.gt_odometry_msg = None
        #self.gt_odometry_sub = rospy.Subscriber("/camera/odom/sample", Odometry, callback=self.gt_odometry_cb, queue_size=12)
        self.gt_odometry_sub = message_filters.Subscriber("/camera/odom/sample", Odometry)

        self.actions_list = []
        self.actions_lock = Lock()
        self.actions_msg = None
        #self.actions_sub = rospy.Subscriber("/actions", Actions, callback=self.actions_cb, queue_size=15)
        self.actions_sub  = message_filters.Subscriber("/actions_stamped", ActionsStamped)

        ts = message_filters.ApproximateTimeSynchronizer([self.gt_odometry_sub, self.actions_sub], 15, 0.1)
        #ts = message_filters.ApproximateTimeSynchronizer([self.gt_odometry_sub, self.actions_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.input_cb)

        self.bl_to_rs_trans = get_static_tf("odom", "camera_odom_frame")
        time.sleep(0.3)

    def input_cb(self, odom_msg, act_msg):
        self.gt_odometry_list.append(odom_msg)
        self.actions_list.append(act_msg)

    def gt_odometry_cb(self, msg):
        with self.gt_odometry_lock:
            self.gt_odometry_msg = msg
        with self.actions_lock:
            if self.actions_msg is not None:
                self.gt_odometry_list.append(msg)

    def actions_cb(self, msg):
        with self.actions_lock:
            self.actions_msg = msg
        with self.gt_odometry_lock:
            if self.gt_odometry_msg is not None:
                self.actions_list.append(msg)

    def rotate_twist(self, odom_msg, tx):
        # Transform the twist in the odom message
        odom_msg.twist.twist.linear = rotate_vector_by_quat(odom_msg.twist.twist.linear,
                                                            tx.transform.rotation)

        odom_msg.twist.twist.angular = rotate_vector_by_quat(odom_msg.twist.twist.angular,
                                                             tx.transform.rotation)

        odom_msg.pose.pose.position.x -= tx.transform.translation.x
        odom_msg.pose.pose.position.y -= tx.transform.translation.y
        odom_msg.pose.pose.position.z -= tx.transform.translation.z

    def gather(self):
        print("Started gathering")

        # Wait until user kills
        rospy.spin()

        print("Gathered: {} action and {} odom messages".format(len(self.actions_list), len(self.gt_odometry_list)))

        for i in range(len(self.gt_odometry_list)):
            self.rotate_twist(self.gt_odometry_list[i], self.bl_to_rs_trans)

        dataset_dict_list = []
        for i in range(np.minimum(len(self.actions_list), len(self.gt_odometry_list))):
            act_msg = self.actions_list[i]
            odom_msg = self.gt_odometry_list[i]
            dataset_dict_list.append({"act_msg" : act_msg, "odom_msg" : odom_msg})

        X, Y = self.process_dataset(dataset_dict_list)

        print("Saving dataset")
        self.save_dataset(X, Y, self.xy_dataset_paths)

    def process_dataset(self, dataset_dict_list):
        #dt = 0.005
        x_list = []
        y_list = []
        for i in range(len(dataset_dict_list) - 1):
            current_act_msg = dataset_dict_list[i]["act_msg"]
            current_odom_msg = dataset_dict_list[i]["odom_msg"]
            next_odom_msg = dataset_dict_list[i+1]["odom_msg"]

            #time_delta = self.get_delta(current_odom_msg, next_odom_msg)
            #time_correction_factor = np.clip(dt / time_delta, 0.8, 1.2)

            # Calculate velocity delta from current to next odom
            current_lin_vel = current_odom_msg.twist.twist.linear
            current_ang_vel = current_odom_msg.twist.twist.angular
            next_lin_vel = next_odom_msg.twist.twist.linear
            next_ang_vel = next_odom_msg.twist.twist.angular

            #lin_vel_diff_x = (next_lin_vel.x - current_lin_vel.x) * time_correction_factor
            #lin_vel_diff_y = (next_lin_vel.y - current_lin_vel.y) * time_correction_factor

            #ang_vel_diff_z = (next_ang_vel.z - current_ang_vel.z) * time_correction_factor

            # Assemble observation and label
            x = np.array([current_act_msg.throttle,
                          current_act_msg.turn,
                          current_lin_vel.x,
                          current_lin_vel.y,
                          current_ang_vel.z], dtype=np.float32)
            y = np.array([next_lin_vel.x,
                          next_lin_vel.y,
                          next_ang_vel.z], dtype=np.float32)

            x_list.append(x)
            y_list.append(y)

        X = np.array(x_list)
        Y = np.array(y_list)

        return X, Y

    def save_dataset(self, X, Y, file_paths):
        pickle.dump(X, open(file_paths[0], "wb"))
        pickle.dump(Y, open(file_paths[1], "wb"))


if __name__=="__main__":
    bagfile_converter = BagfileConverter()
    bagfile_converter.gather()