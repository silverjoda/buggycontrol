#!/usr/bin/env python3
import pickle
import time

import rospy
from buggycontrol.msg import Actions
from nav_msgs.msg import Odometry

from src.utils_ros import *
from threading import Lock

import message_filters

class BagfileConverter:
    def __init__(self):
        self.create_dataset_path()
        self.init_ros()

    def create_dataset_path(self):
        # Save dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/buggy_real_dataset")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Find last indexed dataset
        self.x_file_path = os.path.join(dataset_dir, "X.pkl")
        self.y_file_path = os.path.join(dataset_dir, "Y.pkl")

    def init_ros(self):
        rospy.init_node("bagfile_converter")

        self.gt_odometry_list = []
        self.gt_odometry_lock = Lock()
        self.gt_odometry_msg = None
        self.gt_odometry_sub = rospy.Subscriber("/camera/odom/sample", Odometry, callback=self.gt_odometry_cb, queue_size=10)

        self.actions_list = []
        self.actions_lock = Lock()
        self.actions_msg = None
        self.actions_sub = rospy.Subscriber("/actions", Actions, callback=self.actions_cb, queue_size=10)

        self.bl_to_rs_trans = get_static_tf("odom", "camera_odom_frame")
        self.ros_rate = rospy.Rate(200)
        time.sleep(0.2)

    def gt_odometry_cb(self, msg):
        with self.gt_odometry_lock:
            self.gt_odometry_msg = msg

    def actions_cb(self, msg):
        with self.actions_lock:
            self.actions_msg = msg

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

        while not rospy.is_shutdown():
            with self.gt_odometry_lock:
                gt_odometry_rdy = self.gt_odometry_msg is not None

            with self.actions_lock:
                action_rdy = self.actions_msg is not None

            if gt_odometry_rdy and action_rdy: break

        # Wait until user kills
        while not rospy.is_shutdown():
            with self.gt_odometry_lock:
                self.gt_odometry_list.append(self.gt_odometry_msg)
            with self.actions_lock:
                self.actions_list.append(self.actions_msg)
            self.ros_rate.sleep()

        print("Gathered: {} action and {} odom messages".format(len(self.actions_list), len(self.gt_odometry_list)))

        for i in range(len(self.gt_odometry_list)):
            self.rotate_twist(self.gt_odometry_list[i], self.bl_to_rs_trans)

        dataset_dict_list = []
        for i in range(np.minimum(len(self.actions_list), len(self.gt_odometry_list))):
            act_msg = self.actions_list[i]
            odom_msg = self.gt_odometry_list[i]
            dataset_dict_list.append({"odom_msg" : odom_msg, "act_msg" : act_msg})

        X_loaded, Y_loaded = self.load_dataset_if_exists()
        X, Y = self.process_dataset(dataset_dict_list, X_loaded, Y_loaded)

        print("Saving dataset")
        self.save_dataset(X, Y)

    def load_dataset_if_exists(self):
        X, Y = None, None
        if os.path.exists(self.x_file_path):
            X = np.load(self.x_file_path, allow_pickle=True)
            Y = np.load(self.y_file_path, allow_pickle=True)
        return X, Y

    def process_dataset(self, dataset_dict_list, X_loaded, Y_loaded):
        x_list = []
        y_list = []
        for i in range(len(dataset_dict_list) - 1):
            current_act_msg = dataset_dict_list[i]["act_msg"]
            current_odom_msg = dataset_dict_list[i]["odom_msg"]
            next_odom_msg = dataset_dict_list[i+1]["odom_msg"]

            # Calculate velocity delta from current to next odom
            current_lin_vel = current_odom_msg.twist.twist.linear
            current_ang_vel = current_odom_msg.twist.twist.angular
            next_lin_vel = next_odom_msg.twist.twist.linear
            next_ang_vel = next_odom_msg.twist.twist.angular

            # Assemble observation and label
            x = np.array([current_lin_vel.x,
                          current_lin_vel.y,
                          current_ang_vel.z,
                          current_act_msg.throttle,
                          current_act_msg.turn], dtype=np.float32)
            y = np.array([next_lin_vel.x,
                          next_lin_vel.y,
                          next_ang_vel.z], dtype=np.float32)

            x_list.append(x)
            y_list.append(y)

        X = np.expand_dims(np.array(x_list), 0)
        Y = np.expand_dims(np.array(y_list), 0)

        # TODO: save Xs, Ys as separate npy files for each bagfile, as before. They will then be loaded and put into a list

        if X_loaded is not None:
            X = np.concatenate((X, X_loaded), axis=0)
            Y = np.concatenate((Y, Y_loaded), axis=0)

        return X, Y

    def save_dataset(self, X, Y):
        pickle.dump(X, open(self.x_file_path, "wb"))
        pickle.dump(Y, open(self.y_file_path, "wb"))

if __name__=="__main__":
    bagfile_converter = BagfileConverter()
    bagfile_converter.gather()