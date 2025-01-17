#!/usr/bin/env python3
import pickle
import time

import rospy
from buggycontrol.msg import Actions, ActionsStamped
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
        for i in range(100):
            self.x_file_path = os.path.join(dataset_dir, "X_{}.pkl".format(i))
            self.y_file_path = os.path.join(dataset_dir, "Y_{}.pkl".format(i))
            if not os.path.exists(self.x_file_path):
                break

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
        #self.actions_sub = rospy.Subscriber("/actions_stamped", ActionsStamped, callback=self.actions_cb, queue_size=10)

        self.bl_to_rs_trans = get_static_tf("odom", "camera_odom_frame")
        time.sleep(0.2)

    def gt_odometry_cb(self, msg):
        with self.actions_lock:
            if self.actions_msg is None: return
            self.actions_list.append(deepcopy(self.actions_msg))
        self.gt_odometry_list.append(msg)

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

        rospy.spin()

        print("Gathered: {} action and {} odom messages".format(len(self.actions_list), len(self.gt_odometry_list)))

        for i in range(len(self.gt_odometry_list)):
            self.rotate_twist(self.gt_odometry_list[i], self.bl_to_rs_trans)

        dataset_dict_list = []
        for i in range(np.minimum(len(self.actions_list), len(self.gt_odometry_list))):
            act_msg = self.actions_list[i]
            odom_msg = self.gt_odometry_list[i]
            dataset_dict_list.append({"odom_msg" : odom_msg, "act_msg" : act_msg})

        # for i in range(100):
        #     print("Odom_msg_seq: ", dataset_dict_list[i]["odom_msg"].header.seq, "act_msg_seq: ", dataset_dict_list[i]["act_msg"].header.seq)
        # exit()

        X, Y = self.process_dataset(dataset_dict_list)

        print("Saving dataset")
        self.save_dataset(X, Y)

    def process_dataset(self, dataset_dict_list):
        x_list = []
        y_list = []
        for i in range(0, len(dataset_dict_list) - 2, 2):
            current_act_msg = dataset_dict_list[i]["act_msg"]
            current_odom_msg = dataset_dict_list[i]["odom_msg"]
            next_odom_msg = dataset_dict_list[i+2]["odom_msg"]

            # Calculate velocity delta from current to next odom
            current_lin_vel = current_odom_msg.twist.twist.linear
            current_ang_vel = current_odom_msg.twist.twist.angular
            next_lin_vel = next_odom_msg.twist.twist.linear
            next_ang_vel = next_odom_msg.twist.twist.angular

            # Assemble observation and label
            x = np.array([current_lin_vel.x,
                          current_lin_vel.y,
                          current_ang_vel.z,
                          current_act_msg.turn,
                          current_act_msg.throttle * 2 - 1], dtype=np.float32)
            y = np.array([next_lin_vel.x,
                          next_lin_vel.y,
                          next_ang_vel.z], dtype=np.float32)

            x_list.append(x)
            y_list.append(y)

        X = np.array(x_list)
        Y = np.array(y_list)

        return X, Y

    def save_dataset(self, X, Y):
        pickle.dump(X, open(self.x_file_path, "wb"))
        pickle.dump(Y, open(self.y_file_path, "wb"))

if __name__=="__main__":
    bagfile_converter = BagfileConverter()
    bagfile_converter.gather()