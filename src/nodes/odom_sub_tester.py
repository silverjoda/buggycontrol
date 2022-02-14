#!/usr/bin/env python3
import pickle
from nav_msgs.msg import Odometry

from src.utils_ros import *
from threading import Lock
import time

class Odomt:
    def __init__(self):
        self.init_ros()

    def save_dataset(self, X, Y, file_paths):
        pickle.dump(X, open(file_paths[0], "wb"))
        pickle.dump(Y, open(file_paths[1], "wb"))

    def init_ros(self):
        rospy.init_node("bagfile_converter")

        self.gt_odometry_lock = Lock()
        self.gt_odometry_sub = rospy.Subscriber("/camera/odom/sample", Odometry, callback=self.gt_odometry_cb, queue_size=10)
        self.msg_list = []

    def test(self):
        time.sleep(3)
        with self.gt_odometry_lock:
            print([m.header.seq for m in self.msg_list])
        exit()

    def gt_odometry_cb(self, msg):
        with self.gt_odometry_lock:
            self.msg_list.append(deepcopy(msg))

if __name__=="__main__":
    odomt = Odomt()
    odomt.test()