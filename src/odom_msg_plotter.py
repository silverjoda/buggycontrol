#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    def odom_cb(msg):
        msg_time_list.append(msg.header.stamp.secs + msg.header.stamp.nsecs / 1000000000.)
        if len(msg_time_list) > n_msgs - 1:
            plot_msg_times()

    def plot_msg_times():
        plt.vlines(msg_time_list, np.zeros(100), np.ones(100), colors='k', linestyles='solid')
        plt.show()
        exit()

    n_msgs = 300
    msg_time_list = []

    rospy.init_node("msg_time_plotter")
    rospy.Subscriber("camera/odom/sample",
                     Odometry,
                     odom_cb,
                     queue_size=3)

    rospy.spin()