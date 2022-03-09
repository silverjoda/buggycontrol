#!/usr/bin/env python3
import threading
from copy import deepcopy

import rospy
import tf.transformations
from buggycontrol.msg import ActionsStamped
from geometry_msgs.msg import Vector3, PoseWithCovariance, Pose, Quaternion, TwistStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
from casadi import *

from src.ctrl.model import make_model
from src.ctrl.simulator import make_simulator

if __name__=="__main__":
    def cb(msg):
        with act_lock:
            global act_msg
            act_msg = msg

    act_msg = None
    act_lock = threading.Lock()

    model = make_model()
    simulator = make_simulator(model)
    simulator.reset_history()
    simulator.x0 = np.array([0., 0., 0., 0., 0., 0.]).reshape(-1, 1)

    print("Starting single track tester node")
    rospy.init_node("single_track_tester")
    tfBuffer = tf2_ros.Buffer()
    tflistener = tf2_ros.TransformListener(tfBuffer)
    broadcaster = tf2_ros.TransformBroadcaster()
    ros_rate = rospy.Rate(200)
    pub_twist = rospy.Publisher("pred/twist_base_link", TwistStamped, queue_size=5)
    pub_odom = rospy.Publisher("pred/odom_base_link", Odometry, queue_size=5)
    rospy.Subscriber("actions",
                     ActionsStamped,
                     cb,
                     queue_size=3)

    turn = 0.
    throttle = 0.

    while not rospy.is_shutdown():
        with act_lock:
            if act_msg is not None:
                throttle = np.maximum(deepcopy(act_msg.throttle), 0.)
                turn = deepcopy(act_msg.turn * 0.38)

        x = simulator.make_step(np.array([turn, throttle * 10]).reshape(2, 1))

        beta, v, ang_vel_z, xpos, ypos, ang_z = x
        orientation_quat = tf.transformations.quaternion_from_euler(0, 0, ang_z)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time(0)
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist = Twist(linear=Vector3(x=v * np.cos(beta), y=v * np.sin(beta)), angular=Vector3(z=ang_vel_z))
        pub_twist.publish(twist_msg)

        pose = Pose()
        pose.position.x = xpos
        pose.position.y = ypos
        pose.position.z = ang_z

        pose.orientation.x = orientation_quat[0]
        pose.orientation.y = orientation_quat[1]
        pose.orientation.z = orientation_quat[2]
        pose.orientation.w = orientation_quat[3]

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time(0)
        odom_msg.header.frame_id = "odom"
        odom_msg.pose = PoseWithCovariance(pose=pose)
        pub_odom.publish(odom_msg)

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z

        t.transform.rotation.x = odom_msg.pose.pose.orientation.x
        t.transform.rotation.y = odom_msg.pose.pose.orientation.y
        t.transform.rotation.z = odom_msg.pose.pose.orientation.z
        t.transform.rotation.w = odom_msg.pose.pose.orientation.w
        broadcaster.sendTransform(t)

        ros_rate.sleep()