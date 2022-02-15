#!/usr/bin/env python3
import pickle
import threading
from copy import deepcopy

import numpy as np
import rospy
import tf.transformations
from buggycontrol.msg import ActionsStamped
from geometry_msgs.msg import Vector3, PoseWithCovariance, Pose, Quaternion, TwistStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
import os

from src.policies import *

def rotate_vector_by_quat(v, q):
    qm = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    new_v = np.matmul(qm, np.array([v.x, v.y, v.z]))
    return Vector3(x=new_v[0], y=new_v[1], z=new_v[2])

if __name__=="__main__":
    def cb(msg):
        with act_lock:
            global act_msg
            act_msg = msg

    act_msg = None
    act_lock = threading.Lock()

    integrated_pose = Pose(orientation=Quaternion(x=0, y=0, z=0, w=1))

    policy = MLP(5, 3, hid_dim=128)
    agent_path = os.path.join(os.path.dirname(__file__), "../opt/agents/buggy_lte.p")
    policy.load_state_dict(T.load(agent_path), strict=False)

    print("Starting buggy tester node")
    rospy.init_node("predicted_buggy_pose_publisher")
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

    buggy_lin_vel_x = 0.
    buggy_lin_vel_y = 0.
    buggy_ang_vel_z = 0.

    delta = 0.005
    throttle = 0.
    turn = 0.

    while not rospy.is_shutdown():
        with act_lock:
            if act_msg is not None:
                throttle = np.maximum(deepcopy(act_msg.throttle), 0)
                turn = deepcopy(act_msg.turn)

        # Predict velocity update
        policy_input = T.tensor([throttle, turn, buggy_lin_vel_x, buggy_lin_vel_y, buggy_ang_vel_z], dtype=T.float32)
        with T.no_grad():
            pred_vel = policy(policy_input)
        buggy_lin_vel_x, buggy_lin_vel_y, buggy_ang_vel_z = pred_vel.numpy()
        #buggy_lin_vel_x = np.maximum(buggy_lin_vel_x, 0)

        #print(buggy_lin_vel_x, buggy_lin_vel_y, buggy_ang_vel_z)

        # Transform linear velocities to base_link frame
        base_link_linear = rotate_vector_by_quat(Vector3(x=buggy_lin_vel_x, y=buggy_lin_vel_y, z=buggy_ang_vel_z),
                                                 integrated_pose.orientation)

        # Modify integrated pose using twist message
        integrated_pose.position.x += base_link_linear.x * delta
        integrated_pose.position.y += base_link_linear.y * delta
        i_q = integrated_pose.orientation
        i_e = list(tf.transformations.euler_from_quaternion([i_q.x, i_q.y, i_q.z, i_q.w]))
        i_e[2] += buggy_ang_vel_z * delta
        i_q_new = tf.transformations.quaternion_from_euler(*i_e)
        integrated_pose.orientation = Quaternion(*i_q_new)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time(0)
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist = Twist(linear=Vector3(x=buggy_lin_vel_x, y=buggy_lin_vel_y), angular=Vector3(z=buggy_ang_vel_z))
        pub_twist.publish(twist_msg)

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time(0)
        odom_msg.header.frame_id = "odom"
        odom_msg.pose = PoseWithCovariance(pose=integrated_pose)
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