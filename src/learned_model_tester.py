#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import Vector3, PoseWithCovariance, Pose, Quaternion
from nav_msgs.msg import Odometry
import tf.transformations
from buggy_control.msg import Actions
import tf2_ros
import torch as T
from policies import *
import pickle
import threading
from copy import deepcopy

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
    policy.set_params(pickle.load(open("agents/buggy_transition_model.p", "rb")))

    rospy.init_node("predicted_buggy_pose_publisher")
    ros_rate = rospy.Rate(0.005)
    pub = rospy.Publisher("pred/odom_base_link", Odometry, queue_size=5)
    rospy.Subscriber("actions",
                     Actions,
                     cb,
                     queue_size=3)

    buggy_lin_vel_x = 0.
    buggy_lin_vel_y = 0.
    buggy_ang_vel_z = 0.

    delta = 0.005

    while not rospy.is_shutdown():
        with act_lock:
            if act_msg is not None:
                throttle = deepcopy(act_msg.throttle)
                turn = deepcopy(act_msg.turn)

        # Predict velocity update
        policy_input = T.tensor([throttle, turn, buggy_lin_vel_x, buggy_lin_vel_y, buggy_ang_vel_z])
        with T.no_grad:
            pred_deltas = policy(policy_input)
        x_delta, y_delta, z_ang_delta = pred_deltas.numpy

        buggy_lin_vel_x += x_delta * delta
        buggy_lin_vel_y += y_delta * delta
        buggy_ang_vel_z += z_ang_delta * delta

        # Transform linear velocities to base_link frame
        base_link_linear = rotate_vector_by_quat(Vector3(x=buggy_lin_vel_x, y=buggy_lin_vel_y, z=buggy_ang_vel_z),
                                                 integrated_pose.orientation)

        # Modify integrated pose using twist message
        integrated_pose.position.x += base_link_linear.x * delta
        integrated_pose.position.y += base_link_linear.y * delta
        integrated_pose.position.z += 0
        i_q = integrated_pose.orientation
        i_e = list(tf.transformations.euler_from_quaternion([i_q.x, i_q.y, i_q.z, i_q.w]))
        i_e[2] += buggy_ang_vel_z * delta
        i_q_new = tf.transformations.quaternion_from_euler(*i_e)
        integrated_pose.orientation = Quaternion(*i_q_new)

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time(0)
        odom_msg.header.frame_id = "camera_odom_frame"
        odom_msg.pose = PoseWithCovariance(pose=integrated_pose)
        pub.publish(odom_msg)

        ros_rate.sleep()