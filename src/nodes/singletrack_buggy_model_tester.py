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

from src.ctrl.model import make_singletrack_model, make_bicycle_model
from src.ctrl.simulator import make_simulator


class BuggyModelTester:
    def __init__(self):
        self.act_msg = None
        self.act_lock = threading.Lock()

        #model = make_singletrack_model([3, 2, 0.14, 0.16, 0.04, 1, 6.9, 1.8, 0.1, 1, 15, 1.7, -0.5, 100])
        model = make_singletrack_model([2.5, 0.9, 0.20, 0.175, 0.041, 1, 6.9, 1.8, 0.1, 1, 15, 1.7, -0.5, 100])
        #model = make_bicycle_model()
        self.simulator = make_simulator(model)
        self.simulator.reset_history()

        print("Starting single track tester node")
        rospy.init_node("single_track_tester")
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.ros_rate = rospy.Rate(100)
        self.pub_twist = rospy.Publisher("pred/twist_base_link", TwistStamped, queue_size=5)
        self.pub_odom = rospy.Publisher("pred/odom_base_link", Odometry, queue_size=5)
        rospy.Subscriber("actions",
                         ActionsStamped,
                         self.act_cb,
                         queue_size=3)

    def act_cb(self, msg):
        with self.act_lock:
            self.act_msg = msg

    def test_bicycle(self):
        turn = 0.
        throttle = 0.01
        self.simulator.x0 = np.array([0.00, 0.00, 0.00, 0.2, 0.01, 0.01]).reshape(-1, 1)

        while not rospy.is_shutdown():
            with self.act_lock:
                if self.act_msg is not None:
                    throttle = np.clip(deepcopy(self.act_msg.throttle), 0.01, 1.)
                    turn = deepcopy(self.act_msg.turn * 0.4)

            for i in range(5):
                x = self.simulator.make_step(np.array([turn, throttle]).reshape(2, 1))

            #beta, v, ang_vel_z, xpos, ypos, ang_z = x
            xpos, ypos, phi, xvel, yvel, omega = x
            orientation_quat = tf.transformations.quaternion_from_euler(0, 0, phi)

            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time(0)
            twist_msg.header.frame_id = "base_link"
            twist_msg.twist = Twist(linear=Vector3(x=xvel, y=yvel), angular=Vector3(z=omega))
            self.pub_twist.publish(twist_msg)

            pose = Pose()
            pose.position.x = xpos
            pose.position.y = ypos
            pose.position.z = 0

            pose.orientation.x = orientation_quat[0]
            pose.orientation.y = orientation_quat[1]
            pose.orientation.z = orientation_quat[2]
            pose.orientation.w = orientation_quat[3]

            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time(0)
            odom_msg.header.frame_id = "odom"
            odom_msg.pose = PoseWithCovariance(pose=pose)
            self.pub_odom.publish(odom_msg)

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
            self.broadcaster.sendTransform(t)

            self.ros_rate.sleep()

    def test_singletrack(self):
        turn = 0.
        throttle = 0.1
        self.simulator.x0 = np.array([0.03, 0.05, 0.01, 0.0, 0.0, 0.0]).reshape(-1, 1)

        while not rospy.is_shutdown():
            with self.act_lock:
                if self.act_msg is not None:
                    turn = deepcopy(self.act_msg.turn * 0.6)
                    throttle = np.clip(deepcopy(self.act_msg.throttle), 0.01, 1.)

            for i in range(5):
                x = self.simulator.make_step(np.array([turn, throttle * 94]).reshape(2, 1))

            beta, v, ang_vel_z, xpos, ypos, ang_z = x
            orientation_quat = tf.transformations.quaternion_from_euler(0, 0, ang_z)

            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time(0)
            twist_msg.header.frame_id = "base_link"
            twist_msg.twist = Twist(linear=Vector3(x=v * np.cos(beta), y=v * np.sin(beta)),
                                    angular=Vector3(z=ang_vel_z))
            self.pub_twist.publish(twist_msg)

            pose = Pose()
            pose.position.x = xpos
            pose.position.y = ypos
            pose.position.z = 0

            pose.orientation.x = orientation_quat[0]
            pose.orientation.y = orientation_quat[1]
            pose.orientation.z = orientation_quat[2]
            pose.orientation.w = orientation_quat[3]

            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time(0)
            odom_msg.header.frame_id = "odom"
            odom_msg.pose = PoseWithCovariance(pose=pose)
            self.pub_odom.publish(odom_msg)

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
            self.broadcaster.sendTransform(t)

            self.ros_rate.sleep()

if __name__=="__main__":
    bmt = BuggyModelTester()
    #bmt.test_bicycle()
    bmt.test_singletrack()

