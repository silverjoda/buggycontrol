#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped, Vector3, PoseWithCovariance, Pose, Quaternion, Twist
from nav_msgs.msg import Odometry
import tf.transformations
import tf2_ros

def rotate_vector_by_quat(v, q):
    qm = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    new_v = np.matmul(qm, np.array([v.x, v.y, v.z]))
    return Vector3(x=new_v[0], y=new_v[1], z=new_v[2])

if __name__=="__main__":
    def cb(msg):
        global prev_ts
        if prev_ts is None:
            prev_ts = msg.header.stamp
        # Get time delta from last message
        delta = msg.header.stamp.secs - prev_ts.secs + \
                (msg.header.stamp.nsecs - prev_ts.nsecs) / 1000000000.
        delta = np.maximum(delta, 0.00001)
        prev_ts = msg.header.stamp

        # Use previous message for integration
        global prev_msg
        if prev_msg is None:
            prev_msg = msg

        # Take the pose diff here
        twist_lin = Vector3()
        twist_lin.x = (msg.pose.pose.position.x - prev_msg.pose.pose.position.x) / delta
        twist_lin.y = (msg.pose.pose.position.y - prev_msg.pose.pose.position.y) / delta
        twist_lin.z = (msg.pose.pose.position.z - prev_msg.pose.pose.position.z) / delta

        q = msg.pose.pose.orientation
        q_inv = tf.transformations.quaternion_inverse((q.x, q.y, q.z, q.w))
        twist_lin_bl = rotate_vector_by_quat(twist_lin, Quaternion(*q_inv))

        q = msg.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        q_p = prev_msg.pose.pose.orientation
        euler_p = tf.transformations.euler_from_quaternion([q_p.x, q_p.y, q_p.z, q_p.w])
        ang_vel = [(ee - eep) / delta for ee, eep in zip(euler, euler_p)]

        twist_ang = Vector3()
        twist_ang.x = ang_vel[0]
        twist_ang.y = ang_vel[1]
        twist_ang.z = ang_vel[2]

        prev_msg = msg

        twist_msg = TwistStamped()
        twist_msg.header.stamp = msg.header.stamp
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist = Twist(linear=twist_lin_bl, angular=twist_ang)
        pub.publish(twist_msg)

    rospy.init_node("reconstructed_twist_republisher")
    pub = rospy.Publisher("recon/twist_base_link", TwistStamped, queue_size=5)
    prev_ts = None
    prev_msg = None
    rospy.Subscriber("gt/base_link_odom",
                     Odometry,
                     cb,
                     queue_size=5)
    rospy.spin()