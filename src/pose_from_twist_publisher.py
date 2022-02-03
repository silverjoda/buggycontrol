#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped, Vector3, PoseWithCovariance, Pose, Quaternion
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
        prev_ts = msg.header.stamp

        # Use previous message for integration
        global prev_msg
        if prev_msg is None:
            prev_msg = msg

        # Transform linear velocities to base_link frame
        base_link_linear = rotate_vector_by_quat(prev_msg.twist.linear, integrated_pose.orientation)

        # Modify integrated pose using twist message
        integrated_pose.position.x += base_link_linear.x * delta
        integrated_pose.position.y += base_link_linear.y * delta
        integrated_pose.position.z += 0 # base_link_linear.z * delta
        i_q = integrated_pose.orientation
        i_e = list(tf.transformations.euler_from_quaternion([i_q.x, i_q.y, i_q.z, i_q.w]))
        i_e[0] += prev_msg.twist.angular.x * delta
        i_e[1] += prev_msg.twist.angular.y * delta
        i_e[2] += prev_msg.twist.angular.z * delta
        i_q_new = tf.transformations.quaternion_from_euler(*i_e)
        integrated_pose.orientation = Quaternion(*i_q_new)

        prev_msg = msg

        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.header.frame_id = "camera_odom_frame"
        odom_msg.pose = PoseWithCovariance(pose=integrated_pose)
        pub.publish(odom_msg)

    integrated_pose = Pose(orientation=Quaternion(x=0, y=0, z=0, w=1))

    rospy.init_node("reconstructed_pose_republisher")
    pub = rospy.Publisher("recon/odom_base_link", Odometry, queue_size=5)
    prev_ts = None
    prev_msg = None
    rospy.Subscriber("gt/twist_base_link",
                     TwistStamped,
                     cb,
                     queue_size=3)
    rospy.spin()