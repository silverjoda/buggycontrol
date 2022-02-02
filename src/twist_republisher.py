#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

if __name__=="__main__":
    def odom_cb(msg):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = msg.header.stamp
        twist_msg.header.frame_id = "camera_accel_frame"
        twist_msg.twist = msg.twist.twist
        twistpub.publish(twist_msg)

    rospy.init_node("twist_republisher")
    twistpub = rospy.Publisher("rs_twist", TwistStamped, queue_size=3)
    rospy.Subscriber("camera/odom/sample",
                     Odometry,
                     odom_cb,
                     queue_size=3)
    rospy.spin()