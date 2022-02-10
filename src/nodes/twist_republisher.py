#!/usr/bin/env python3
import numpy as np
import rospy
import tf.transformations
import tf2_ros
from geometry_msgs.msg import TwistStamped, Vector3
from nav_msgs.msg import Odometry


def rotate_vector_by_quat(v, q):
    qm = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    new_v = np.matmul(qm, np.array([v.x, v.y, v.z]))
    return Vector3(x=new_v[0], y=new_v[1], z=new_v[2])

if __name__=="__main__":
    def odom_cb(msg):
        msg.twist.twist.linear = rotate_vector_by_quat(msg.twist.twist.linear,
                                                       trans.transform.rotation)
        msg.twist.twist.angular = rotate_vector_by_quat(msg.twist.twist.angular,
                                                       trans.transform.rotation)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = msg.header.stamp
        twist_msg.header.frame_id = "base_link"
        twist_msg.twist = msg.twist.twist
        twistpub.publish(twist_msg)

    rospy.init_node("twist_republisher")

    tfBuffer = tf2_ros.Buffer()
    tflistener = tf2_ros.TransformListener(tfBuffer)
    while True:
        try:
            trans = tfBuffer.lookup_transform("camera_pose_frame",
                                          "base_link",
                                          rospy.Time(0),
                                          rospy.Duration(0))
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn_throttle(1, "Pose republisher could not lookup tf: {}".format(err))
            continue

    twistpub = rospy.Publisher("gt/twist_base_link", TwistStamped, queue_size=3)
    rospy.Subscriber("camera/odom/sample",
                     Odometry,
                     odom_cb,
                     queue_size=3)
    rospy.spin()