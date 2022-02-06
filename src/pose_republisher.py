#!/usr/bin/env python3
import numpy as np
import rospy
import tf.transformations
import tf2_ros
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TwistWithCovariance, Vector3
from nav_msgs.msg import Odometry


def rotate_vector_by_quat(v, q):
    qm = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    new_v = np.matmul(qm, np.array([v.x, v.y, v.z]))
    return Vector3(x=new_v[0], y=new_v[1], z=new_v[2])

if __name__=="__main__":
    def odom_cb(msg):
        twist_msg = TwistWithCovariance()
        twist_msg.twist.linear = rotate_vector_by_quat(msg.twist.twist.linear,
                                                       trans.transform.rotation)

        twist_msg.twist.angular = rotate_vector_by_quat(msg.twist.twist.angular,
                                                        trans.transform.rotation)

        msg.pose.pose.position.x -= trans.transform.translation.x
        msg.pose.pose.position.y -= trans.transform.translation.y
        msg.pose.pose.position.z -= trans.transform.translation.z

        q1 = msg.pose.pose.orientation
        q2 = trans.transform.rotation
        qr = tf.transformations.quaternion_multiply(np.array([q1.x, q1.y, q1.z, q1.w]),
                                                    np.array([q2.x, q2.y, q2.z, q2.w]))
        msg.pose.pose.orientation = Quaternion(*qr)
        new_msg = Odometry()
        new_msg.twist = twist_msg
        new_msg.header.stamp = msg.header.stamp
        new_msg.header.frame_id = msg.header.frame_id
        new_msg.pose = msg.pose
        posepub.publish(new_msg)

    rospy.init_node("pose_republisher")

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

    posepub = rospy.Publisher("gt/base_link_odom", Odometry, queue_size=20)
    rospy.Subscriber("camera/odom/sample", Odometry, odom_cb, queue_size=20)
    rospy.spin()