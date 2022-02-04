#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import QuaternionStamped, TransformStamped

if __name__=="__main__":
    def cb(msg):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "imu_link"
        t.child_frame_id = "imu"
        t.transform.rotation.x = msg.quaternion.x
        t.transform.rotation.y = msg.quaternion.y
        t.transform.rotation.z = msg.quaternion.z
        t.transform.rotation.w = msg.quaternion.w
        broadcaster.sendTransform(t)

    rospy.init_node("quat_republisher")

    tfBuffer = tf2_ros.Buffer()
    tflistener = tf2_ros.TransformListener(tfBuffer)
    broadcaster = tf2_ros.TransformBroadcaster()

    rospy.Subscriber("filter/quaternion",
                     QuaternionStamped,
                     cb,
                     queue_size=10)
    rospy.spin()