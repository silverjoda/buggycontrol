#!/usr/bin/env python3
import rospy
import tf.msg
import numpy as np
from buggycontrol.msg import Actions
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, TransformStamped
from scripts.engine.engines.cnnengine import CNNEngine


class LearnedOdomPublisher:
    def __init__(self):
        rospy.init_node("learnedodompublisher")
        self.throttle = 0.
        self.turn = 0.
        rospy.Subscriber("actions", Actions, self.callback)
        pub = rospy.Publisher("learnedodom", Odometry, queue_size=10)
        pub_tf = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=10)
        self.engine = CNNEngine(timestamp="2021_11_18_11_51_17_799478")
        self.base_link = "base_link"
        self.odom_frame = "odom"
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            observation = {"throttle": self.throttle, "turn": self.turn}
            self.engine.step(observation=observation)
            msg = self.makeodommsg()
            pub.publish(msg)
            tfmsg = self.maketfmsg()
            pub_tf.publish(tfmsg)
            rate.sleep()

    def makeodommsg(self) -> Odometry:
        """
        :return: odometry message
        """
        pos = self.engine.get_position()
        orn = self.engine.get_orientation()
        odom = Odometry()
        odom.header.frame_id = self.odom_frame
        odom.header.stamp = rospy.Time.now()
        odom.child_frame_id = self.base_link
        odom.pose.pose.position.x = pos[0]
        odom.pose.pose.position.y = pos[1]
        odom.pose.pose.position.z = pos[2]
        odom.pose.pose.orientation.x = orn[0]
        odom.pose.pose.orientation.y = orn[1]
        odom.pose.pose.orientation.z = orn[2]
        odom.pose.pose.orientation.w = orn[3]
        return odom

    def maketfmsg(self) -> tf.msg.tfMessage:
        t = TransformStamped()
        pos = self.engine.get_position()
        orn = self.engine.get_orientation()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_link
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = orn[0]
        t.transform.rotation.y = orn[1]
        t.transform.rotation.z = orn[2]
        t.transform.rotation.w = orn[3]
        return tf.msg.tfMessage([t])

    def callback(self, msg: Actions):
        """
        make engine step with received actions

        :param: actions message containing throttle and turn
        """
        self.throttle = msg.throttle
        self.turn = msg.turn


if __name__ == "__main__":
    LearnedOdomPublisher()
