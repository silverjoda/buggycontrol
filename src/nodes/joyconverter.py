#!/usr/bin/env python3
import rospy
from buggycontrol.msg import ActionsStamped
from sensor_msgs.msg import Joy
import numpy as np
from threading import Lock

class JoyConverter:
    def __init__(self):
        rospy.init_node("joyconverter")
        self.lock = Lock()
        rospy.Subscriber("joy", Joy, self.callback)
        pub = rospy.Publisher("actions", ActionsStamped, queue_size=10)

        self.throttle = 0
        self.turn = 0
        self.a = False
        self.b = False
        self.update = False
        self.data = None
        seq = 0
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            with self.lock:
                if self.update:
                    self.throttle, self.turn, self.a, self.b = self.data
                    self.update = False
            msg = ActionsStamped()
            msg.throttle, msg.turn = self.throttle, self.turn
            msg.buttonA, msg.buttonB = self.a, self.b
            msg.header.seq = seq
            msg.header.frame_id = "base_link"
            msg.header.stamp = rospy.Time(0)

            pub.publish(msg)
            seq += 1
            rate.sleep()


    def callback(self, msg):
        """
        :param msg: raw input from joystick
        """
        throttle, turn = msg.axes[1], msg.axes[3]
        a, b = msg.buttons[0], msg.buttons[1]
        with self.lock:
            self.data = (throttle, turn, a, b)
            self.update = True


if __name__ == "__main__":
    JoyConverter()
