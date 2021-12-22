#!/usr/bin/env python3
import rospy
from utils import loaddefaultconfig
from buggycontrol.msg import Actions
import numpy as np


class ActionsConverter:
    def __init__(self):
        self.config = loaddefaultconfig()
        rospy.init_node("actionsconverter")
        rospy.Subscriber("actions", Actions, self.callback)
        self.pub = rospy.Publisher("servos", Actions, queue_size=10)
        rospy.spin()

    def action_to_servos(self, throttle: float, turn: float) -> (float, float):
        """
        map joystick or agent actions to motors

        :param throttle: joystick input throttle
        :param turn: joystick input turn
        :return: (servo throttle, servo turn)
        """
        sthrottle = 0.5 * throttle * self.config["motor_scalar"] + self.config["throttle_offset"]
        sthrottle = np.clip(sthrottle, 0.5, 1)
        sturn = turn / 2 + 0.5
        return sthrottle, sturn

    def callback(self, msg: Actions):
        """
        :param msg: actions input from joystick or agent
        """
        smsg = Actions()
        smsg.throttle, smsg.turn = self.action_to_servos(throttle=msg.throttle, turn=msg.turn)
        smsg.buttonA, smsg.buttonB = msg.buttonA, msg.buttonB
        self.pub.publish(smsg)


if __name__ == "__main__":
    ActionsConverter()
