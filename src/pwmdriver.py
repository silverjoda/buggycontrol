#!/usr/bin/env python
import Adafruit_PCA9685
import time
import rospy
import numpy as np
from buggycontrol.msg import Actions
from utils import loaddefaultconfig


class PWMDriver:
    def __init__(self):
        self.config = loaddefaultconfig()
        self.pwm_freq = int(1. / self.config["update_period"])
        self.servo_ids = [0, 1] # MOTOR IS 0, TURN is 1
        self.throttlemin = 0.5
        self.turnmiddle = 0.5
        print("Initializing the PWMdriver. ")
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(self.pwm_freq)
        self.arm_escs()
        print("Finished initializing the PWMdriver. ")
        rospy.init_node("pwmdriver")
        rospy.Subscriber("servos", Actions, self.callback)
        rospy.spin()

    def write_servos(self, vals):
        """
        :param vals: [throttle, turn]
        :return: None
        """
        for sid in self.servo_ids:
            pulse_length = ((np.clip(vals[sid], 0, 1) + 1) * 1000) / ((1000000. / self.pwm_freq) / 4096.)
            self.pwm.set_pwm(sid, 0, int(pulse_length))

    def arm_escs(self):
        """
        write lowest value to servos

        :return: None
        """
        time.sleep(0.1)
        print("Setting escs to lowest value. ")
        self.write_servos([self.throttlemin, self.turnmiddle])
        time.sleep(0.3)

    def callback(self, msg: Actions):
        """
        :param msg: actions that are to be written to motors
        """
        self.write_servos(vals=[msg.throttle, msg.turn])


if __name__ == "__main__":
    PWMDriver()
    
