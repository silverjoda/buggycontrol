#!/usr/bin/env python3
import pickle
import threading
from copy import deepcopy

import numpy as np
import rospy

from buggycontrol.msg import ActionsStamped
from src.envs.buggy_env_mujoco import BuggyEnv
from src.utils import load_config
import os
import time

if __name__=="__main__":
    def cb(msg):
        with act_lock:
            global act_msg
            act_msg = msg

    act_msg = None
    act_lock = threading.Lock()

    print("Starting buggy tester node")
    rospy.init_node("mujoco_buggy_env_tester")
    rospy.Subscriber("actions",
                     ActionsStamped,
                     cb,
                     queue_size=3)

    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
    env = BuggyEnv(config)
    env.reset()
    throttle, turn = [0, 0]
    while not rospy.is_shutdown():
        with act_lock:
            if act_msg is not None:
                throttle = np.maximum(deepcopy(act_msg.throttle), 0) * 2 - 1
                turn = deepcopy(act_msg.turn)

        _, r, done, _ = env.step([turn, throttle])
        print(r)
        if done:
            env.reset()
        env.render()
        time.sleep(0.007)