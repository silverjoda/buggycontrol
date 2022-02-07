import os
import time

import gym
import mujoco_py
import numpy as np
import yaml
from gym import spaces
from xml_gen import *

class BuggyEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        "video.frames_per_second": 100
    }

    def __init__(self):
        self.config = self.load_config()
        self.buddy_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/cars/base_car/buddy.xml")
        self.buddy_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/cars/base_car/buddy_rnd.xml")

        self.car_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car.xml")
        self.car_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car_rnd.xml")

        self.sim = self.load_random_env()
        self.bodyid = self.model.body_name2id('buddy')
        self.viewer = mujoco_py.MjViewerBasic(self.sim) if self.config["render"] else None

        self.n_trajectory_pts = 15
        self.obs_dim = 5 + self.config["n_trajectory_pts"] * 3
        self.act_dim = 2

        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/buggy_env_mujoco.yaml"), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def load_random_env(self):
        self.random_params = [0,0,0,0]

        buddy_xml = gen_buddy_xml(self.random_params)
        with open(self.buddy_rnd_path, "w") as out_file:
            for s in buddy_xml.splitlines():
                out_file.write(s)

        car_xml = gen_car_xml(self.random_params)
        with open(self.car_rnd_path, "w") as out_file:
            for s in car_xml.splitlines():
                out_file.write(s)

        self.model = mujoco_py.load_model_from_path(self.car_template_path)
        return mujoco_py.MjSim(self.model, nsubsteps=self.config['n_substeps'])

    def get_env_obs(self):
        pos = self.sim.data.body_xpos[self.bodyid].copy()
        ori = self.sim.data.body_xquat[self.bodyid].copy()
        vel = self.sim.data.body_xvelp[self.bodyid].copy()
        return None

    def get_reward(self):
        return 0

    def step(self, action):
        # Step simulation
        self.sim.data.ctrl[:] = action
        self.sim.forward()
        self.sim.step()

        # Get new observation
        obs = self.get_env_obs()

        # calculate reward
        r = self.get_reward()

        # Calculate termination
        done = False

        return obs, r, done, {}

    def reset(self):
        # Reset simulation
        self.sim.reset()

        # Reset environment variables
        return self.get_env_obs()

    def render(self, mode=None):
        if self.viewer:
            self.viewer.render()

    def demo(self):
        while True:
            self.sim.forward()
            self.sim.step()
            if self.config["render"]:
                self.render()
            time.sleep(1. / self.config["rate"])


if __name__ == "__main__":
    be = BuggyEnv()
    be.demo()
