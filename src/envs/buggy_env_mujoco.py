import os
import time

import gym
import mujoco_py
import numpy as np
import yaml
from gym import spaces


class BuggyEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        "video.frames_per_second": 100
    }

    def __init__(self):
        self.config = self.load_config()
        self.xml_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car.xml")
        self.xml_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/rnd_car.xml")

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
        self.model = mujoco_py.load_model_from_path(self.xml_template_path)
        return mujoco_py.MjSim(self.model, nsubsteps=self.config['n_substeps'])

    def get_env_obs(self):
        pass

    def get_reward(self):
        pass

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
