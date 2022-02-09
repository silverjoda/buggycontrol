import os
import time

import gym
import mujoco_py
import numpy as np
import yaml
from gym import spaces
from xml_gen import *
from src.policies import LTE
import torch as T
from src.utils import load_config
from engines import *
from simplex_noise import SimplexNoise

class BuggyEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        "video.frames_per_second": 100
    }

    def __init__(self):
        self.config = load_config(os.path.join(os.path.dirname(__file__), "configs/buggy_env_mujoco.yaml"))
        self.buddy_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/cars/base_car/buddy.xml")
        self.buddy_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/cars/base_car/buddy_rnd.xml")

        self.car_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car.xml")
        self.car_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car_rnd.xml")

        self.sim, self.engine = self.load_random_env()

        self.obs_dim = self.config["state_dim"] + self.config["n_trajectory_pts"] * 3 + self.config["allow_latent_input"] * self.config["latent_dim"] + self.config["allow_lte"]
        self.act_dim = 2

        if self.config["allow_lte"]:
            self.lte = LTE(obs_dim=self.config["state_dim"] + 2, act_dim=self.config["state_dim"])
            self.lte.load_state_dict(T.load("opt/agents/buggy_lte.p"), strict=False)

        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

    def load_random_env(self):
        self.random_params = np.random.randn(5)
        if self.config["allow_lte"]:
            self.random_params = np.concatenate([self.random_params, np.array([-1])])
            if np.random.rand() < self.config["lte_prob"]:
                self.random_params = [0,0,0,0,0,1.]
            model = mujoco_py.load_model_from_path(self.car_template_path)
            sim = mujoco_py.MjSim(model, nsubsteps=self.config['n_substeps'])
            engine = LTEEngine(sim, self.lte)
        else:
            buddy_xml = gen_buddy_xml(self.random_params)
            with open(self.buddy_rnd_path, "w") as out_file:
                for s in buddy_xml.splitlines():
                    out_file.write(s)

            car_xml = gen_car_xml(self.random_params)
            with open(self.car_rnd_path, "w") as out_file:
                for s in car_xml.splitlines():
                    out_file.write(s)
            model = mujoco_py.load_model_from_path(self.car_template_path)
            sim = mujoco_py.MjSim(model, nsubsteps=self.config['n_substeps'])
            engine = MujocoEngine(sim)

        return sim, engine

    def get_obs_dict(self):
        return self.engine.get_obs_dict()

    def get_state_vec(self):
        return self.engine.get_state_vec()

    def get_reward(self, obs_dict):
        return 0

    def step(self, action):
        self.engine.step(action)

        # Get new observation
        obs_dict = self.engine.get_obs_dict()
        state_vec = self.engine.get_state_vec()
        complete_obs_vec = self.engine.get_complete_obs_vec()

        # calculate reward
        r = self.get_reward(obs_dict)

        # Calculate termination
        done = False

        return complete_obs_vec, r, done, {}

    def reset(self):
        # Reset simulation
        self.engine.reset()

        # Reset environment variables
        return self.engine.get_complete_obs_vec()

    def render(self, mode=None):
        self.engine.render()

    def generate_random_traj(self, n_pts):
        self.noise = SimplexNoise(dim=2, smoothness=100, multiplier=0.1)
        self.traj_pts = []
        current_xy = np.zeros(2)

        # Generate fine grained trajectory
        for i in range(1000):
            current_xy += self.noise()
            self.traj_pts.append(current_xy)

        # Sample equidistant points


    def demo(self):
        while True:
            self.engine.step([0,0])
            if self.config["render"]:
                self.engine.render()
            time.sleep(1. / self.config["rate"])

if __name__ == "__main__":
    be = BuggyEnv()
    be.demo()
