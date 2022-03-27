import os
import random
import time
from pprint import pprint
import pickle
import time

import torch as T
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecMonitor

from src.envs import buggy_env_mujoco
from src.opt.simplex_noise import SimplexNoise
from src.utils import merge_dicts
import numpy as np
GLOBAL_DEBUG = True

class BuggyMujocoDatasetGatherer:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/buggy_env_dataset_gatherer.yaml"), 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
            self.env_config = yaml.load(f, Loader=yaml.FullLoader)
        self.env = buggy_env_mujoco.BuggyEnv(self.env_config)

        self.noise = SimplexNoise(dim=2, smoothness=30, multiplier=1.0)

        dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/buggy_mujoco_dataset")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.x_file_path = os.path.join(dir_path, "X.npy")
        self.y_file_path = os.path.join(dir_path, "Y.npy")
        self.p_file_path = os.path.join(dir_path, "P.npy")

    def gather_data(self):
        X = []
        Y = []
        P = []

        # loop:
        for i in range(self.config["n_traj"]):
            obs = self.env.reset()
            obs_list = []
            act_list = []

            zero_throttle = False
            if np.random.rand() < 0.07:
                zero_throttle = True

            for j in range(self.config["traj_len"]):
                obs_list.append(obs[:5])

                # Get random action
                rnd_act = self.noise()

                # Condition act (turn, throttle)
                rnd_act[0] = np.clip(rnd_act[0], -1, 1)
                rnd_act[1] = np.clip(rnd_act[1] * 2, -1, 1)

                if zero_throttle: rnd_act[1] = -1

                act_list.append(rnd_act)

                obs, _, _, _ = self.env.step(rnd_act)

                if GLOBAL_DEBUG:
                    self.env.render()
                    time.sleep(0.008)

            x_list = []
            y_list = []

            for ti in range(self.config["traj_len"] - 1):
                x_list.append(obs_list[ti] + list(act_list[ti]))
                y_list.append(obs_list[ti + 1])

            X.append(np.array(x_list))
            Y.append(np.array(y_list))
            P.append(np.array([self.env.scaled_random_params] * self.config["traj_len"]))

            if i % 10 == 0:
                print("Trajectory gathering: {}/{}".format(i, self.config["n_traj"]))

        print("Saving trajectories")

        # Save as npy dataset
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        P_arr = np.array(P)

        np.save(self.x_file_path, X_arr)
        np.save(self.y_file_path, Y_arr)
        np.save(self.p_file_path, P_arr)

class BuggyWithPosMujocoDatasetGatherer:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/buggy_env_dataset_gatherer.yaml"), 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
            self.env_config = yaml.load(f, Loader=yaml.FullLoader)
        self.env = buggy_env_mujoco.BuggyEnv(self.env_config)

        self.noise = SimplexNoise(dim=2, smoothness=30, multiplier=1.0)

        dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/buggy_with_pos_mujoco_dataset")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.x_file_path = os.path.join(dir_path, "X.npy")
        self.y_file_path = os.path.join(dir_path, "Y.npy")
        self.p_file_path = os.path.join(dir_path, "P.npy")

    def gather_data(self):
        X = []
        Y = []
        P = []

        # loop:
        for i in range(self.config["n_traj"]):
            obs = self.env.reset()
            obs_list = []
            act_list = []

            zero_throttle = False
            if np.random.rand() < 0.05:
                zero_throttle = True

            for j in range(self.config["traj_len"]):
                obs_dict = self.env.get_obs_dict()
                xy = obs_dict['pos']
                phi = obs_dict['phi']
                obs_list.append(obs[:5])

                # Get random action
                rnd_act = self.noise()

                # Condition act (turn, throttle)
                rnd_act[0] = np.clip(rnd_act[0], -1, 1)
                rnd_act[1] = np.clip(rnd_act[1], -1, 1)

                if zero_throttle: rnd_act[1] = -1

                act_list.append(rnd_act)

                obs, _, _, _ = self.env.step(rnd_act)

                if GLOBAL_DEBUG:
                    self.env.render()
                    time.sleep(0.008)

            x_list = []
            y_list = []

            for ti in range(self.config["traj_len"] - 1):
                x_list.append(obs_list[ti] + list(act_list[ti]))
                y_list.append(obs_list[ti + 1])

            X.append(np.array(x_list))
            Y.append(np.array(y_list))
            P.append(np.array([self.env.scaled_random_params] * self.config["traj_len"]))

            if i % 10 == 0:
                print("Trajectory gathering: {}/{}".format(i, self.config["n_traj"]))

        print("Saving trajectories")

        # Save as npy dataset
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        P_arr = np.array(P)

        np.save(self.x_file_path, X_arr)
        np.save(self.y_file_path, Y_arr)
        np.save(self.p_file_path, P_arr)


if __name__ == "__main__":
    dg = BuggyMujocoDatasetGatherer()
    dg.gather_data()
