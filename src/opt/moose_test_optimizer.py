import os
import pathlib
import pickle
import tempfile
import time

import numpy as np
import stable_baselines3 as sb3
import torch as T
from imitation.algorithms.adversarial import gail, airl
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.rewards import reward_nets
from imitation.util import logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

from src.envs.buggy_env_mujoco import BuggyEnv
from src.opt.simplex_noise import SimplexNoise
from src.policies import MLP, RNN
from src.utils import load_config

import os
import random
import time
from pprint import pprint

import torch as T
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecMonitor

from src.envs import buggy_env_mujoco
from src.utils import merge_dicts

os.environ["CUDA_VISIBLE_DEVICES"] = ""
T.set_num_threads(1)

class MooseTestOptimizer:
    def __init__(self):
        self.policy_ID = "TRN"
        env_config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
        self.env, self.sb_model = self.load_model_and_env(env_config)

        self.b1_pos, self.b2_pos = [4, 0.0], [5.5, 1.0]
        self.barrier_loss_fun = self.barriers_lf()

    def load_model_and_env(self, env_config):
        # Policy + VF
        sb_model = A2C.load(f"agents/{self.policy_ID}_SB_policy")

        # Wrapped env
        vec_env = DummyVecEnv(env_fns=[lambda: BuggyEnv(env_config)] * 1)
        monitor_env = VecMonitor(vec_env)
        normed_env = VecNormalize(venv=monitor_env, training=False, norm_obs=True, norm_reward=True, clip_reward=10.0)
        stats_path = f"agents/{self.policy_ID}_vecnorm.pkl"
        env = VecNormalize.load(stats_path, normed_env)

        return env, sb_model

    def generate_initial_traj(self):
        pass

    def set_visual_barriers_env(self):
        pass

    def set_visual_traj_env(self):
        pass

    def barriers_lf(self, traj):
        mse_loss = T.nn.MSELoss()
        total_loss = T.tensor(0., requires_grad=True)

        b1_T = T.tensor(self.b1_pos, requires_grad=False)
        b2_T = T.tensor(self.b2_pos, requires_grad=False)

        traj_T = [T.tensor(pt, dtype=T.float32, requires_grad=True) for pt in traj]
        # Barrier constraints
        for xy_T in traj_T:
            total_loss = total_loss + mse_loss(xy_T, b1_T) + mse_loss(xy_T, b2_T)

        # Initial point has to stay put

        # End point stretched out as much as possible
        end_pt_loss = -traj_T[-1, 0]

        # Minimize square distance between points
        inter_pt_loss = T.tensor(0., requires_grad=True)
        for i in range(len(traj_T) - 1):
            inter_pt_loss = inter_pt_loss + mse_loss(xy_T[i], xy_T[i+1])

        return total_loss


