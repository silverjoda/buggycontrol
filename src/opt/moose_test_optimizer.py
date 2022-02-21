import os
import pathlib
import pickle
import tempfile
import time

import numpy as np
import stable_baselines3 as sb3
import torch as T
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
        self.env_config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
        self.env, self.venv, self.sb_model = self.load_model_and_env(self.env_config)

        self.b1_pos, self.b2_pos = [4, 0.5], [6, 0.5]

        #self.barrier_loss_fun = self.barriers_lf()

    def test_agent(self, render=True, print_rew=True):
        total_rew = 0

        traj = self.generate_initial_traj()
        traj = self.optimize_traj_dist(traj, n_iters=1000)
        exit()

        for _ in range(100):
            # Reset
            obs = self.venv.reset()

            self.env.engine.set_trajectory(traj)
            self.env.set_barrier_positions([4.0, 0.0], [6.0, 1.0])

            episode_rew = 0
            while True:
                action, _states = self.sb_model.predict(obs, deterministic=True)
                obs, reward, done, info = self.venv.step(action)
                episode_rew += self.venv.get_original_reward()
                total_rew += self.venv.get_original_reward()
                if render:
                    self.venv.render()
                if done:
                    if print_rew:
                        print(episode_rew)
                    break
        return total_rew

    def optimize_traj_dist(self, traj, n_iters=1):
        mse_loss = T.nn.MSELoss()

        def explf(a, b):
            return 1. / T.exp_(10 * T.abs(a - b))

        traj_T = [T.tensor(pt, dtype=T.float32, requires_grad=True) for pt in traj]

        # Plot
        import matplotlib.pyplot as plt
        plt.ion()
        figure, ax = plt.subplots(figsize=(8, 6))
        line1, = ax.plot(list(zip(*traj))[0], list(zip(*traj))[1])
        ax.scatter([4,6],[.5,.5])
        #

        for i in range(n_iters):
            b1_T = T.tensor(self.b1_pos, requires_grad=False)
            b2_T = T.tensor(self.b2_pos, requires_grad=False)

            # Barrier constraints
            barrier_loss_list = []
            for xy_T in traj_T:
                barrier_loss_list.append( (explf(xy_T, b1_T) + explf(xy_T, b2_T)) * 0.01)

            # End point stretched out as much as possible
            final_pt_loss = mse_loss(traj_T[-1], T.tensor([12., 0])) * 0.2

            # Minimize square distance between points
            inter_pt_loss_list = []
            for i in range(len(traj_T) - 1):
                inter_pt_loss_list.append(mse_loss(traj_T[i], traj_T[i + 1]) * 0.9)

            total_loss = T.stack(barrier_loss_list).sum() + final_pt_loss + T.stack(inter_pt_loss_list).sum()
            total_loss.backward()

            with T.no_grad():
                for pt in traj_T[1:]:
                    pt -= pt.grad * 0.01

            # PLOT
            x, y = list(zip(*[t.detach().numpy() for t in traj_T]))
            line1.set_xdata(x)
            line1.set_ydata(y)

            figure.canvas.draw()

            figure.canvas.flush_events()
            #time.sleep(0.1)
            #

        optimized_traj = [pt.detach().numpy() for pt in traj_T]
        return optimized_traj

    def load_model_and_env(self, env_config):
        # Policy + VF
        sb_model = A2C.load(f"agents/{self.policy_ID}_SB_policy")

        # Wrapped env
        env = BuggyEnv(env_config)
        vec_env = DummyVecEnv(env_fns=[lambda: env] * 1)
        monitor_env = VecMonitor(vec_env)
        normed_env = VecNormalize(venv=monitor_env, training=False, norm_obs=True, norm_reward=True, clip_reward=10.0)
        stats_path = f"agents/{self.policy_ID}_vecnorm.pkl"
        venv = VecNormalize.load(stats_path, normed_env)

        return env, venv, sb_model

    def generate_initial_traj(self):
        N = 500
        x = np.linspace(0, 10, N)
        y = np.zeros_like(x)
        y[N // 2 - 200:N // 2 + 200] = np.sin((x[N // 2 - 200:N // 2 + 200] - 1) * 3.1415 * 0.25)
        windowing_func = 6. / np.exp(1.2 * np.abs(x - 5))
        y = y * windowing_func

        # import matplotlib.pyplot as plt
        # plt.plot(x, y)
        # plt.show()

        #import matplotlib.pyplot as plt
        #x = np.linspace(0, 1, N)
        #plt.plot(x, 1. / np.exp(10 * np.abs(x - .5)))
        #plt.show()
        #exit()

        ftraj = list(zip(x, y))

        straj = self.ftraj_to_straj(ftraj)
        return straj

    def dist_between_wps(self, wp_1, wp_2):
        return np.sqrt(np.square(wp_1[0] - wp_2[0]) + np.square(wp_1[1] - wp_2[1]))

    def ftraj_to_straj(self, ftraj):
        # Sample equidistant points
        wp_list = []
        cum_wp_dist = 0
        for i in range(1, len(ftraj)):
            cum_wp_dist += self.dist_between_wps(ftraj[i], ftraj[i - 1])
            if cum_wp_dist > self.env_config["wp_sample_dist"]:
                wp_list.append(ftraj[i])
                cum_wp_dist = 0
        return wp_list

    def set_visual_barriers_env(self):
        pass

    def set_visual_traj_env(self):
        pass



if __name__ == "__main__":
    be = MooseTestOptimizer()
    be.test_agent()
