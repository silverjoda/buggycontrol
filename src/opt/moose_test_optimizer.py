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
from src.policies import TEPTX, TEPMLP, TEPRNN
from src.utils import load_config
import math as m


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
import matplotlib.pyplot as plt
plt.ion()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
T.set_num_threads(1)
T.set_default_dtype(T.float32)

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
        #traj = self.optimize_traj_dist(traj, n_iters=1000)
        #traj = self.optimize_traj_exec(traj, self.sb_model.policy, n_iters=1000)
        traj = traj[:50]
        traj = self.optimize_traj_exec_full(traj, n_iters=300)
        #exit()

        for _ in range(100):
            #traj = self.optimize_traj_dist(traj, n_iters=10)
            #traj = self.optimize_traj_exec_full(traj, n_iters=10)

            obs = self.venv.reset()

            self.env.engine.set_trajectory(traj)
            self.env.set_barrier_positions([4.0, 0.0], [6.0, 1.0])

            episode_rew = 0
            step_ctr = 0
            while True:
                step_ctr += 1
                action, _states = self.sb_model.predict(obs, deterministic=False)
                obs, reward, done, info = self.venv.step(action)
                episode_rew += self.venv.get_original_reward()
                total_rew += self.venv.get_original_reward()
                if render:
                    self.venv.render()
                if done:
                    if print_rew:
                        print(episode_rew, step_ctr)
                    break
        return total_rew

    def transform_wp_to_buggy_frame(self, wp_arr_T, pos, ori_q):
        wp_arr_centered = wp_arr_T - T.tensor(pos[0:2])
        _, _, theta = self.q2e(*ori_q)
        t_mat = T.tensor([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        wp_buggy = T.matmul(t_mat, wp_arr_centered.T).T
        return wp_buggy

    def q2e(self, w, x, y, z):
        pitch = -m.asin(2.0 * (x * z - w * y))
        roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
        yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
        return (roll, pitch, yaw)

    def optimize_traj_exec(self, traj, etp, n_iters=1):
        def clipped_mse(a, b):
            loss = T.minimum(T.sqrt(T.square(a[0] - b[0]) + T.square(a[1] - b[1])), T.tensor(0.5))
            return loss

        def flattened_mse(p1, p2):
            a = T.tensor(-1e2)
            c = T.tensor(.2)
            x = p1 - p2
            loss = (T.abs(a - 2.) / (a)) * (T.pow((T.square(x/c)/T.abs(a - 2)) + 1, 0.5 * a) - 1.)
            return loss

        def dist_T(a, b):
            return T.sqrt(T.square(a[0] - b[0]) + T.square(a[1] - b[1]))

        mse_lf = T.nn.MSELoss()
        barrier_lf = flattened_mse

        traj_T = T.nn.ParameterList([T.nn.Parameter(T.tensor(pt, dtype=T.float32, requires_grad=True)) for pt in traj])
        optim = T.optim.Adam(params=traj_T, lr=0.001)

        # Plot
        figure, ax = plt.subplots(figsize=(14, 6))
        line1, = ax.plot(list(zip(*traj))[0], list(zip(*traj))[1], marker="o")
        ax.scatter([4, 6, 17], [.5, .5, 0], s=200, c=['r', 'r', 'w'])

        # self.venv.normalize_obs(numpy_obs)
        # self.venv.unnormalize_obs(numpy_obs)

        rms_mean = self.venv.obs_rms.mean
        rms_std = self.venv.obs_rms.var

        for it in range(n_iters):
            obs = self.venv.reset()

            self.env.engine.set_trajectory([t.detach().numpy() for t in traj_T])
            self.env.set_barrier_positions([4.0, 0.0], [6.0, 1.0])

            mse_loss = T.nn.MSELoss()

            while True:
                obs_dict = self.env.get_obs_dict()

                # Optimize single step and keep going
                traj_indeces = range(self.env.engine.cur_wp_idx, self.env.engine.cur_wp_idx + self.env_config["n_traj_pts"])
                cur_traj_T = T.stack([traj_T[ti] for ti in traj_indeces])
                cur_traj_T_buggy_frame = self.transform_wp_to_buggy_frame(cur_traj_T, obs_dict["pos"], obs_dict["ori_q"])
                cur_state_T = T.tensor(obs[:, :5])
                cur_obs_T = T.concat([cur_state_T, cur_traj_T_buggy_frame.resize(1, 30)], dim=1)

                cur_obs_T_norm = (cur_obs_T - T.tensor(rms_mean)) / T.tensor(rms_std)

                feats = etp.mlp_extractor(cur_obs_T_norm.float())[0]
                value = etp.value_net(feats)

                # etp loss
                value_loss = -value

                # Trajectory losses
                b1_T = T.tensor(self.b1_pos, requires_grad=False)
                b2_T = T.tensor(self.b2_pos, requires_grad=False)

                # Barrier constraints
                barrier_loss_list = []
                for xy_T in traj_T:
                    barrier_loss_list.append(-(barrier_lf(xy_T, b1_T) + barrier_lf(xy_T, b2_T)) * 0.06)

                # End point stretched out as much as possible
                # final_pt_loss = mse_loss(traj_T[-1], T.tensor([13., 0.])) * 0.1
                final_pt_loss = -traj_T[-1][0] * 0.03 + T.square(traj_T[-1][1])

                # Minimize square distance between points
                inter_pt_loss_list = []
                for i in range(len(traj_T) - 1):
                    inter_pt_loss_list.append(mse_loss(dist_T(traj_T[i], traj_T[i + 1]), T.tensor(0.17)) * 3)

                total_loss = value_loss + T.stack(inter_pt_loss_list).sum() + final_pt_loss + T.stack(barrier_loss_list).sum()
                total_loss.backward()
                optim.step()
                optim.zero_grad()

                # execute one step
                action, _states = self.sb_model.predict(obs, deterministic=True)
                obs, reward, done, info = self.venv.step(action)
                self.venv.render()

                if done:
                    break

            # PLOT
            if it % 1 == 0:
                x, y = list(zip(*[t.detach().numpy() for t in traj_T]))
                line1.set_xdata(x)
                line1.set_ydata(y)
                figure.canvas.draw()
                figure.canvas.flush_events()

            if it % 100 == 0:
                print(f"Iter: {it}")

        optimized_traj = [pt.detach().numpy() for pt in traj_T]
        return optimized_traj

    def optimize_traj_exec_full(self, traj, n_iters=1):
        traj_len = 50
        # Load TEP
        tep = TEPMLP(obs_dim=traj_len * 2, act_dim=1)
        # tep = TEPRNN(n_waypts=len(traj), hid_dim=32, hid_dim_2=6)
        # tep = TEPTX(n_waypts=len(traj) * 2, embed_dim=36, num_heads=6, kdim=36)
        tep.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

        def clipped_mse(a, b):
            loss = T.minimum(T.sqrt(T.square(a[0] - b[0]) + T.square(a[1] - b[1])), T.tensor(0.5))
            return loss

        def flattened_mse(p1, p2):
            a = T.tensor(-1e2)
            c = T.tensor(.2)
            x = p1 - p2
            loss = (T.abs(a - 2.) / (a)) * (T.pow((T.square(x/c)/T.abs(a - 2)) + 1, 0.5 * a) - 1.)
            return loss

        def dist_T(a, b):
            return T.sqrt(T.square(a[0] - b[0]) + T.square(a[1] - b[1]))

        def calc_traj_loss(traj_T):
            traj_pairwise = traj_T.reshape((len(traj_T) // 2, 2))
            cur_traj_T_delta = T.concat((traj_T[:2], traj_T[2:] - traj_T[:-2]))

            # etp loss
            pred_rew = tep(cur_traj_T_delta)

            # Trajectory losses
            b1_T = T.tensor(self.b1_pos, requires_grad=False)
            b2_T = T.tensor(self.b2_pos, requires_grad=False)

            # Barrier constraints
            barrier_loss_list = []
            for xy_T in traj_pairwise:
                barrier_loss_list.append(-(barrier_lf(xy_T, b1_T) + barrier_lf(xy_T, b2_T)) * 0.06)
            barrier_loss_sum = T.stack(barrier_loss_list).sum()

            # End point stretched out as much as possible
            # final_pt_loss = mse_loss(traj_T[-1], T.tensor([13., 0.])) * 0.1
            final_pt_loss = -traj_pairwise[-1][0] * 0.2 + T.square(traj_pairwise[-1][1])

            # Minimize square distance between points
            inter_pt_loss_list = []
            for i in range(len(traj_pairwise) - 1):
                inter_pt_loss_list.append(mse_loss(dist_T(traj_pairwise[i], traj_pairwise[i + 1]), T.tensor(0.17)) * 3)

            total_loss = -pred_rew * 0.005 + T.stack(inter_pt_loss_list).sum() + final_pt_loss + barrier_loss_sum


            return total_loss

        barrier_lf = flattened_mse
        mse_loss = T.nn.MSELoss()

        traj_T = T.tensor(traj, dtype=T.float32, requires_grad=True).reshape(len(traj) * 2)
        #traj_T = T.nn.ParameterList([T.nn.Parameter(T.tensor(pt, dtype=T.float32, requires_grad=True)) for pt in traj[:traj_len]])
        #optim = T.optim.Adam(params=traj_T, lr=0.001)

        # quiver_plot = ax.quiver([],
        #           [],
        #           [],
        #           [],
        #           width=0.002,
        #           color=[1, 0, 0])

        # Plot
        figure, ax = plt.subplots(figsize=(14, 6))

        for it in range(n_iters):
            total_loss = calc_traj_loss(traj_T)

            traj_grad = T.autograd.grad(total_loss, traj_T, allow_unused=True)[0]

            hess = T.autograd.functional.hessian(calc_traj_loss, traj_T)
            hess_inv = T.linalg.inv(hess)

            # Plot gradient changed by hessian
            scaled_grad = hess_inv @ traj_grad
            traj_T = traj_T + 0.01 * scaled_grad

            # Plot gradient
            with T.no_grad():
                traj_reshaped = traj_T.reshape(len(traj_T) // 2, 2).detach().numpy()
                grad_reshaped = traj_grad.reshape(len(traj_T) // 2, 2).detach().numpy()
                scaled_grad_reshaped = scaled_grad.reshape(len(traj_T) // 2, 2).detach().numpy()

            #optim.step()
            #optim.zero_grad()

            # PLOT
            if it % 1 == 0:
                line1, = ax.plot(list(zip(*traj))[0], list(zip(*traj))[1], marker="o")
                ax.scatter([4, 6, 17], [.5, .5, 0], s=200, c=['r', 'r', 'w'])

                ax.quiver(traj_reshaped[:, 0],
                          traj_reshaped[:, 1],
                          grad_reshaped[:, 0],
                          grad_reshaped[:, 1],
                          width=0.001,
                          color=[1, 0, 0])

                ax.quiver(traj_reshaped[:, 0],
                          traj_reshaped[:, 1],
                          scaled_grad_reshaped[:, 0],
                          scaled_grad_reshaped[:, 1],
                          width=0.001,
                          color=[0, 0, 1])

                x, y = list(zip(*[t.detach().numpy() for t in traj_T.reshape((len(traj_T) // 2, 2))]))
                line1.set_xdata(x)
                line1.set_ydata(y)
                figure.canvas.draw()
                figure.canvas.flush_events()

                ax.clear()

            if it % 1 == 0:
                #print(f"Iter: {it}, total_loss: {total_loss.data}, tep_pred_rew: {pred_rew.data}, final_pt_loss: {final_pt_loss.data}, barrier_loss: {barrier_loss_sum.data}")
                print(f"Iter: {it}, total_loss: {total_loss.data}")

        optimized_traj = [pt.detach().numpy() for pt in traj_T]
        return optimized_traj

    def optimize_traj_dist(self, traj, n_iters=1):
        def explf(a, b):
            loss = T.clip(1. / (T.exp(10 * T.abs(a - b)) + 0.01), -1, 1)
            return loss

        def clipped_mse(a, b):
            loss = T.minimum(T.sqrt(T.square(a[0] - b[0]) + T.square(a[1] - b[1])), T.tensor(0.5))
            return loss

        def flattened_mse(p1, p2):
            a = T.tensor(-1e2)
            c = T.tensor(.2)
            x = p1 - p2
            loss = (T.abs(a - 2.) / (a)) * (T.pow((T.square(x/c)/T.abs(a - 2)) + 1, 0.5 * a) - 1.)
            return loss

        def dist_T(a, b):
            return T.sqrt(T.square(a[0] - b[0]) + T.square(a[1] - b[1]))

        barrier_lf = flattened_mse

        #traj_T = [T.tensor(pt, dtype=T.float32, requires_grad=True) for pt in traj]
        traj_T = T.nn.ParameterList([T.nn.Parameter(T.tensor(pt, dtype=T.float32, requires_grad=True)) for pt in traj])

        mse_loss = T.nn.MSELoss()
        optim = T.optim.Adam(params=traj_T, lr=0.03)

        # Plot
        # import matplotlib.pyplot as plt
        # plt.ion()
        # figure, ax = plt.subplots(figsize=(14, 6))
        # line1, = ax.plot(list(zip(*traj))[0], list(zip(*traj))[1], marker="o")
        # ax.scatter([4,6,17], [.5,.5, 0], s=200, c=['r', 'r', 'w'])
        #

        for it in range(n_iters):
            b1_T = T.tensor(self.b1_pos, requires_grad=False)
            b2_T = T.tensor(self.b2_pos, requires_grad=False)

            # Barrier constraints
            barrier_loss_list = []
            for xy_T in traj_T:
                barrier_loss_list.append(-(barrier_lf(xy_T, b1_T) + barrier_lf(xy_T, b2_T)) * 0.06)

            # End point stretched out as much as possible
            #final_pt_loss = mse_loss(traj_T[-1], T.tensor([13., 0.])) * 0.1
            final_pt_loss = -traj_T[-1][0] * 0.2 + T.square(traj_T[-1][1])

            # Minimize square distance between points
            inter_pt_loss_list = []
            for i in range(len(traj_T) - 1):
                inter_pt_loss_list.append(mse_loss(dist_T(traj_T[i], traj_T[i + 1]), T.tensor(0.17)) * 10)

            total_loss = T.stack(inter_pt_loss_list).sum() + final_pt_loss + T.stack(barrier_loss_list).sum()
            total_loss.backward()
            traj_T[0].grad.fill_(0)
            #T.nn.utils.clip_grad_norm_(traj_T, 0.001)
            optim.step()
            optim.zero_grad()

            #with T.no_grad():
            #    for pt in traj_T[1:]:
            #        pt -= pt.grad * 0.03

            #with T.no_grad():

            # PLOT
            # if it % 30 == 0:
            #     x, y = list(zip(*[t.detach().numpy() for t in traj_T]))
            #     line1.set_xdata(x)
            #     line1.set_ydata(y)
            #     figure.canvas.draw()
            #     figure.canvas.flush_events()

            if it % 100 == 0:
                print(f"Iter: {it}")

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
        y[N // 2 - 250:N // 2 + 250] = np.sin((x[N // 2 - 250:N // 2 + 250] - 1) * 3.1415 * 0.25)
        windowing_func = 5 / np.exp(1.0 * np.abs(x - 5))
        y = y * windowing_func

        #y = np.log(6 * np.abs(y) + 1) * np.sign(y) * 0.5

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.plot(x, y)
        # plt.show()
        # exit()

        # import matplotlib.pyplot as plt
        # x = np.linspace(0, 1, N)
        # plt.plot(x, 1. / np.exp(10 * np.abs(x - .5)))
        # plt.show()
        # exit()

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
