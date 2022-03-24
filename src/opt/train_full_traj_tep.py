# MLP and Transformer training on whole trajectory

import os

import matplotlib.pyplot as plt
import numpy as np
import torch as T
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor

from src.envs.buggy_env_mujoco import BuggyEnv
from src.policies import TEPTX, TEPMLP, TEPRNN, TEPRNN2
from src.utils import load_config

plt.ion()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
T.set_num_threads(8)

class TEPDatasetMaker:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/train_full_traj_tep.yaml"), 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy_ID = "TRN"
        self.env_config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
        self.env, self.venv, self.sb_model = self.load_model_and_env(self.env_config)
        self.n_dataset_pts = self.config["n_dataset_pts"]
        self.max_num_wp = self.config["max_num_wp"]

        dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/buggy_tep_dataset")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.x_file_path = os.path.join(dir_path, "X.npy")
        self.y_file_path = os.path.join(dir_path, "Y.npy")

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

    def make_dataset(self, render=True):
        print("Starting dataset creation")
        obs_list = []
        rew_list = []
        for i in range(self.n_dataset_pts):
            obs = self.venv.reset()
            episode_rew = 0
            step_ctr = 0
            while True:
                step_ctr += 1
                action, _states = self.sb_model.predict(obs, deterministic=True)
                obs, reward, done, info = self.venv.step(action)
                episode_rew += self.venv.get_original_reward()
                if render:
                    self.venv.render()
                if done:
                    break
            obs_list.append([item for sublist in self.env.engine.wp_list[:self.max_num_wp] for item in sublist])
            rew_list.append(episode_rew)

            if i % 10 == 0:
                print(f"Iter: {i}/{self.n_dataset_pts}")

        obs_arr = np.array(obs_list)
        rew_arr = np.array(rew_list)

        np.save(self.x_file_path, obs_arr)
        np.save(self.y_file_path, rew_arr)

        return obs_arr, rew_arr

    def get_successive_angle_representation(self, X):
        X_new = np.zeros((X.shape[0], X.shape[1] // 2))

        X_new[:, 0] = np.arctan2(X[:, 1], X[:, 0])
        for i in range(1, X_new.shape[1]):
            X_new[:, i] = np.arctan2(X[:, i * 2 + 1] - X[:, (i - 1) * 2 + 1], X[:, i * 2] - X[:, (i - 1) * 2])
        return X_new

    def get_delta_representation(self, X):
        X_new = np.zeros_like(X)
        X_new[:, :2] = X[:, :2]
        X_new[:, 2:] = X[:, 2:] - X[:, :-2]
        return X_new

    def train_tep(self):
        # Load dataset
        X = np.load(self.x_file_path, allow_pickle=True)
        Y = np.load(self.y_file_path)

        # Change X to relative coordinates
        #X_new = self.get_delta_representation(X)

        # Change to successive angle representation
        X_new = self.get_successive_angle_representation(X)

        # Prepare policy and training
        emb_dim = 36
        policy = TEPMLP(obs_dim=X_new.shape[1], act_dim=1, n_hidden=1)
        #policy = TEPRNN(n_waypts=X_new.shape[1] // 2, hid_dim=64, hid_dim_2=32, num_layers=1, bidirectional=False)
        #policy = TEPRNN2(n_waypts=X_new.shape[1], hid_dim=64, hid_dim_2=32, num_layers=1, bidirectional=False)
        #policy = TEPTX(n_waypts=X.shape[1], embed_dim=emb_dim, num_heads=6, kdim=36)
        policy_optim = T.optim.Adam(params=policy.parameters(),
                                    lr=self.config['policy_lr'],
                                    weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()

        for i in range(self.config["trn_iters"]):
            rnd_start_idx = np.random.randint(low=0, high=len(X_new) - self.config["batchsize"] - 1)
            x = X_new[rnd_start_idx:rnd_start_idx + self.config["batchsize"]]
            y = Y[rnd_start_idx:rnd_start_idx + self.config["batchsize"]]

            x_T = T.tensor(x, dtype=T.float32)
            y_T = T.tensor(y, dtype=T.float32)

            y_ = policy(x_T)
            policy_loss = lossfun(y_, y_T)

            total_loss = policy_loss
            total_loss.backward()

            policy_optim.step()
            policy_optim.zero_grad()

            if i % 50 == 0:
                print(
                    "Iter {}/{}, policy_loss: {}".format(i, self.config['trn_iters'], policy_loss.data))
        print("Done training, saving model")
        if not os.path.exists("agents"):
            os.makedirs("agents")
        T.save(policy.state_dict(), "agents/full_traj_tep.p")

    def train_tep_1step_grad(self):
        # Load pretrained tep
        tep = TEPMLP(obs_dim=50, act_dim=1)
        tep.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

        # Core dataset
        X = np.load(self.x_file_path, allow_pickle=True)
        Y = np.load(self.y_file_path)

        # Change to successive angle representation
        X = T.tensor(self.get_successive_angle_representation(X), dtype=T.float32)
        Y = T.tensor(Y, dtype=T.float32)

        # Make updated dataset
        X_ud = X.clone().detach()
        Y_ud = Y.clone().detach()

        # Prepare policy and training
        policy_optim = T.optim.Adam(params=tep.parameters(),
                                    lr=self.config['policy_lr'],
                                    weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()

        n_epochs = 100
        n_data = len(X)
        for ep in range(n_epochs):
            # Do single epoch
            rnd_indeces = np.random.shuffle(np.arange(n_data))
            for i in range(0, n_data, self.config["batchsize"] // 2):
                # Halfbatch from core dataset
                x_c = X[i:i + self.config["batchsize"] // 2]
                y_c = Y[i:i + self.config["batchsize"] // 2]

                # Halfbatch from ud dataset
                x_ud = X_ud[rnd_indeces[i:i + self.config["batchsize"] // 2]]
                y_ud = Y_ud[rnd_indeces[i:i + self.config["batchsize"] // 2]]

                # Combine datasets
                x = T.concat((x_c, x_ud), dim=0)
                y = T.concat((y_c, y_ud), dim=0)

                y_ = tep(x)
                policy_loss = lossfun(y_, y)

                total_loss = policy_loss
                total_loss.backward()

                policy_optim.step()
                policy_optim.zero_grad()

                if i % 50 == 0:
                    print(
                        "Iter {}/{}, policy_loss: {}".format(i, self.config['trn_iters'], policy_loss.data))

            # Update ud dataset
            for t_idx in range(len(X)):
                X_ud[t_idx] = self.perform_grad_update_full_traj(X[t_idx], tep)

        print("Done training, saving model")
        if not os.path.exists("agents"):
            os.makedirs("agents")
        T.save(tep.state_dict(), "agents/full_traj_tep_1step.p")

    def test_tep(self):
        tep = TEPMLP(obs_dim=50, act_dim=1)
        tep.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

        N_eval = 100
        render = True
        for i in range(N_eval):
            obs = self.env.reset()
            obs = self.venv.normalize_obs(obs)

            episode_rew = 0
            step_ctr = 0
            while True:
                step_ctr += 1
                action, _states = self.sb_model.predict(obs.unsqueeze(0), deterministic=True)
                obs, reward, done, info = self.env.step(action)
                obs = self.venv.normalize_obs(obs)
                episode_rew += reward

                if render:
                    self.env.render()
                if self.env.engine.cur_wp_idx > 50 or step_ctr > 700:
                    break

            # Make tep prediction
            traj = self.env.engine.wp_list
            traj_sar = self.xy_to_sar(traj[:50])
            traj_T_sar = T.tensor(traj_sar, dtype=T.float32)
            tep_pred = tep(traj_T_sar)

            print(f"Reward gathered: {episode_rew}, reward predicted: {tep_pred}, time taken: {step_ctr * 0.01}")

    def test_tep_full(self):
        tep_def = TEPMLP(obs_dim=50, act_dim=1)
        tep_def.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

        tep_1step = TEPMLP(obs_dim=50, act_dim=1)
        tep_1step.load_state_dict(T.load("agents/full_traj_tep_1step.p"), strict=False)

        N_eval = 100
        render = True
        for i in range(N_eval):
            obs = self.env.reset()
            obs = self.venv.normalize_obs(obs)

            # Rollout on randomly generated traj
            episode_rew = 0
            step_ctr = 0
            while True:
                step_ctr += 1
                action, _states = self.sb_model.predict(obs.unsqueeze(0), deterministic=True)
                obs, reward, done, info = self.env.step(action)
                obs = self.venv.normalize_obs(obs)
                episode_rew += reward

                if render:
                    self.env.render()
                if self.env.engine.cur_wp_idx > 50 or step_ctr > 700:
                    break

            # Make tep prediction
            traj = self.env.engine.wp_list
            traj_sar = self.xy_to_sar(traj[:50])
            traj_T_sar = T.tensor(traj_sar, dtype=T.float32)
            tep_def_pred = tep_def(traj_T_sar)
            tep_1step_pred = tep_1step(traj_T_sar)

            # Make trajectory update
            traj_T_sar_ud = self.perform_grad_update_full_traj(traj_T_sar)
            traj_T_ud = self.sar_to_xy(traj_T_sar_ud)
            self.env.reset()
            self.env.engine.wp_list = list(traj_T_ud)

            # Rollout on randomly generated traj
            episode_rew_1step = 0
            step_ctr_1step = 0
            while True:
                step_ctr_1step += 1
                action, _states = self.sb_model.predict(obs.unsqueeze(0), deterministic=True)
                obs, reward, done, info = self.env.step(action)
                obs = self.venv.normalize_obs(obs)
                episode_rew_1step += reward

                if render:
                    self.env.render()
                if self.env.engine.cur_wp_idx > 50 or step_ctr_1step > 700:
                    break

            # Make tep prediction on updated traj
            traj = self.env.engine.wp_list
            traj_sar = self.xy_to_sar(traj[:50])
            traj_T_sar = T.tensor(traj_sar, dtype=T.float32)
            tep_def_pred_1step = tep_def(traj_T_sar)
            tep_1step_pred_1step = tep_1step(traj_T_sar)

            print(f"Reward gathered on initial traj: {episode_rew}, reward predicted: {tep_def_pred}, reward predicted by tep_1step: {tep_1step_pred}, time taken: {step_ctr * 0.01}")
            print(f"Reward gathered on 1step traj: {episode_rew_1step}, reward predicted: {tep_def_pred_1step}, reward predicted by tep_1step: {tep_1step_pred_1step},  time taken: {step_ctr_1step * 0.01}")
            print("------------------------------------------------------------------------------------------------------------")

    def perform_grad_update_full_traj(self, traj, tep):
        # etp loss
        loss = -tep(traj)

        traj_grad = T.autograd.grad(loss, traj, allow_unused=True)[0]

        hess = T.autograd.functional.hessian(loss, traj)
        hess_inv = T.linalg.inv(hess)

        scaled_grad = hess_inv @ traj_grad

        with T.no_grad():
            traj = traj - 0.05 * scaled_grad

        return traj

    def xy_to_sar(self, X):
        X_new = np.zeros(len(X))
        X_new[0] = np.arctan2(X[0][1], X[0][0])
        for i in range(1, len(X)):
            X_new[i] = np.arctan2(X[i][1] - X[i - 1][1], X[i][0] - X[i - 1][0])
        return X_new

    def sar_to_xy(self, X):
        wp_dist = 0.17
        pd_x = T.cumsum(T.cos(X) * wp_dist, dim=0).unsqueeze(1)
        pd_y = T.cumsum(T.sin(X) * wp_dist, dim=0).unsqueeze(1)
        traj_T = T.concat((pd_x, pd_y), dim=1)
        return traj_T

if __name__ == "__main__":
    tm = TEPDatasetMaker()
    #tm.make_dataset(render=False)
    #tm.train_tep()
    #tm.train_tep_1step_grad()
    tm.test_tep()
