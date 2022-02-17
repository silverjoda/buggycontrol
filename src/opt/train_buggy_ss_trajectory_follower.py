import os
import pickle
import time

import numpy as np
import torch as T

from src.opt.simplex_noise import SimplexNoise
from src.envs.buggy_env_mujoco import BuggyEnv
from src.policies import MLP, RNN
from src.utils import load_config
import math as m
from imitation.data.types import Trajectory

class BuggySSTrajectoryTrainer:
    def __init__(self):
        self.config = load_config(os.path.join(os.path.dirname(__file__),
                                               "configs/train_buggy_ss_trajectory_follower.yaml"))
        self.noise = SimplexNoise(dim=2, smoothness=300, multiplier=1.7)

        dir_path = os.path.join(os.path.dirname(__file__), "supervised_trajs")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.x_file_path = os.path.join(dir_path, "X.pkl")
        self.y_file_path = os.path.join(dir_path, "Y.pkl")
        self.p_file_path = os.path.join(dir_path, "P.pkl")

        self.traj_file_path = os.path.join(dir_path, "traj.pkl")

    def generate_random_action_vec(self):
        # noise -1,1
        rnd_action_vec = self.noise.gen_noise_seq()

        # Scale
        rnd_action_vec[:, 0] /= 2 + 0.5

        return rnd_action_vec

    def gather_ss_dataset(self):
        # Make buggy env
        config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
        env = BuggyEnv(config)

        X = []
        Y = []
        P = []

        # loop:
        for i in range(self.config["n_traj"]):
            _ = env.reset()
            obs_list = []
            act_list = []

            for j in range(self.config["traj_len"]):
                obs_dict = env.get_obs_dict()
                obs_list.append(obs_dict)

                # Get random action
                rnd_act = self.noise()

                # Condition act (turn, throttle)
                rnd_act[0] *= 0.4
                rnd_act[1] += 0.5

                act_list.append(rnd_act)

                _, _, _, _ = env.step(rnd_act)

                # DEBUG
                # env.render()
                # time.sleep(0.01)

            # make dataset out of given traj
            x, y = self.make_trn_examples_from_traj(env, obs_list, act_list)

            X.extend(x)
            Y.extend(y)
            P.extend([env.scaled_random_params] * len(x))

        # Save as npy dataset
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        P_arr = np.array(P)

        pickle.dump(X_arr, open(self.x_file_path, "wb"))
        pickle.dump(Y_arr, open(self.y_file_path, "wb"))
        pickle.dump(P_arr, open(self.p_file_path, "wb"))

        traj_list = []
        for x, y in zip(X, Y):
            traj = Trajectory(obs=np.array(x), acts=y, infos=None)
            traj_list.append(traj)
        pickle.dump(traj_list, open(self.traj_file_path, "wb"))

    def make_trn_examples_from_traj(self, env, obs_list, act_list):
        X = []
        Y = []
        for current_state_idx in range(0, len(obs_list) - 200, self.config["traj_jump_dist"]):
            rnd_offset = np.random.randint(1, 20)

            # Iterate over trajectory and append every *dist* points
            current_anchor_idx = current_state_idx
            current_trajectory = []
            for ti in range(current_state_idx + rnd_offset, len(obs_list)):
                if self.get_dist_between_pts(obs_list[ti]["pos"], obs_list[current_anchor_idx]["pos"]) > self.config["traj_sampling_dist"]:
                    wp = obs_list[ti]["pos"][0:2]
                    current_trajectory.append(wp)
                    current_anchor_idx = ti
                if len(current_trajectory) == self.config["n_traj_pts"]: break

            if len(current_trajectory) == self.config["n_traj_pts"]:
                state_vec = [obs_list[current_state_idx]["vel"][0],
                             obs_list[current_state_idx]["vel"][1],
                             obs_list[current_state_idx]["ang_vel"][2]]
                current_trajectory_buggy_frame = env.engine.transform_wp_to_buggy_frame(current_trajectory,
                                                                                        obs_list[current_state_idx][
                                                                                            "pos"],
                                                                                        obs_list[current_state_idx][
                                                                                            "ori_q"])

                X.append(state_vec + list(current_trajectory_buggy_frame.reshape(-1)))
                Y.append(act_list[current_state_idx])

            # DEBUG
            # env.engine.set_wp_visuals_externally(current_trajectory)
            # env.engine.step([0,0])
            # env.render()
            # time.sleep(3)
        # DEBUG
        #exit()

        return X, Y


    def get_dist_between_pts(self, p1, p2):
        return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))

    def train_imitator_on_dataset(self):
        # Load dataset
        X = pickle.load(open(self.x_file_path, "rb"))
        Y = pickle.load(open(self.y_file_path, "rb"))

        # Prepare policy and training
        policy = MLP(obs_dim=X.shape[1], act_dim=2)
        optim = T.optim.Adam(params=policy.parameters(), lr=self.config['policy_lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()

        for i in range(self.config["trn_iters"]):
            rnd_indeces = np.random.choice(np.arange(len(X)), self.config["batchsize"], replace=False)
            x = T.tensor(X[rnd_indeces], dtype=T.float32)
            y = T.tensor(Y[rnd_indeces], dtype=T.float32)
            y_ = policy(x)
            loss = lossfun(y_, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 50 == 0:
                print("Iter {}/{}, loss: {}".format(i, self.config['trn_iters'], loss.data))
        print("Done training, saving model")
        if not os.path.exists("agents"):
            os.makedirs("agents")
        T.save(policy.state_dict(), "agents/buggy_imitator.p")

    def train_imitator_with_param_estimator_on_dataset(self):
        # Load dataset
        X = pickle.load(open(self.x_file_path, "rb"))
        Y = pickle.load(open(self.y_file_path, "rb"))
        P = pickle.load(open(self.p_file_path, "rb"))

        # Prepare policy and training
        policy = MLP(obs_dim=X.shape[1] + self.config["n_latent_params"], act_dim=2)
        policy_optim = T.optim.Adam(params=policy.parameters(), lr=self.config['policy_lr'], weight_decay=self.config['w_decay'])

        param_estimator = RNN(obs_dim=X.shape[1], act_dim=self.config["n_latent_params"], hid_dim=64)
        estimator_optim = T.optim.Adam(params=param_estimator.parameters(), lr=self.config['param_estimator_lr'],
                                    weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()

        for i in range(self.config["trn_iters"]):
            rnd_start_idx = np.random.randint(low=0, high=len(X) - self.config["batchsize"] - 1)
            x = X[rnd_start_idx:rnd_start_idx+self.config["batchsize"]]
            y = Y[rnd_start_idx:rnd_start_idx+self.config["batchsize"]]
            p = P[rnd_start_idx:rnd_start_idx+self.config["batchsize"]]

            x_T = T.tensor(x)
            y_T = T.tensor(y)
            p_T = T.tensor(p)

            p_ = param_estimator(x)
            param_est_loss = lossfun(p_, p)

            y_ = policy(T.concat([x_T, p_T], dim=1))
            policy_loss = lossfun(y_, y_T)

            total_loss = policy_loss + param_est_loss
            total_loss.backward()

            if i % self.config["recurrent_batchsize"] == 0:
                policy_optim.step()
                estimator_optim.step()
                policy_optim.zero_grad()
                estimator_optim.zero_grad()

            if i % 50 == 0:
                print("Iter {}/{}, policy_loss: {}, param_est_loss: {}".format(i, self.config['iters'], policy_loss.data, param_est_loss.data))
        print("Done training, saving model")
        if not os.path.exists("agents"):
            os.makedirs("agents")
        T.save(policy.state_dict(), "agents/buggy_imitator.p")
        T.save(param_estimator.state_dict(), "agents/buggy_latent_param_estimator.p")

    def visualize_imitator(self):
        pass

if __name__ == "__main__":
    bt = BuggySSTrajectoryTrainer()
    bt.gather_ss_dataset()
    #bt.train_imitator_on_dataset()