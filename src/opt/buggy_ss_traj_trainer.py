import os
import pickle

import numpy as np
import torch as T

from simplex_noise import SimplexNoise
from src.envs.buggy_env_mujoco import BuggyEnv
from src.policies import MLP, RNN
from src.utils import load_config


class BuggySSTrajectoryTrainer:
    def __init__(self):
        self.config = load_config(os.path.join(os.path.dirname(__file__), "configs/buggy_ss_traj_trainer.yaml"))
        self.noise = SimplexNoise(dim=2, smoothness=100, multiplier=2)

        self.x_file_path = os.path.join(os.path.dirname(__file__), "X.pkl")
        self.y_file_path = os.path.join(os.path.dirname(__file__), "Y.pkl")
        self.p_file_path = os.path.join(os.path.dirname(__file__), "P.pkl")

    def generate_random_action_vec(self):
        # noise -1,1
        rnd_action_vec = self.noise.gen_noise_seq()

        # Scale
        rnd_action_vec[:, 0] /= 2 + 0.5

        return rnd_action_vec

    def gather_ss_dataset(self):
        # Make buggy env
        env = BuggyEnv()

        X = []
        Y = []

        # loop:
        for i in range(self.config["n_traj"]):
            obs = env.reset()
            obs_list = []
            act_list = []

            for j in range(self.config["traj_len"]):
                obs_list.append(obs)

                # Get random action
                rnd_act = self.noise()
                act_list.append(rnd_act)

                obs, _, _, _ = env.step(rnd_act)

            # make dataset out of given traj
            x, y = self.make_trn_examples_from_traj(np.array(obs_list), np.array(act_list))

            X.extend(x)
            Y.extend(y)

        # Save as npy dataset
        X = np.concatenate(X)
        Y = np.concatenate(Y)

        pickle.dump(X, open(self.x_file_path, "wb"))
        pickle.dump(Y, open(self.y_file_path, "wb"))

    def make_trn_examples_from_traj(self, obs_list, act_list):
        X = []
        Y = []
        for current_state_idx in range(0, len(obs_list), self.config["traj_jump_dist"]):
            rnd_offset = np.random.randint(1, 20)

            current_state = obs_list[current_state_idx]

            # Iterate over trajectory and append every *dist* points
            current_anchor_idx = current_state_idx
            current_trajectory = []
            for ti in range(current_state_idx + rnd_offset, len(obs_list)):
                if self.get_dist_between_pts(obs_list[ti]["xy"], obs_list[current_anchor_idx]["xy"]):
                    # Transform wp to buggy frame
                    wp_buggy_frame = self.transform_wp_to_buggy_frame(obs_list[ti])

                    if self.config["add_act_to_traj"]:
                        current_trajectory.append([*wp_buggy_frame, obs_list[ti]["act"]])
                    else:
                        current_trajectory.append([*wp_buggy_frame])
                    current_anchor_idx = ti
                if len(current_trajectory) > self.config["n_traj_pts"]:
                    pass

            X.append([*current_state["obs"], *current_trajectory])
            Y.append([current_state["act"]])

        return X, Y

    def transform_wp_to_buggy_frame(self, wp, buggy_obs):
        wp_x = wp[0] - buggy_obs["pos"][0]
        wp_y = wp[1] - buggy_obs["pos"][1]
        theta = buggy_obs["theta"]
        t_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        wp_buggy = np.matmul(t_mat, np.array([wp_x, wp_y]))
        return wp_buggy

    def get_dist_between_pts(self, p1, p2):
        return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))

    def train_imitator_on_dataset(self):
        # Load dataset
        X = pickle.load(open(self.x_file_path, "rb"))
        Y = pickle.load(open(self.y_file_path, "rb"))

        # Prepare policy and training
        policy = MLP(obs_dim=X.shape[1], act_dim=2)
        optim = T.optim.Adam(params=policy.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()

        for i in range(self.config["trn_iters"]):
            rnd_indeces = np.random.choice(np.arange(len(X)), self.config["batchsize"], replace=False)
            x = X[rnd_indeces]
            y = Y[rnd_indeces]
            y_ = policy(x)
            loss = lossfun(y_, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 50 == 0:
                print("Iter {}/{}, loss: {}".format(i, self.config['iters'], loss.data))
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
            policy_loss = lossfun(y_, y)

            total_loss = policy_loss + param_est_loss
            total_loss.backward()

            if i % self.config["recurrent_batchsize"] == 0:
                policy_optim.step()
                estimator_optim.step()
                policy_optim.zero_grad()
                estimator_optim.zero_grad()

            if i % 50 == 0:
                print("Iter {}/{}, loss: {}".format(i, self.config['iters'], loss.data))
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
