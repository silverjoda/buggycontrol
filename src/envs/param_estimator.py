from src.policies import RNN
import torch as T
import os
import numpy as np

class ParamEstimator:
    def __init__(self, config):
        self.config = config
        self.param_estimator = RNN(self.config["state_dim"], self.config["latent_dim"])
        self.param_estimator_optim = T.optim.Adam(params=self.param_estimator.parameters(), lr=self.config["param_estimator_lr"])
        self.loss_fun = T.nn.MSELoss()
        self.hidden_state = None
        self.dataset = []
        self.episode_transitions = []

    def load_param_estimator(self):
        self.param_estimator.load_state_dict(T.load("agents/param_estimator.p"), strict=False)

    def save_param_estimator(self):
        T.save(self.param_estimator.state_dict(), "agents/param_estimator.p")

    def update_param_estimator(self):
        if len(self.dataset) < self.config["param_batchsize"]: return

        # Sample random batch
        X, P = self.sample_random_batch(self.config["param_batchsize"])

        # Make forward pass
        P_ = self.param_estimator(X)

        # Backward pass
        loss = self.loss_fun(P_, P)
        loss.backward()
        self.param_estimator_optim.step()
        self.param_estimator_optim.zero_grad()

        return loss.data

    def sample_random_batch(self, batchsize):
        rnd_indeces = np.random.choice(np.arange(len(self.dataset), batchsize, replace=False))

        P_list = []
        X_list = []
        for r_i in rnd_indeces:
            X_list.append(self.dataset[r_i][1])
            P_list.append(self.dataset[r_i][0])

        X = T.tensor(X_list).unsqueeze(0)
        P = T.tensor(P_list)

        return X, P

    def step_param_estimator(self, params, obs, act):
        self.episode_transitions.append([params, obs, act])
        obs_T = T.tensor(obs).reshape([1, 1, -1])
        out, self.hidden_state = self.param_estimator(obs_T, self.hidden_state)
        return out.detach().numpy()

    def clear_param_estimator(self):
        self.hidden_state = None
        self.dataset.append(self.episode_transitions)
        self.episode_transitions = []

    def add_trajectory(self, param_vec, transition_list):
        self.dataset.append((param_vec, transition_list))