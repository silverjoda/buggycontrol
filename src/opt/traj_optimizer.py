from enum import Enum
import torch as T
import os
import yaml
from src.utils import load_config

class TrajOptMode(Enum):
    WP_ONLY = 1
    THR_ONLY = 2
    WP_AND_THR = 3

class TrajOptimizer:
    def __init__(self, ETP):
        self.ETP = ETP
        self.config = load_config(os.path.join(os.path.dirname(__file__), "configs/traj_optimizer.yaml"))
        self.x_indeces, self.y_indeces, self.xy_indeces, self.thr_indeces = self.calculate_trajectory_indeces()

    def calculate_trajectory_indeces(self):
        if self.config["thr_in_traj"]:
            x_indeces = range(self.config["state_dim"], 3, self.config["n_traj_pts"] * 3)
            y_indeces = range(self.config["state_dim"] + 1, 3, self.config["n_traj_pts"] * 3)
            thr_indeces = range(self.config["state_dim"] + 2, 3, self.config["n_traj_pts"] * 3)
        else:
            x_indeces = range(self.config["state_dim"], 2, self.config["n_traj_pts"] * 2)
            y_indeces = range(self.config["state_dim"] + 1, 2, self.config["n_traj_pts"] * 2)
            thr_indeces = None
        xy_indeces = list(x_indeces) + list(y_indeces)
        return x_indeces, y_indeces, xy_indeces, thr_indeces

    def optimize_traj(self, obs, mode=TrajOptMode.WP_ONLY):
        # First 4 vars are state, rest are trajectory (x,y,thr ... repeated)
        obs_T = T.tensor(obs, requires_grad=True)

        for i in range(self.config["opt_iters"]):
            t_pred = self.ETP(obs_T)
            t_pred.backward()

            # Apply grads
            if mode==TrajOptMode.WP_ONLY or mode==TrajOptMode.WP_AND_THR:
                for i in self.xy_indeces:
                    obs_T[i] -= obs_T[i].grad * self.config["traj_grad_update_rate"]
            if mode==TrajOptMode.THR_ONLY or mode==TrajOptMode.WP_AND_THR:
                for i in self.thr_indeces:
                    obs_T[i] -= obs_T[i].grad * self.config["traj_grad_update_rate"]

            obs_T.grad.fill_(0.)

