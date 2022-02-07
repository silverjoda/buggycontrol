import mujoco_py
import os
import time
import numpy as np
from multiprocessing import Lock
import yaml

class BuggySSTrajectoryTrainer:
    def __init__(self):
        pass

    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/buggy_ss_traj_trainer.yaml"), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def generate_random_action_vec(self):
        pass

    def gather_ss_dataset(self):
        # Make buggy env

        # loop:
        # generate new action trajetory
        # rollout env with given action traj
        # make dataset out of given traj
        # Save as npy dataset

        pass

    def train_imitator_on_dataset(self):
        # Load dataset
        # Prepare policy and training
        # Make imitation
        pass

    def visualize_imitator(self):
        pass

if __name__ == "__main__":
    bt = BuggySSTrajectoryTrainer()
    bt.gather_ss_dataset()
