import math as m

import yaml

from src.ctrl.model import *
from src.ctrl.mpc import *
from src.ctrl.simulator import make_simulator
from src.envs.buggy_env_mujoco import BuggyEnv
from src.utils import q2e

class ControlBuggyMPPI:
    def __init__(self, mppi_config, buggy_config):
        self.mppi_config = mppi_config
        self.buggy_config = buggy_config
        self.buggy_env_mujoco = BuggyEnv(self.buggy_config)
        self.dynamics_model = self.load_dynamics_model()

    def load_dynamics_model(self):
        pass

    def test_mppi(self, env, N=100, render=False):
        for _ in range(N):
            # New env and trajectory
            env.reset()

            waypts = [[3, 0], [-3, 0]]
            current_wp_idx = 0
            self.waypoints[:] = waypts[current_wp_idx]

            for _ in range(3000):
                # Predict using MPC
                u = self.mppi_predict(x)

                if np.sqrt((x[0] - self.waypoints[0]) ** 2 + (x[1] - self.waypoints[1]) ** 2) < 0.5:
                    current_wp_idx = int(not current_wp_idx)
                    self.waypoints[:] = waypts[current_wp_idx]
                    print("Visited", current_wp_idx, self.waypoints)

                if render:
                    env.render()

    def mppi_predict(self, state):
        pass

    def mppi_predict_mujoco(self, state):
        pass

    def make_mppi_rollouts(self, dynamics_model, n):
        pass

    def make_mppi_rollouts_mujoco(self, env, n):
        pass

    def evaluate_mppi_rollouts(self, rollouts):
        pass



if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "configs/control_buggy_mppi.yaml"), 'r') as f:
        mppi_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
        buggy_config = yaml.load(f, Loader=yaml.FullLoader)
    cbm = ControlBuggyMPPI(mppi_config, buggy_config)
