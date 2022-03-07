
import numpy as np

from src.envs.buggy_env_mujoco import BuggyEnv
import yaml
from casadi import *
import do_mpc

from src.ctrl.model import make_model
from src.ctrl.mpc import make_mpc
from src.ctrl.simulator import make_simulator

class ControlBuggyMPC:
    def __init__(self, config):
        self.config = config

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/control_buggy_mpc.yaml"), 'r') as f:
            buggy_config = yaml.load(f, Loader=yaml.FullLoader)
        self.buggy_env_mujoco = BuggyEnv(buggy_config)

        self.model, self.mpc, self.simulator = self.setup_mpc()

        self.test_mpc(self.buggy_env_mujoco, self.model, self.simulator, self.mpc)

    def setup_mpc(self):
        model = make_model()
        mpc = make_mpc(model)
        simulator = make_simulator(model)
        return model, mpc, simulator

    def get_partial_mpc_state(self, obs_dict):
        #velocity =
        pass

    def test_mpc(self, env, model, simulator, N=100, render=True, print_rew=False):
        for _ in range(N):
            # New env and trajectory
            env.reset()

            # Setup new mpc cost function with target traj
            mpc = make_mpc(model, env.engine.wp_list)
            simulator.reset_history()

            episode_rew = 0

            # Slip angle, velocity, yaw rate, x, y, phi
            x0 = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)
            simulator.x0 = x0
            x = x0
            while True:
                # Predict using MPC
                u = mpc.make_step(x)
                obs, reward, done, info = env.step(u)
                obs_dict = env.get_obs_dict()

                # Get next side slip angle from simulator
                y_next = simulator.make_step(u)
                s_b = y_next[0]

                # Make next mpc state
                x_part = self.get_partial_mpc_state(obs_dict)
                x = np.array([s_b, *x_part]).reshape(-1, 1)

                if render:
                    env.render()
                if done:
                    if print_rew:
                        print(episode_rew)
                    break

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../opt/configs/train_buggy_a2c.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cbm = ControlBuggyMPC(config)
