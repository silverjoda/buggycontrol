
import numpy as np

from src.envs.buggy_env_mujoco import BuggyEnv
import yaml
from casadi import *
import do_mpc

from src.ctrl.model import make_bicycle_model, make_singletrack_model
from src.ctrl.mpc import make_mpc_singletrack, make_mpc_bicycle
from src.ctrl.simulator import make_simulator
import math as m

class ControlBuggyMPC:
    def __init__(self, config):
        self.config = config

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
            buggy_config = yaml.load(f, Loader=yaml.FullLoader)
        self.buggy_env_mujoco = BuggyEnv(buggy_config)

        self.model = make_bicycle_model()
        self.mpc = make_mpc_bicycle(self.model)
        self.simulator = make_simulator(self.model)

        self.test_mpc_bicycle(self.buggy_env_mujoco, self.model, self.mpc, self.simulator)

    def q2e(self, w, x, y, z):
        pitch = -m.asin(2.0 * (x * z - w * y))
        roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
        yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
        return (roll, pitch, yaw)

    def get_mpc_state_singletrack(self, obs_dict):
        # beta, velocity (vec), yaw rate, x, y, phi
        vel = obs_dict["vel"]
        beta = np.atan2(vel[1], vel[0])
        vel_vec = np.sqrt(np.square(vel[0:2]).sum())
        yaw_rate = obs_dict["ang_vel"][2]
        x_pos, y_pos = obs_dict["pos"][0:2]
        phi = self.q2e(*obs_dict["ori_q"])
        return beta, vel_vec, yaw_rate, x_pos, y_pos, phi

    def get_mpc_state_bicycle(self, obs_dict):
        # beta, velocity (vec), yaw rate, x, y, phi
        x_pos, y_pos = obs_dict["pos"][0:2]
        _, _, phi = self.q2e(*obs_dict["ori_q"])
        x_vel, y_vel = obs_dict["vel"][0:2]
        yaw_rate = obs_dict["ang_vel"][2]
        return x_pos, y_pos, phi, x_vel, y_vel, yaw_rate

    def test_mpc_singletrack(self, env, model, mpc, simulator, N=100, render=True, print_rew=False):
        for _ in range(N):
            # New env and trajectory
            env.reset()
            simulator.reset_history()

            # Slip angle, velocity, yaw rate, x, y, phi
            x0 = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)
            simulator.x0 = x0
            x = x0

            episode_rew = 0
            while True:
                # Predict using MPC
                u = mpc.make_step(x)
                obs, reward, done, info = env.step(u)
                obs_dict = env.get_obs_dict()

                # Get next side slip angle from simulator
                y_next = simulator.make_step(u)
                s_b = y_next[0]

                # Make next mpc state
                x_part = self.get_mpc_state(obs_dict)
                x = np.array([s_b, *x_part]).reshape(-1, 1)

                if render:
                    env.render()
                if done:
                    if print_rew:
                        print(episode_rew)
                    break

    def test_mpc_bicycle(self, env, model, mpc, simulator, N=100, render=True, print_rew=False):
        for _ in range(N):
            # New env and trajectory
            env.reset()
            simulator.reset_history()

            x0 = np.array([0, 0, 0, 0.01, 0.01, 0]).reshape(-1, 1)
            simulator.x0 = x0
            mpc.x0 = x0
            mpc.set_initial_guess()
            x = x0

            use_mujoco = False

            episode_rew = 0
            for i in range(1000):
                # Predict using MPC
                u = mpc.make_step(x)

                if use_mujoco:
                    obs, reward, done, info = env.step(u)
                    obs_dict = env.get_obs_dict()

                    # Make next mpc state
                    x = self.get_mpc_state_bicycle(obs_dict)
                    x = np.array(x).reshape(-1, 1)
                else:
                    # Get next side slip angle from simulator
                    y_next = simulator.make_step(u)
                    env.set_state({"x_pos" : y_next[0],
                                   "y_pos" : y_next[1],
                                   "phi" : y_next[2]})

                if render:
                    env.render()


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../opt/configs/train_buggy_a2c.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cbm = ControlBuggyMPC(config)
