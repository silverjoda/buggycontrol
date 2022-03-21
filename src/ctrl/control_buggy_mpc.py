import math as m

import yaml

from src.ctrl.model import *
from src.ctrl.mpc import *
from src.ctrl.simulator import make_simulator
from src.envs.buggy_env_mujoco import BuggyEnv
from src.utils import q2e

class ControlBuggyMPC:
    def __init__(self, config):
        self.config = config

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
            buggy_config = yaml.load(f, Loader=yaml.FullLoader)
        self.buggy_env_mujoco = BuggyEnv(buggy_config)

        self.waypoints = [3, 3]

        #self.model = make_bicycle_model()
        self.model = make_singletrack_model([3.2, 1.5, 0.12, 0.11, 0.056, 1.13, 9.9, 1.88, 0.1, 1.7, 10, 1.69, -0.43, 144])
        #self.model = make_linmod_model()

        #self.mpc = make_mpc_bicycle(self.model, waypoints=self.waypoints)
        self.mpc = make_mpc_singletrack(self.model, waypoints=self.waypoints)
        #self.mpc = make_mpc_linmod_hybrid(self.model)
        self.simulator = make_simulator(self.model)

        #self.test_mpc_bicycle(self.buggy_env_mujoco, self.model, self.mpc, self.simulator)
        self.test_mpc_singletrack(self.buggy_env_mujoco, self.model, self.mpc, self.simulator)
        #self.test_mpc_linmod(self.buggy_env_mujoco, self.model, self.mpc, self.simulator)

    def get_mpc_state_singletrack(self, obs_dict):
        # beta, velocity (vec), yaw rate, x, y, phi
        vel = obs_dict["vel"]
        beta = np.atan2(vel[1], vel[0])
        vel_vec = np.sqrt(np.square(vel[0:2]).sum())
        yaw_rate = obs_dict["ang_vel"][2]
        x_pos, y_pos = obs_dict["pos"][0:2]
        _, _, phi = q2e(*obs_dict["ori_q"])
        return beta, vel_vec, yaw_rate, x_pos, y_pos, phi

    def get_mpc_state_bicycle(self, obs_dict):
        # beta, velocity (vec), yaw rate, x, y, phi
        x_pos, y_pos = obs_dict["pos"][0:2]
        _, _, phi = q2e(*obs_dict["ori_q"])
        x_vel, y_vel = obs_dict["vel"][0:2]
        yaw_rate = obs_dict["ang_vel"][2]
        return x_pos, y_pos, phi, x_vel, y_vel, yaw_rate

    def test_mpc_singletrack(self, env, model, mpc, simulator, N=100, render=True, print_rew=False):
        for _ in range(N):
            # New env and trajectory
            env.reset()
            simulator.reset_history()

            # Slip angle, velocity, yaw rate, x, y, phi
            x0 = np.array([0.02, 1.0, 0.03, 0, 0, 0]).reshape(-1, 1)
            simulator.x0 = x0
            mpc.x0 = x0
            mpc.set_initial_guess()
            x = x0

            use_mujoco = False
            waypts = [[3, 0], [-3, 0]]
            current_wp_idx = 0
            self.waypoints[:] = waypts[current_wp_idx]
            for i in range(1000):
                # Predict using MPC
                u = mpc.make_step(x)

                if use_mujoco:
                    obs, reward, done, info = env.step(u)
                    obs_dict = env.get_obs_dict()

                    # Make next mpc state
                    x = self.get_mpc_state_singletrack(obs_dict)
                    x = np.array(x).reshape(-1, 1)
                else:
                    # Get next side slip angle from simulator
                    #u = np.array([0.1, 144]).reshape(-1, 1)
                    for _ in range(5):
                        x = simulator.make_step(u)

                    env.set_external_state({"x_pos": x[3],
                                            "y_pos": x[4],
                                            "phi": x[5]})

                if np.sqrt((x[3] - self.waypoints[0]) ** 2 + (x[4] - self.waypoints[1]) ** 2) < 0.5:
                    current_wp_idx = int(not current_wp_idx)
                    self.waypoints[:] = waypts[current_wp_idx]
                    print("Visited", current_wp_idx, self.waypoints)

                if render:
                    env.render()

    def test_mpc_bicycle(self, env, model, mpc, simulator, N=100, render=True, print_rew=False):
        for _ in range(N):
            # New env and trajectory
            env.reset()
            simulator.reset_history()

            x0 = np.array([0, 0, 0, 0.03, -0.01, 0.001]).reshape(-1, 1)
            simulator.x0 = x0
            mpc.x0 = x0
            mpc.set_initial_guess()
            x = x0

            use_mujoco = False
            episode_rew = 0
            waypts = [[3,0], [-3,0]]
            current_wp_idx = 0
            self.waypoints[:] = waypts[current_wp_idx]

            for _ in range(30000):
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
                    #u = np.array([0.0, 1]).reshape(-1, 1)
                    for _ in range(5):
                        x = simulator.make_step(u)

                    env.set_external_state({"x_pos" : x[0],
                                           "y_pos" : x[1],
                                           "phi" : x[2]})

                if np.sqrt((x[0] - self.waypoints[0]) ** 2 + (x[1] - self.waypoints[1]) ** 2) < 0.5:
                    current_wp_idx = int(not current_wp_idx)
                    self.waypoints[:] = waypts[current_wp_idx]
                    print("Visited", current_wp_idx, self.waypoints)

                if render:
                    env.render()

    def test_mpc_linmod(self, env, model, mpc, simulator, N=100, render=True, print_rew=False):
        for _ in range(N):
            # New env and trajectory
            env.reset()
            simulator.reset_history()

            x0 = np.array([0, 0, 0, 0.1, 0, 0] + [0] * 12).reshape(-1, 1)
            simulator.x0 = x0
            mpc.x0 = x0
            mpc.set_initial_guess()
            x = x0

            use_mujoco = False

            episode_rew = 0
            waypts = [[3,0], [-3,0]]
            current_wp_idx = 0
            self.waypoints[:] = waypts[current_wp_idx]

            for _ in range(30000):
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
                    #u = np.array([0.0, 1]).reshape(-1, 1)
                    for _ in range(5):
                        x = simulator.make_step(u)

                    env.set_external_state({"x_pos" : x[0],
                                           "y_pos" : x[1],
                                           "phi" : x[2]})

                if np.sqrt((x[0] - self.waypoints[0]) ** 2 + (x[1] - self.waypoints[1]) ** 2) < 0.5:
                    current_wp_idx = int(not current_wp_idx)
                    self.waypoints[:] = waypts[current_wp_idx]
                    print("Visited", current_wp_idx, self.waypoints)

                if render:
                    env.render()

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../opt/configs/train_buggy_a2c.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cbm = ControlBuggyMPC(config)
