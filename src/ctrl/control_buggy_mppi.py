import yaml

from src.policies import *
from src.envs.buggy_env_mujoco import BuggyEnv
import os

class ControlBuggyMPPI:
    def __init__(self, mppi_config, buggy_config):
        self.mppi_config = mppi_config
        self.buggy_config = buggy_config
        self.buggy_env_mujoco = BuggyEnv(self.buggy_config)
        self.dynamics_model = self.load_dynamics_model()
        self.dt = self.mppi_config["dt"]

    def load_dynamics_model(self):
        dynamics_model = MLP(self.mppi_config["model_obs_dim"], self.mppi_config["model_act_dim"], hid_dim=256)
        model_path = os.path.join(os.path.dirname(__file__), "../opt/agents/buggy_lte.p")
        dynamics_model.load_state_dict(T.load(model_path), strict=False)
        return dynamics_model

    def test_mppi(self, env, n_episodes=100, n_samples=100, n_horizon=100, act_std=1, render=False):
        for _ in range(n_episodes):
            # New env and trajectory
            mujoco_obs = env.reset()

            # Initial action trajectory
            u_vec = np.zeros(n_horizon, dtype=np.float32)

            while True:
                # Predict using MPC
                u_vec = self.mppi_predict(mujoco_obs, n_samples, n_horizon, u_vec, act_std)
                mujoco_obs, _, done, _ = env.step(u_vec[0])

                if render: env.render()
                if done: break

    def mppi_predict(self, mujoco_obs, n_samples, n_horizon, act_mean_seq, act_std):
        # Sample random action matrix
        act_noises = np.random.randn(n_samples, n_horizon, self.mppi_config["act_dim"]) * act_std
        acts = act_mean_seq + act_noises

        # Sample rollouts from learned dynamics
        init_model_state = mujoco_obs[:5]
        mppi_rollouts = self.make_mppi_rollouts(self.dynamics_model, init_model_state, acts)

        # Evaluate rollouts
        costs = self.evaluate_mppi_rollouts(mppi_rollouts)

        # Choose trajectory using MPPI update
        acts_opt = self.calculate_mppi_trajectory(act_mean_seq, act_noises, costs)

        return acts_opt

    def make_mppi_rollouts(self, dynamics_model, init_model_state, acts):
        n_samples, n_horizon, acts_dim = acts.shape
        states = np.tile(init_model_state, (n_samples, 1))
        positions = np.zeros((n_samples, n_horizon, 3))

        obs = np.concatenate((states, acts[:, 0]), dim=1)
        for h in range(n_horizon - 1):
            states = dynamics_model(obs)

            # Update positions array
            positions[:, h, 0] = positions[:, h, 0] + np.cos(states[:, 2]) * self.dt
            positions[:, h, 1] = positions[:, h, 1] + np.sin(states[:, 3]) * self.dt
            positions[:, h, 2] = positions[:, h, 2] + states[:, 3] * self.dt

            obs = np.concatenate((states, acts[:, h + 1]), dim=1)

        return positions

    def evaluate_mppi_rollouts(self, rollouts):
        costs = []
        for rollout in rollouts:
            cost = self.buggy_env_mujoco.evaluate_rollout(rollout)
            costs.append(cost)
        return np.array(costs)

    def calculate_mppi_trajectory(self, act_mean_seq, act_noises, costs):
        # acts: n_samples, n_horizon, act_dim
        # costs: n_samples
        weights = np.exp(-costs / (self.mppi_config["mppi_lambda"]))
        acts = act_mean_seq + np.sum(weights * act_noises, axis=1) / np.sum(weights)

        return acts

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "configs/control_buggy_mppi.yaml"), 'r') as f:
        mppi_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
        buggy_config = yaml.load(f, Loader=yaml.FullLoader)
    cbm = ControlBuggyMPPI(mppi_config, buggy_config)
