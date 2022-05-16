import os

from src.envs.buggy_env_mujoco import BuggyEnv
from src.envs.buggy_maize_env_mujoco import BuggyMaizeEnv
from src.policies import *
from src.utils import *


class ControlBuggyMPPI:
    def __init__(self, mppi_config):
        self.mppi_config = mppi_config
        self.dynamics_model = self.load_dynamics_model()
        self.dt = self.mppi_config["dt"]

    def load_dynamics_model(self):
        dynamics_model = MLP(self.mppi_config["model_obs_dim"], self.mppi_config["model_act_dim"], hid_dim=128)
        model_path = os.path.join(os.path.dirname(__file__), "../opt/agents/buggy_lte.p")
        dynamics_model.load_state_dict(T.load(model_path), strict=False)
        return dynamics_model

    def test_mppi(self, env, n_episodes=100, n_samples=100, n_horizon=100, act_std=1, render=False):
        for _ in range(n_episodes):
            # New env and trajectory
            mujoco_obs = env.reset()

            # Initial action trajectory
            u_vec = np.zeros((n_horizon, 2), dtype=np.float32)

            step_ctr = 0
            while True:
                # Predict using MPC
                u_vec = self.mppi_predict(env, mujoco_obs, "traj", n_samples, n_horizon, u_vec, act_std)
                mujoco_obs, _, done, _ = env.step(u_vec[0])

                if render: env.render()
                if done: break

    def mppi_predict(self, env, mujoco_obs, mode, n_samples, n_horizon, act_mean_seq, act_std):
        # Sample random action matrix
        act_noises = np.random.randn(n_samples, n_horizon, self.mppi_config["act_dim"]) * act_std
        acts = np.tile(act_mean_seq, (n_samples, 1, 1)) + act_noises

        # Sample rollouts from learned dynamics
        init_model_state = mujoco_obs[:3]
        mppi_rollouts = self.make_mppi_rollouts(self.dynamics_model, init_model_state, acts)

        # Evaluate rollouts
        costs = self.evaluate_mppi_rollouts(env, mppi_rollouts, mode)

        # Choose trajectory using MPPI update
        acts_opt = self.calculate_mppi_trajectory(act_mean_seq, act_noises, costs)

        return acts_opt

    def make_mppi_rollouts(self, dynamics_model, init_model_velocities, acts):
        n_samples, n_horizon, acts_dim = acts.shape
        velocities = T.tensor(np.tile(init_model_velocities, (n_samples, 1)), dtype=T.float32)
        positions = T.zeros((n_samples, n_horizon, 3))

        obs = T.concat((velocities, T.tensor(acts[:, 0], dtype=T.float32)), dim=1)
        for h in range(n_horizon - 1):
            states = dynamics_model(obs)

            # Update positions array
            positions[:, h, 0] = positions[:, h, 0] + T.cos(states[:, 0]) * self.dt
            positions[:, h, 1] = positions[:, h, 1] + T.sin(states[:, 1]) * self.dt
            positions[:, h, 2] = positions[:, h, 2] + states[:, 2] * self.dt

            obs = T.concat((states, T.tensor(acts[:, h + 1], dtype=T.float32)), dim=1)

        return positions.detach().numpy()

    def evaluate_mppi_rollouts(self, env, rollouts, mode):
        costs = []
        for rollout in rollouts:
            if mode == "traj":
                cost = env.evaluate_rollout(rollout)
            else:
                cost = env.evaluate_rollout_free(rollout)
            costs.append(cost)
        return np.array(costs)

    def calculate_mppi_trajectory(self, act_mean_seq, act_noises, costs):
        # acts: n_samples, n_horizon, act_dim
        # costs: n_samples
        weights = np.exp(-costs / (self.mppi_config["mppi_lambda"]))
        acts = act_mean_seq + np.sum(weights[:, np.newaxis, np.newaxis] * act_noises, axis=0) / np.sum(weights)
        acts_clipped = np.clip(acts, -2, 2)
        return acts_clipped

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "configs/control_buggy_mppi.yaml"), 'r') as f:
        mppi_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_maize_env_mujoco.yaml"), 'r') as f:
        buggy_maize_config = yaml.load(f, Loader=yaml.FullLoader)
    env = BuggyMaizeEnv(buggy_maize_config, seed=np.random.randint(0, 10000))
    cbm = ControlBuggyMPPI(mppi_config)

    # Test
    cbm.test_mppi(env, n_episodes=10, n_samples=100, n_horizon=10, act_std=1, render=True)
