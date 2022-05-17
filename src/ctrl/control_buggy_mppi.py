import os

from src.envs.buggy_env_mujoco import BuggyEnv
from src.envs.buggy_maize_env_mujoco import BuggyMaizeEnv
from src.policies import *
from src.utils import *
from torch import device


class ControlBuggyMPPI:
    def __init__(self, mppi_config):
        self.mppi_config = mppi_config
        self.dynamics_model = self.load_dynamics_model()
        self.dt = self.mppi_config["dt"]
        self.device = device('cpu')
        self.dynamics_model.to(self.device)

    def load_dynamics_model(self):
        dynamics_model = MLP(self.mppi_config["model_obs_dim"], self.mppi_config["model_act_dim"], hid_dim=128)
        model_path = os.path.join(os.path.dirname(__file__), "../opt/agents/buggy_lte_mujoco.p")
        dynamics_model.load_state_dict(T.load(model_path), strict=False)
        return dynamics_model

    def test_mppi(self, env, n_episodes=100, n_samples=100, n_horizon=100, act_std=1, mode="traj", render=False):
        for _ in range(n_episodes):
            # New env and trajectory
            mujoco_obs = env.reset()
            model_pos = env.get_xytheta()

            # Initial action trajectory
            u_vec = np.zeros((n_horizon, 2), dtype=np.float32)

            while True:
                # Predict using MPC
                u_vec = self.mppi_predict(env, mujoco_obs, model_pos, mode, n_samples, n_horizon, u_vec, act_std)
                mujoco_obs, _, done, _ = env.step(u_vec[0])
                model_pos = env.get_xytheta()

                if render: env.render()
                if done: break

    def mppi_predict(self, env, mujoco_obs, init_model_positions, mode, n_samples, n_horizon, act_mean_seq, act_std):
        # Sample random action matrix
        act_noises = np.clip(np.random.randn(n_samples, n_horizon, self.mppi_config["act_dim"]) * act_std, -1, 1)
        acts = np.tile(act_mean_seq, (n_samples, 1, 1)) + act_noises

        # Sample rollouts from learned dynamics
        init_model_state = mujoco_obs[:3]
        mppi_rollout_positions, mppi_rollout_velocities = self.make_mppi_rollouts(self.dynamics_model, init_model_state, init_model_positions, acts)

        # Evaluate rollouts
        costs = self.evaluate_mppi_rollouts(env, mppi_rollout_positions, mppi_rollout_velocities, mode)

        # Choose trajectory using MPPI update
        acts_opt = self.calculate_mppi_trajectory(act_mean_seq, act_noises, costs)

        return acts_opt

    def make_mppi_rollouts(self, dynamics_model, init_model_velocities, init_model_positions, acts):
        n_samples, n_horizon, acts_dim = acts.shape
        velocities = T.tensor(np.tile(init_model_velocities, (n_samples, 1)), dtype=T.float32)
        positions = T.tensor(np.tile(init_model_positions, (n_samples, n_horizon, 1)), dtype=T.float32)
        rollout_velocities = np.zeros(positions.shape)

        obs = T.concat((velocities, T.tensor(acts[:, 0], dtype=T.float32)), dim=1)
        for h in range(n_horizon - 1):
            pred_velocities = dynamics_model(obs.to(self.device)).to(device('cpu'))
            rollout_velocities[:, h, :] = pred_velocities.detach().numpy()

            # Update positions array
            positions[:, h + 1, 0] = positions[:, h, 0] + T.cos(pred_velocities[:, 0]) * self.dt
            positions[:, h + 1, 1] = positions[:, h, 1] + T.sin(pred_velocities[:, 1]) * self.dt
            positions[:, h + 1, 2] = positions[:, h, 2] + pred_velocities[:, 2] * self.dt

            obs = T.concat((pred_velocities, T.tensor(acts[:, h + 1], dtype=T.float32)), dim=1)

        return positions.detach().numpy(), rollout_velocities

    def evaluate_mppi_rollouts(self, env, rollout_positions, rollout_velocities, mode):
        costs = []
        for rp, rv in zip(rollout_positions, rollout_velocities):
            if mode == "traj":
                cost = env.evaluate_rollout(rp)
            else:
                cost = env.evaluate_rollout_free(rp, rv)
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
    cbm.test_mppi(env, n_episodes=10, n_samples=50, n_horizon=10, act_std=0.5, mode="traj", render=True)
