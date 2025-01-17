import os

from src.envs.buggy_env_mujoco import BuggyEnv
from src.envs.buggy_maize_env_mujoco import BuggyMaizeEnv
from src.policies import *
from src.utils import *
from torch import device
import time
from multiprocessing import Pool
from src.opt.simplex_noise import SimplexNoise
GLOBAL_DEBUG = False

class ControlBuggyMPPI:
    def __init__(self, mppi_config):
        self.mppi_config = mppi_config
        self.dynamics_model = self.load_dynamics_model()
        self.dt = self.mppi_config["dt"]
        self.device = device('cpu')
        self.dynamics_model.to(self.device)
        self.simplex_action_noise = SimplexNoise(dim=2, smoothness=300, multiplier=1.0)

    def load_dynamics_model(self):
        dynamics_model = MLP(self.mppi_config["model_obs_dim"], self.mppi_config["model_act_dim"], hid_dim=128)
        model_path = os.path.join(os.path.dirname(__file__), "../opt/agents/buggy_lte_mujoco.p")
        dynamics_model.load_state_dict(T.load(model_path), strict=False)
        return dynamics_model

    def test_mppi(self, env, seed=1337, test_traj=None, n_samples=100, n_horizon=100, act_std=1, mode="traj", render=False):
        # New env and trajectory
        env.seed(seed)
        mujoco_obs = env.reset()

        if test_traj is not None:
            env.engine.wp_list = test_traj
            env.engine.update_wp_visuals()

        # Initial action trajectory
        u_vec = np.zeros((n_horizon, 2), dtype=np.float32)

        cum_rew = 0
        step_ctr = 0
        time_delta = 0.01
        while True:
            model_pos = env.get_xytheta()

            # Predict using MPC
            u_vec = self.mppi_predict(env, mujoco_obs, model_pos, mode, n_samples, n_horizon, u_vec, act_std)
            mujoco_obs, r, done, _ = env.step(u_vec[0] * 2)

            cum_rew += r
            step_ctr += 1

            if render: env.render()
            if done: break

        return cum_rew, step_ctr * time_delta

    def mppi_predict(self, env, mujoco_obs, init_model_positions, mode, n_samples, n_horizon, act_mean_seq, act_std):
        # Sample random action matrix
        act_noises = np.clip(np.random.randn(n_samples, n_horizon, self.mppi_config["act_dim"]) * act_std, -1, 1)
        #act_noises = self.simplex_action_noise.sample_parallel(n_samples, n_horizon, self.mppi_config["act_dim"])

        #act_noises[:, :, 0] = -0.3
        #act_noises[:, :, 1] = -0.3 # TEMPORARY, TO MAKE THE THROTTLE CONSTANT
        acts = np.clip(np.tile(act_mean_seq, (n_samples, 1, 1)) + act_noises, -1, 1)

        # Sample rollouts from learned dynamics
        init_model_state = mujoco_obs[:3]
        mppi_rollout_positions, mppi_rollout_velocities = self.make_mppi_rollouts(self.dynamics_model, init_model_state, init_model_positions, acts)

        # Evaluate rollouts
        costs, ctrs = self.evaluate_mppi_rollouts(env, mppi_rollout_positions, mppi_rollout_velocities, mode)

        # Choose trajectory using MPPI update
        acts_opt = self.calculate_mppi_trajectory(act_mean_seq, act_noises, costs)

        if GLOBAL_DEBUG:
            print("###############")
            env.maize.plot_grid_with_trajs(env.maize.dense_grid, env.maize.shortest_path_pts_spline, mppi_rollout_positions, costs, ctrs)

        return acts_opt

    def make_mppi_rollouts(self, dynamics_model, init_model_velocities, init_model_positions, acts):
        n_samples, n_horizon, acts_dim = acts.shape
        velocities = T.tensor(np.tile(init_model_velocities, (n_samples, 1)), dtype=T.float32)
        positions = T.tensor(np.tile(init_model_positions, (n_samples, n_horizon, 1)), dtype=T.float32)
        rollout_velocities = np.zeros(positions.shape)

        obs = T.concat((velocities, T.tensor(acts[:, 0], dtype=T.float32)), dim=1)

        for h in range(n_horizon - 1):
            with T.no_grad():
                pred_velocities = dynamics_model(obs.to(self.device)).to(device('cpu'))
            rollout_velocities[:, h, :] = pred_velocities.detach().numpy()

            # Update positions array
            positions[:, h + 1, 0] = positions[:, h, 0] + T.cos(positions[:, h, 2]) * pred_velocities[:, 0] * self.dt + T.sin(positions[:, h, 2]) * pred_velocities[:, 1] * self.dt
            positions[:, h + 1, 1] = positions[:, h, 1] + T.sin(positions[:, h, 2]) * pred_velocities[:, 0] * self.dt + T.cos(positions[:, h, 2]) * pred_velocities[:, 1] * self.dt
            positions[:, h + 1, 2] = positions[:, h, 2] + pred_velocities[:, 2] * self.dt

            obs = T.concat((pred_velocities, T.tensor(acts[:, h + 1], dtype=T.float32)), dim=1)

        return positions.detach().numpy(), rollout_velocities

    def evaluate_mppi_rollouts(self, env, rollout_positions, rollout_velocities, mode):
        costs = []
        ctrs = []
        #t1 = time.time()

        step_skip = 5
        for rp, rv in zip(rollout_positions, rollout_velocities):
            if mode == "traj":
                cost, ctr = evaluate_rollout((rp[::step_skip, :], env.engine.wp_list, env.engine.cur_wp_idx, env.maize))
            else:
                cost, ctr = evaluate_rollout_free((rp[::step_skip, :], rv[::step_skip, :], env.maize))
            costs.append(cost)
            ctrs.append(ctr * step_skip)

        #print(time.time() - t1)
        return np.array(costs), np.array(ctrs)

    def evaluate_mppi_rollouts_mp(self, env, rollout_positions, rollout_velocities, mode):
        #t1 = time.time()

        step_skip = 5
        maizes = [env.maize] * (len(rollout_positions) // step_skip)
        wp_lists = [env.engine.wp_list] * (len(rollout_positions) // step_skip)

        if mode == "traj":
            with Pool(1) as p:
                costs = p.map(evaluate_rollout, zip(rollout_positions[:, ::step_skip, :], wp_lists))
        else:
            with Pool(1) as p:
                costs = p.map(evaluate_rollout_free_par, zip(rollout_positions[:, ::step_skip, :], rollout_positions[:, ::step_skip, :], maizes))

        costs_arr = np.array(costs) / 1000.
        costs_arr_full = np.repeat(costs_arr, step_skip)

        #print(time.time() - t1)
        return costs_arr_full

    def calculate_mppi_trajectory(self, act_mean_seq, act_noises, costs):
        # acts: n_samples, n_horizon, act_dim
        # costs: n_samples
        costs_norm = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
        weights = np.exp(-costs_norm / (self.mppi_config["mppi_lambda"]))
        acts = act_mean_seq + np.sum(weights[:, np.newaxis, np.newaxis] * act_noises, axis=0) / np.sum(weights)
        acts_clipped = np.clip(acts, -1, 1)
        acts_clipped[:, 1] = -0.25
        return acts_clipped

def evaluate_rollout(args):
        rollout, wp_list, wp_idx, maize = args
        wp_reach_dist = 0.5
        cur_wp_idx = wp_idx
        cum_cost = np.random.rand() * 0.01
        #print(dist_between_wps(rollout[0], wp_list[cur_wp_idx]))
        ctr = 0
        barrier_cost = 0
        for pos in rollout:
            # Distance between current waypoint
            cur_wp_dist = dist_between_wps(pos, wp_list[cur_wp_idx])

            ## Abort if trajectory goes to shit
            #if cur_wp_dist > 0.6:
            #    break

            in_barrier = maize.position_in_barrier(pos[0], pos[1])
            if in_barrier:
                barrier_cost = 100

            wp_visited = False
            # Check if visited waypoint, if so, move index
            if cur_wp_dist < wp_reach_dist:
                cur_wp_idx += 1
                wp_visited = True

            # Calculate visitation + deviation reward
            cur_wp = np.array(wp_list[cur_wp_idx], dtype=np.float32)
            if cur_wp_idx > 0:
                prev_wp = np.array(wp_list[cur_wp_idx - 1], dtype=np.float32)
                path_dev_nom = np.abs(
                    (cur_wp[0] - prev_wp[0]) * (prev_wp[1] - pos[1]) - (prev_wp[0] - pos[0]) * (cur_wp[1] - prev_wp[1]))
                path_deviation_denom = np.sqrt(np.square(cur_wp[0] - prev_wp[0]) + np.square(cur_wp[1] - prev_wp[1]))
                path_deviation = path_dev_nom / path_deviation_denom
            else:
                path_deviation = 0

            #r = wp_visited * (1 / (1 + 5. * path_deviation)) - cur_wp_dist * 0.2
            cost = -wp_visited + path_deviation + barrier_cost
            #r = wp_visited
            cum_cost += cost

            if in_barrier:
                break

            ctr += 1

        return cum_cost, ctr

def evaluate_rollout_free(args):
    mppi_rollout_positions, mppi_rollout_velocities, maize = args
    velocity_cost = 0
    progress_cost = 0
    center_deviation_cost = 0
    ctr = 0
    for rp, rv in zip(mppi_rollout_positions, mppi_rollout_velocities):
        # Calculate x-velocity cost (maximize speed)
        velocity_cost += (-rv[0])

        # Calculate if in barrier
        in_barrier = maize.position_in_barrier(rp[0], rp[1])

        progress_cost += maize.get_progress_cost(rp[0], rp[1])

        center_deviation_cost += maize.dist_from_centerline(*rp[:2])

        ctr += 1

        if in_barrier: break

    # Calculate center deviation of final state cost
    center_deviation_cost += maize.dist_from_centerline(*mppi_rollout_positions[-1, :2])
    total_cost = center_deviation_cost * 10. + velocity_cost + progress_cost

    return total_cost, ctr

def evaluate_rollout_free_par(args):
    mppi_rollout_positions, mppi_rollout_velocities, maize = args

    # Calculate x-velocity cost (maximize speed)
    velocity_cost = -np.sum(mppi_rollout_velocities[:, 0])

    # Calculate if in barrier
    in_barrier = maize.position_in_barrier_parallel(mppi_rollout_positions)
    barrier_cost = np.sum(in_barrier) * 300

    # Calculate center deviation of final state cost
    center_deviation_cost = maize.dist_from_centerline(*mppi_rollout_positions[-1, :2])
    total_cost = center_deviation_cost * 10. + velocity_cost + barrier_cost
    return total_cost

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "configs/control_buggy_mppi.yaml"), 'r') as f:
        mppi_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_maize_env_mujoco.yaml"), 'r') as f:
        buggy_maize_config = yaml.load(f, Loader=yaml.FullLoader)
    env = BuggyMaizeEnv(buggy_maize_config, seed=np.random.randint(0, 10000))
    cbm = ControlBuggyMPPI(mppi_config)

    rnd_seed = np.random.randint(0, 10000)

    # Test
    cbm.test_mppi(env, seed=rnd_seed, test_traj=None, n_samples=300, n_horizon=100, act_std=0.7, mode="traj", render=True)
