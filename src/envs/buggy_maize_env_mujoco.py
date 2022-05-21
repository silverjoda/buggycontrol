import random

import gym
import matplotlib.pyplot as plt
from gym import spaces
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy import interpolate

from src.envs.engines import *
from src.opt.simplex_noise import SimplexNoise
from src.utils import e2q, dist_between_wps

GLOBAL_DEBUG = False

class BuggyMaize():
    def __init__(self, config):
        self.config = config

        self.block_size = 1.5 # Block size in meters
        self.n_blocks = self.config["n_blocks"]
        self.grid_resolution = 0.1 # Number of points per meter
        self.grid_X_len = int(self.block_size * (self.n_blocks + 2) / self.grid_resolution)
        self.grid_Y_len = int(self.block_size * (self.n_blocks * 2 + 1) / self.grid_resolution)
        self.grid_block_len = int(self.block_size / self.grid_resolution)
        self.grid_meter_len = int(1 / self.grid_resolution)

        deadzone = self.config["deadzone"]
        self.barrier_halflength_coeff = 0.5 + deadzone
        self.barrier_halfwidth_coeff = 0.2 + deadzone * 0.5
        self.barrier_real_halflength = self.barrier_halflength_coeff * self.block_size
        self.barrier_real_halfwidth = self.barrier_halfwidth_coeff * self.block_size

        self.reset()

    def generate_random_maize(self):
        all_possible_shifts = [[1,0],[-1,0],[0,1],[0,-1]]
        while True:
            gen_failed = False
            # Make list of blocks which represent the maize
            cur_x, cur_y = self.block_size, 0
            blocks = [[0, 0], [cur_x, cur_y]]
            for i in range(self.n_blocks):
                possible_shifts = []
                for aps_x, aps_y in all_possible_shifts:
                    cur_x_candidate = cur_x + aps_x * self.block_size
                    cur_y_candidate = cur_y + aps_y * self.block_size
                    if [cur_x_candidate, cur_y_candidate] not in blocks and cur_x_candidate > 0:
                        possible_shifts.append([cur_x_candidate, cur_y_candidate])

                if len(possible_shifts) == 0:
                    gen_failed = True
                    break

                cur_x, cur_y = deepcopy(random.choice(possible_shifts))
                blocks.append(deepcopy([cur_x, cur_y]))

            # If we have generated a successful maize, then finish
            if not gen_failed: break

        # Decide barriers from block list
        all_barriers = []
        for i, bl in enumerate(blocks):
            # Generate list of barriers using given block location
            bl_x, bl_y = bl

            # True when vertical
            bar_n = (bl_x + self.block_size / 2, bl_y, False)
            bar_s = (bl_x - self.block_size / 2, bl_y, False)
            bar_w = (bl_x, bl_y + self.block_size / 2, True)
            bar_e = (bl_x, bl_y - self.block_size / 2, True)
            barriers = [bar_n, bar_s, bar_w, bar_e]
            all_barriers.extend(barriers)

            # Filter barriers according to previous neighbors
            if i > 0:
                p_bl_x, p_bl_y = blocks[i - 1]
                for _ in range(2):
                    if p_bl_x > bl_x and bar_n in all_barriers: all_barriers.remove(bar_n)
                    if p_bl_x < bl_x and bar_s in all_barriers: all_barriers.remove(bar_s)
                    if p_bl_y > bl_y and bar_w in all_barriers: all_barriers.remove(bar_w)
                    if p_bl_y < bl_y and bar_e in all_barriers: all_barriers.remove(bar_e)

        # Make dense grid representation
        dense_grid = np.ones((self.grid_X_len, self.grid_Y_len))
        grid_barrier_half_length = int(self.barrier_halflength_coeff * self.grid_block_len)
        grid_barrier_half_width = int(self.barrier_halfwidth_coeff * self.grid_block_len)
        for bar in all_barriers:
            bar_x, bar_y, vert = bar

            m_side = grid_barrier_half_width
            n_side = grid_barrier_half_length
            if vert:
                m_side = grid_barrier_half_length
                n_side = grid_barrier_half_width

            bar_m, bar_n = self.xy_to_grid(bar_x, bar_y)

            # Add barrier to grid
            dense_grid[bar_m - m_side: bar_m + m_side, bar_n - n_side: bar_n + n_side] = 0

        start = self.xy_to_grid(0, 0)
        finish = self.xy_to_grid(*blocks[-1])

        return blocks, all_barriers, dense_grid, start, finish

    def xy_to_grid(self, x, y):
        m = int(self.grid_X_len - x * self.grid_meter_len - 0.5 * self.grid_meter_len)
        n = int(0.5 * self.grid_Y_len - y * self.grid_meter_len)

        m = np.clip(m, 0, self.grid_X_len - 1)
        n = np.clip(n, 0, self.grid_Y_len - 1)

        return m, n

    def xy_to_grid_parallel(self, positions):
        m = (self.grid_X_len - positions[:, 0] * self.grid_meter_len - 0.5 * self.grid_meter_len).astype(np.int32)
        n = (0.5 * self.grid_Y_len - positions[:, 1] * self.grid_meter_len).astype(np.int32)

        m = np.clip(m, 0, self.grid_X_len - 1)
        n = np.clip(n, 0, self.grid_Y_len - 1)

        return m, n

    def grid_to_xy(self, m, n):
        x = self.grid_X_len / self.grid_meter_len - 0.5 - m / self.grid_meter_len
        y = 0.5 * self.grid_Y_len / self.grid_meter_len - n / self.grid_meter_len
        return x, y

    def plot_grid(self, grid, path_astar, path_spline):
        grid_cpy = np.copy(grid)
        for pt in path_astar:
            grid_cpy[pt[0], pt[1]] = 10

        x_spln, y_spln = zip(*path_spline)

        # Plot
        plt.imshow(grid_cpy)
        plt.plot(y_spln, x_spln)
        plt.show()

    def plot_grid_with_trajs(self, grid, path_spline, positions, costs):
        grid_cpy = np.copy(grid)
        x_spln, y_spln = zip(*path_spline)

        # Plot grid and shortest path
        plt.imshow(grid_cpy)
        plt.plot(y_spln, x_spln)

        # Select 10 random positions
        rnd_indeces = np.random.choice(np.arange(len(positions)), 10)
        rnd_positions = positions[rnd_indeces]
        rnd_costs = costs[rnd_indeces]

        min_cost, max_cost = min(rnd_costs), max(rnd_costs)
        cost_range = max_cost - min_cost

        for rp, rc in zip(rnd_positions, rnd_costs):
            cost_intensity = (rc - min_cost) / cost_range
            rp_m, rp_n = self.xy_to_grid_parallel(rp)
            plt.plot(rp_n, rp_m, color=(cost_intensity, 0, 0))

        plt.show()

    def generate_shortest_path(self, grid, start, finish):
        grid = Grid(matrix=grid)
        start = grid.node(start[1],start[0])
        end = grid.node(finish[1], finish[0])
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path_astar, runs = finder.find_path(start, end, grid)

        y, x = zip(*path_astar)
        path_astar = zip(*[x[::3], y[::3]])

        # Make resample as spline
        tck, u = interpolate.splprep([x, y], s=15, k=3)
        xi, yi = interpolate.splev(np.linspace(0, 1, 300), tck)

        path_spline = list(zip(*[xi,yi]))

        return path_astar, path_spline

    def get_barriers(self):
        return self.all_barriers

    def get_barrier_edgepoints(self):
        edge_pts = []
        for x, y, is_vert in self.all_barriers:
            bhl, bhw = self.barrier_real_halfwidth, self.barrier_real_halflength
            if is_vert:
                bhw, bhl = self.barrier_real_halfwidth, self.barrier_real_halflength
            edge_pts.append([x-bhl, y - bhw])
            edge_pts.append([x+bhl, y - bhw])
            edge_pts.append([x-bhl, y + bhw])
            edge_pts.append([x+bhl, y + bhw])
        return edge_pts

    def position_in_barrier(self, pos_x, pos_y):
        m, n = self.xy_to_grid(pos_x, pos_y)
        if self.dense_grid[m, n] < 1:
            return True
        return False

    def position_in_barrier_parallel(self, positions):
        grid_m, grid_n = self.xy_to_grid_parallel(positions)
        pib = self.dense_grid[grid_m, grid_n] < 1
        return pib

    def dist_from_centerline(self, pos_x, pos_y):
        m, n = self.xy_to_grid(pos_x, pos_y)
        return self.dense_grid_field[m, n]

    def get_shortest_path_pts_xy(self):
        sptxy = []
        for pt_x, pt_y in self.shortest_path_pts_spline:
            sptxy.append(self.grid_to_xy(pt_x, pt_y))
        return sptxy

    def generate_dense_grid_field(self, grid, blocks):
        grid_field = np.copy(grid)
        grid_field = -(grid_field - 1)

        for _ in range(7):
            # Make new copy for iteration
            grid_new = np.copy(grid_field)

            # Go over all blocks and make value iteration for each gridpoint
            for x, y in blocks:
                m_c, n_c = self.xy_to_grid(x, y)
                m_min = np.maximum(m_c - int(self.grid_block_len / 2) - 7, 0)
                m_max = np.minimum(m_c + int(self.grid_block_len / 2) + 7, self.grid_X_len - 1)
                n_min = np.maximum(n_c - int(self.grid_block_len / 2) - 7, 0)
                n_max = np.minimum(n_c + int(self.grid_block_len / 2) + 7, self.grid_Y_len - 1)

                for i in range(m_min, m_max):
                    for j in range(n_min, n_max):
                        if grid_field[i, j] < 1:
                            aggr_fun = np.max
                            grid_new[i, j] = aggr_fun([grid_field[i-1, j],
                                                grid_field[i+1, j],
                                                grid_field[i, j-1],
                                                grid_field[i, j+1]]) - 0.2

            grid_field = grid_new

        # Add minimum value to grid to normalize to zero at center point
        grid_field = np.clip(grid_field, 0, 1)
        return grid_field

    def reset(self):
        self.blocks, self.all_barriers, self.dense_grid, self.start, self.finish = self.generate_random_maize()
        self.shortest_path_pts, self.shortest_path_pts_spline = self.generate_shortest_path(self.dense_grid, self.start, self.finish)
        self.dense_grid_field = self.generate_dense_grid_field(self.dense_grid, self.blocks)
        if GLOBAL_DEBUG:
            self.plot_grid(self.dense_grid_field, self.shortest_path_pts, self.shortest_path_pts_spline)

class BuggyMaizeEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 100
    }

    def __init__(self, config, seed=1337):
        self.config = config

        self.obs_dim = self.config["state_dim"] \
                       + self.config["n_traj_pts"] * 2 \
                       + self.config["allow_latent_input"] * self.config["latent_dim"] \
                       + self.config["allow_lte"]
        self.act_dim = 2

        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.act_dim,), dtype=np.float32)

        self.sim, self.engine = self.load_random_env()

        self.seed(seed)
        self.maize = BuggyMaize(config)

        if self.config["rew_type"] == "free":
            self.config["wp_reach_dist"] = 0.5

    def load_random_env(self):
        if self.config["allow_lte"] and np.random.rand() < self.config["lte_prob"]:
            engine = LTEEngine(self.config)
        elif self.config["use_hybrid_engine"]:
            engine = HybridEngine(self.config)
        else:
            engine = MujocoEngine(self.config)
        sim = engine.mujoco_sim
        return sim, engine

    def set_barrier_positions(self, barriers_list):
        for i, p in enumerate(barriers_list):
            x, y, vert = p
            pos = (x,y,0)
            quat = 1, 0, 0, 0
            if vert:
                quat = 0.707, 0, 0, 0.707
            self.sim.data.set_mocap_pos(f"barrier{i}", pos)
            self.sim.data.set_mocap_quat(f"barrier{i}", quat)

    def get_obs_dict(self):
        return self.engine.get_obs_dict()

    def get_state_vec(self, obs_dict):
        return self.engine.get_state_vec(obs_dict)

    def get_complete_obs_vec(self):
        complete_obs_vec, _ = self.engine.get_complete_obs_vec(allow_latent_input=self.config["allow_latent_input"])
        return complete_obs_vec

    def get_xytheta(self):
        return self.engine.get_xytheta()

    def get_reward(self, obs_dict, wp_visited):
        pos = np.array(obs_dict["pos"], dtype=np.float32)
        cur_wp = np.array(self.engine.wp_list[self.engine.cur_wp_idx], dtype=np.float32)
        if self.engine.cur_wp_idx > 0:
            prev_wp = np.array(self.engine.wp_list[self.engine.cur_wp_idx - 1], dtype=np.float32)
            path_deviation = np.abs((cur_wp[0] - prev_wp[0]) * (prev_wp[1] - pos[1]) - (prev_wp[0] - pos[0]) * (
                        cur_wp[1] - prev_wp[1])) / np.sqrt(
                np.square(cur_wp[0] - prev_wp[0]) + np.square(cur_wp[1] - prev_wp[1]))
        else:
            path_deviation = 0

        dist_between_cur_wp = np.sqrt(np.square((cur_wp[0] - pos[0])) + np.square((cur_wp[1] - pos[1])))

        if self.config["rew_type"] == "traj":
            r = wp_visited * (1 / (1 + 0.5 * path_deviation)) - dist_between_cur_wp * 0.01
        else:
            # Calculate x-velocity cost (maximize speed)
            velocity_rew = obs_dict["vel"][0]

            # Calculate if in barrier
            #in_barrier = self.maize.position_in_barrier(pos[0], pos[1])

            # Calculate center deviation of final state cost
            center_deviation_cost = self.maize.dist_from_centerline(*pos[:2])
            r = velocity_rew - center_deviation_cost * 1. - 3

        return r, dist_between_cur_wp


    def set_external_state(self, state_dict):
        old_state = self.sim.get_state()
        qpos = old_state.qpos # qvel
        qpos[0:2] = state_dict["x_pos"], state_dict["y_pos"]
        quat = e2q(0,0,state_dict["phi"])
        qpos[3:7] = quat
        new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state.qvel,
                                         old_state.act, old_state.udd_state)

        self.sim.set_state(new_state)
        self.sim.forward()

    def step(self, act):
        self.step_ctr += 1

        # Turn, throttle
        scaled_act = act
        if self.engine.__class__.__name__ != 'LTEEngine':
            scaled_act = [np.clip(act[0] * 0.2, -0.42, 0.42), np.clip(act[1] * 0.5 + 0.5, 0, 1)]

        done, wp_visited = self.engine.step(scaled_act)

        # Get new observation
        complete_obs_vec, obs_dict = self.engine.get_complete_obs_vec(allow_latent_input=self.config["allow_latent_input"])

        # calculate reward
        r, dist_to_cur_wp = self.get_reward(obs_dict, wp_visited)

        act_pen = np.mean(np.square(act - self.prev_scaled_act)) * self.config["act_pen"]
        self.prev_scaled_act = act

        r -= act_pen

        # Calculate termination
        done = done or self.step_ctr > self.config["max_steps"]
        #done = self.step_ctr > self.config["max_steps"]
        #done = done or self.step_ctr > self.config["max_steps"]

        return complete_obs_vec, r, done, {"visited" : wp_visited}

    def reset(self):
        # Reset variables
        self.step_ctr = 0
        self.prev_scaled_act = np.zeros(2)

        # Reset simulation
        self.engine.reset()

        # Reset maize
        self.maize.reset()

        # Force engine trajectory to be same as maize
        self.engine.reset_trajectory(traj_pts_in=self.maize.get_shortest_path_pts_xy())

        # Draw the barrier visuals
        self.set_barrier_positions(self.maize.get_barriers())

        # Reset environment variables
        obs_vec, _ = self.engine.get_complete_obs_vec(allow_latent_input=self.config["allow_latent_input"])

        return obs_vec

    def render(self, mode=None):
        self.engine.render()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed, seed)

    def demo(self):
        while True:
            self.noise = SimplexNoise(dim=2, smoothness=30, multiplier=1.6)
            self.reset()

            cum_rew = 0
            while True:
                zero_act = np.array([-0.3, -0.5])
                rnd_act = np.clip(self.noise(), -1, 1)
                act = zero_act
                obs, r, done, _ = self.step(act) # turn, throttle

                cum_rew += r
                if self.config["render"]:
                    if self.engine.__class__.__name__ == 'LTEEngine':
                        self.set_external_state({"x_pos": self.engine.xy_pos[0],
                                                 "y_pos": self.engine.xy_pos[1],
                                                 "phi": self.engine.theta})

                    self.render()

            print("Cumulative rew: {}".format(cum_rew))

    def evaluate_rollout(self, rollout):
        # Get current wp list in buggy frame and position of buggy
        obs_dict = self.engine.get_obs_dict()
        wp_list = obs_dict["wp_list"]

        cur_wp_idx = 0
        cum_rew = 0
        for pos in rollout:
            # Distance between current waypoint
            cur_wp_dist = dist_between_wps(pos, wp_list[cur_wp_idx])

            ## Abort if trajectory goes to shit
            #if cur_wp_dist > 0.5:
            #    break

            wp_visited = False
            # Check if visited waypoint, if so, move index
            if cur_wp_dist < self.config["wp_reach_dist"]:
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

            dist_between_cur_wp = np.sqrt(np.square((cur_wp[0] - pos[0])) + np.square((cur_wp[1] - pos[1])))

            r = wp_visited * (1 / (1 + 3. * path_deviation)) - dist_between_cur_wp * 0.01
            #r = wp_visited
            cum_rew += r

        return -cum_rew

    def evaluate_rollout_free(self, mppi_rollout_positions, mppi_rollout_velocities):
        velocity_cost = 0
        barrier_cost = 0
        for rp, rv in zip(mppi_rollout_positions, mppi_rollout_velocities):
            # Calculate x-velocity cost (maximize speed)
            velocity_cost += (-rv[0])

            # Calculate if in barrier
            in_barrier = self.maize.position_in_barrier(rp[0], rp[1])

            barrier_cost += in_barrier * 300

            if in_barrier: break

        # Calculate center deviation of final state cost
        center_deviation_cost = self.maize.dist_from_centerline(*mppi_rollout_positions[-1, :2])
        total_cost = center_deviation_cost * 10. + velocity_cost + barrier_cost

        return total_cost


if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_maize_env_mujoco.yaml"))

    #bm = BuggyMaize(config)
    be = BuggyMaizeEnv(config, seed=np.random.randint(0,1000))
    be.demo()
