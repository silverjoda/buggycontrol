import random
import time

import gym
import matplotlib.pyplot as plt
import mujoco_py
from gym import spaces
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from src.envs.engines import *
from src.opt.simplex_noise import SimplexNoise
from src.utils import e2q


class BuggyMaize():
    def __init__(self, config):
        self.config = config

        self.block_size = 1.0 # 1 meter block size
        self.n_blocks = 5
        self.grid_resolution = 0.05
        self.grid_X_len = int(self.block_size * (self.n_blocks + 2) / self.grid_resolution)
        self.grid_Y_len = int(self.block_size * (self.n_blocks * 2 + 1) / self.grid_resolution)
        self.grid_block_len = int(self.block_size / self.grid_resolution)
        self.barrier_halflength_coeff = 0.5
        self.barrier_halfwidth_coeff = 0.05

        self.reset()

    def generate_random_maize(self):
        # Make list of blocks which represent the maize
        cur_x, cur_y = self.block_size, 0
        blocks = [[0, 0], [cur_x, cur_y]]
        for i in range(self.n_blocks):
            while True:
                shift_in_x = random.randint(0, 1)
                cur_x_candidate = cur_x + shift_in_x * random.randint(-1, 1) * self.block_size
                cur_y_candidate = cur_y + (1 - shift_in_x) * random.randint(-1, 1) * self.block_size
                if [cur_x_candidate, cur_y_candidate] not in blocks and cur_x_candidate > 0:
                    break
            cur_x = cur_x_candidate
            cur_y = cur_y_candidate
            blocks.append([cur_x, cur_y])

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
        m = int(self.grid_X_len - x * self.grid_block_len - 0.5 * self.grid_block_len)
        n = int(0.5 * self.grid_Y_len - y * self.grid_block_len)

        m = np.clip(m, 0, self.grid_X_len)
        n = np.clip(n, 0, self.grid_Y_len)

        return m, n

    def grid_to_xy(self, m, n):
        x = self.grid_X_len / self.grid_block_len - 0.5 - m / self.grid_block_len
        y = 0.5 * self.grid_Y_len / self.grid_block_len - n / self.grid_block_len
        return x, y

    def plot_grid(self, grid, shortest_path_pts):
        grid_cpy = np.copy(grid)
        for pt in shortest_path_pts:
            grid_cpy[pt[0], pt[1]] = 10

        # Plot
        plt.imshow(grid_cpy)
        plt.show()

    def generate_shortest_path(self, grid, start, finish):
        grid = Grid(matrix=grid)
        start = grid.node(start[1],start[0])
        end = grid.node(finish[1], finish[0])
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, runs = finder.find_path(start, end, grid)
        #print('operations:', runs, 'path length:', len(path))
        #print(grid.grid_str(path=path, start=start, end=end))
        path_cpy = []
        for p in path:
            path_cpy.append((p[1], p[0]))
        return path_cpy

    def get_barriers(self):
        return self.all_barriers

    def get_shortest_path_pts_xy(self):
        sptxy = []
        for pt_x, pt_y in self.shortest_path_pts:
            sptxy.append(self.grid_to_xy(pt_x, pt_y))
        return sptxy

    def reset(self):
        self.blocks, self.all_barriers, self.dense_grid, self.start, self.finish = self.generate_random_maize()
        self.shortest_path_pts = self.generate_shortest_path(self.dense_grid, self.start, self.finish)
        # self.plot_grid(self.dense_grid, self.shortest_path_pts)

class BuggyMaizeEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 100
    }

    def __init__(self, config):
        self.config = config

        self.obs_dim = self.config["state_dim"] \
                       + self.config["n_traj_pts"] * 2 \
                       + self.config["allow_latent_input"] * self.config["latent_dim"] \
                       + self.config["allow_lte"]
        self.act_dim = 2

        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.act_dim,), dtype=np.float32)

        self.sim, self.engine = self.load_random_env()

        self.maize = BuggyMaize(config)

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

    def get_reward(self, obs_dict, wp_visited):
        pos = np.array(obs_dict["pos"], dtype=np.float32)
        cur_wp = np.array(self.engine.wp_list[self.engine.cur_wp_idx], dtype=np.float32)
        if self.engine.cur_wp_idx > 0:
            prev_wp = np.array(self.engine.wp_list[self.engine.cur_wp_idx - 1], dtype=np.float32)
        else:
            prev_wp = np.array(cur_wp, dtype=np.float32)

        path_deviation = np.abs((cur_wp[0] - prev_wp[0]) * (prev_wp[1] - pos[1]) - (prev_wp[0] - pos[0]) * (cur_wp[1] - prev_wp[1])) / np.sqrt(np.square(cur_wp[0] - prev_wp[0]) + np.square(cur_wp[1] - prev_wp[1]))
        dist_between_cur_wp = np.sqrt(np.square((cur_wp[0] - pos[0])) + np.square((cur_wp[1] - pos[1])))

        r = wp_visited * (1 / (1 + 0.5 * path_deviation)) - dist_between_cur_wp * 0.01
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
        done = done or dist_to_cur_wp > 0.7 or self.step_ctr > self.config["max_steps"]
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
            self.engine.noise.set_seed(seed)

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
        for r in rollout:
            pass
        # TODO: HERE

if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_maize_env_mujoco.yaml"))

    #bm = BuggyMaize(config)
    be = BuggyMaizeEnv(config)
    be.demo()
