import gym
import torch as T
from gym import spaces

from src.envs.engines import *
from src.envs.xml_gen import *

import mujoco_py
from src.opt.simplex_noise import SimplexNoise
from src.utils import e2q
import timeit
import random

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class BuggyMaize():
    def __init__(self, config):
        self.config = config

        self.block_size = 1 # 1 meter block size
        self.n_blocks = 5
        self.grid_resolution = 0.05

        self.blocks, self.all_barriers, self.dense_grid, self.start, self.finish = self.generate_random_maize()
        self.shortest_path = self.generate_shortest_path(self.dense_grid, self.start, self.finish)

    def generate_random_maize(self):
        # Make list of blocks which represent the maize
        cur_x, cur_y = 1, 0
        blocks = [[0, 0], [cur_x, cur_y]]
        for i in range(self.n_blocks):
            while True:
                cur_x_candidate = cur_x + random.randint(-1, 1)
                cur_y_candidate = cur_y + random.randint(-1, 1)
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
            barriers = [(bl_x + self.block_size / 2, bl_y + self.block_size / 2, False),
                        (bl_x - self.block_size / 2, bl_y - self.block_size / 2, False),
                        (bl_x + self.block_size / 2, bl_y + self.block_size / 2, True),
                        (bl_x - self.block_size / 2, bl_y - self.block_size / 2, True)] # True when vertical

            # Filter barriers according to previous neighbors
            if i > 0:
                p_bl_x, p_bl_y = blocks[i - 1]
                if p_bl_x > bl_x: del barriers[0]
                if p_bl_x < bl_x: del barriers[1]
                if p_bl_x > bl_y: del barriers[2]
                if p_bl_x < bl_x: del barriers[3]
            all_barriers.append(barriers)

        # Make dense grid representation
        n_squares_per_meter = int(1. / self.grid_resolution)
        dense_grid = np.zeros(int((self.n_blocks + 2) * n_squares_per_meter),
                              int(2 * self.n_blocks * n_squares_per_meter))
        grid_offset_x, grid_offset_y = int(self.block_size * n_squares_per_meter), int(self.n_blocks * n_squares_per_meter)
        grid_barrier_half_length = int(0.5 * self.block_size * n_squares_per_meter)
        grid_barrier_half_width = int(0.5 * self.block_size * 5)
        for bar in all_barriers:
            bar_x, bar_y, vert = bar

            x_side = grid_barrier_half_width
            y_side = grid_barrier_half_length
            if vert:
                x_side = grid_barrier_half_length
                y_side = grid_barrier_half_width

            # Add barrier to grid
            dense_grid[
            bar_x * n_squares_per_meter + grid_offset_x - x_side: bar_x * n_squares_per_meter + grid_offset_x + x_side,
            bar_y * n_squares_per_meter + grid_offset_y - y_side: bar_y * n_squares_per_meter + grid_offset_y + y_side] = 1

        start = blocks[0][0] * n_squares_per_meter, blocks[0][1] * n_squares_per_meter
        finish = blocks[-1][0] * n_squares_per_meter, blocks[-1][1] * n_squares_per_meter

        return blocks, all_barriers, dense_grid, start, finish

    def generate_shortest_path(self, grid, start, finish):
        grid = Grid(matrix=grid)
        start = grid.node(*start)
        end = grid.node(*finish)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, runs = finder.find_path(start, end, grid)
        print('operations:', runs, 'path length:', len(path))
        print(grid.grid_str(path=path, start=start, end=end))

    def get_barriers(self):
        return self.all_barriers

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
        p1, p2 = barriers_list
        # TODO: This will accept a whole list of rectangular (ellipsoidal) barriers for the maize env. Pos + orientation
        self.sim.data.set_mocap_pos("barrier1", p1 + [0.2])
        self.sim.data.set_mocap_pos("barrier2", p2 + [0.2])

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
        self.engine.generate_random_traj(traj_pts=self.maize.shortest_path_pts)

        # Draw the barrier visuals
        self.engine.draw_barriers(self.maize.get_barriers())

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

            self.set_barrier_positions([[4.0, 0.0], [6.0, 1.0]])
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

if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_maize_env_mujoco.yaml"))
    be = BuggyMaizeEnv(config)
    be.demo()
