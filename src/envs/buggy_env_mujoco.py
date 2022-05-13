import gym
import torch as T
from gym import spaces

from src.envs.engines import *

import mujoco_py
from src.opt.simplex_noise import SimplexNoise
from src.utils import e2q

class BuggyEnv(gym.Env):
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

    def load_random_env(self):
        if self.config["allow_lte"] and np.random.rand() < self.config["lte_prob"]:
            engine = LTEEngine(self.config)
        elif self.config["use_hybrid_engine"]:
            engine = HybridEngine(self.config)
        else:
            engine = MujocoEngine(self.config)
        sim = engine.mujoco_sim
        return sim, engine

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

if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
    be = BuggyEnv(config)
    be.demo()
