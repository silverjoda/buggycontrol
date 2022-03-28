import gym
import torch as T
from gym import spaces

from src.envs.engines import *
from src.envs.xml_gen import *
from src.policies import LTE
import mujoco_py
from src.opt.simplex_noise import SimplexNoise
from src.utils import e2q
import timeit

class BuggyEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        "video.frames_per_second": 100
    }

    def __init__(self, config):
        self.config = config
        self.buddy_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/cars/base_car/buddy.xml")
        self.buddy_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/cars/base_car/buddy_rnd.xml")

        self.car_template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car.xml")
        self.car_rnd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/models/one_car_rnd.xml")

        n_traj_obs = self.config["n_traj_pts"] * 2 #(3 - self.config["use_engine_2"])
        self.obs_dim = self.config["state_dim"] + n_traj_obs + self.config["allow_latent_input"] * \
                       self.config["latent_dim"] + self.config["allow_lte"]
        self.act_dim = 2

        if self.config["allow_lte"]:
            self.lte = LTE(obs_dim=self.config["state_dim"] + self.act_dim, act_dim=self.config["state_dim"])
            self.lte.load_state_dict(T.load("agents/buggy_lte.p"), strict=False)

        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.act_dim,), dtype=np.float32)
        self.current_difficulty = 0.

        self.sim, self.engine = self.load_random_env()

    def load_random_env(self):
        # 0: friction (0.4), 1: steering_range (0.38), 2: body mass (3.47), 3: kv (3000), 4: gear (0.003)
        random_param_scale_offset_list = [[0.15, 0.4], [0.1, 0.38], [1, 3.5], [1000, 3000], [0.001, 0.003]]
        self.scaled_random_params = list(np.clip(np.random.randn(len(random_param_scale_offset_list)) * 0.5, -1, 1))
        self.sim_random_params = [self.scaled_random_params[i] * rso[0] + rso[1] for i, rso in enumerate(random_param_scale_offset_list)]

        if self.config["allow_lte"]:
            self.random_params = np.concatenate([self.sim_random_params, np.array([-1])])

        if self.config["allow_lte"] and np.random.rand() < self.config["lte_prob"]:
            self.scaled_random_params = [0,0,0,0,0,1.]
            self.sim_random_params = [0,0,0,0,0,1.]
            model = mujoco_py.load_model_from_path(self.car_template_path)
            sim = mujoco_py.MjSim(model, nsubsteps=self.config['n_substeps'])
            engine = LTEEngine(self.config, sim, self.lte)
        else:
            if self.config["randomize_env"]:
                buddy_xml = gen_buddy_xml(self.random_params)
                with open(self.buddy_rnd_path, "w") as out_file:
                    for s in buddy_xml.splitlines():
                        out_file.write(s)
                car_xml = gen_car_xml(self.random_params)
                model = mujoco_py.load_model_from_xml(car_xml)
                # with open(self.car_rnd_path, "w") as out_file:
                #     for s in car_xml.splitlines():
                #         out_file.write(s)
                # model = mujoco_py.load_model_from_path(self.car_rnd_path)
            else:
                model = mujoco_py.load_model_from_path(self.car_template_path)
            sim = mujoco_py.MjSim(model, nsubsteps=self.config['n_substeps'])
            if self.config["use_engine_2"]:
                engine = MujocoEngine2(self.config, sim)
            else:
                engine = MujocoEngine(self.config, sim)

        return sim, engine

    def set_barrier_positions(self, p1, p2):
        self.sim.data.set_mocap_pos("barrier1", p1 + [0.2])
        self.sim.data.set_mocap_pos("barrier2", p2 + [0.2])

    def set_trajectory_pts(self, traj_pts):
        pass

    def get_obs_dict(self):
        return self.engine.get_obs_dict()

    def get_state_vec(self):
        return self.engine.get_state_vec()

    def get_complete_obs_vec(self):
        complete_obs_vec, _ = self.engine.get_complete_obs_vec()
        if self.config["allow_latent_input"]:
            complete_obs_vec += self.scaled_random_params
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
        #r = wp_visited * 1
        return r, dist_between_cur_wp

    def is_mirror_obs(self, obs):
        return obs[3] < 0

    def mirror_obs(self, obs):
        mirrored_obs = deepcopy(obs)

        # Wheel turn obs
        mirrored_obs[0] *= -1

        # y obs
        mirrored_obs[3] *= -1
        mirrored_obs[5] *= -1

        # Waypoint y coordinate
        n_wpts = (len(obs) - 5) // 2
        for i in range(n_wpts):
            mirrored_obs[5 + i * 2 + 1] *= -1
        return mirrored_obs

    def set_external_state(self, state_dict):
        old_state = self.sim.get_state()
        qpos = old_state.qpos
        qpos[0:2] = state_dict["x_pos"], state_dict["y_pos"]
        quat = e2q(0,0,state_dict["phi"])
        qpos[3:7] = quat
        new_state = mujoco_py.MjSimState(old_state.time, qpos, old_state.qvel,
                                         old_state.act, old_state.udd_state)

        self.sim.set_state(new_state)
        self.sim.forward()

    def step(self, act):
        self.step_ctr += 1

        if self.config["enforce_bilateral_symmetry"]:
            if self.prev_obs_mirrored:
                act = [act[0] * -1, act[1]]

        # Turn, throttle
        scaled_act = [np.clip(act[0] * 0.2, -0.42, 0.42), np.clip(act[1] * 0.5 + 0.5, 0, 1)]
        done, wp_visited = self.engine.step(scaled_act)

        # Get new observation
        complete_obs_vec, obs_dict = self.engine.get_complete_obs_vec()

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
        self.current_difficulty = np.minimum(self.current_difficulty + 0.00003, 1.)
        self.engine.current_difficulty = self.current_difficulty
        self.prev_scaled_act = np.zeros(2)

        # Reset simulation
        self.engine.reset()

        # Reset environment variables
        obs_vec, _ = self.engine.get_complete_obs_vec()

        if self.config["enforce_bilateral_symmetry"]:
            self.prev_obs_mirrored = self.is_mirror_obs(obs_vec)
            if self.prev_obs_mirrored:
                obs_vec = self.mirror_obs(obs_vec)

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

            self.set_barrier_positions([4.0, 0.0], [6.0, 1.0])
            cum_rew = 0
            while True:
                zero_act = np.array([-1.0, -1.0])
                rnd_act = np.clip(self.noise(), -1, 1)
                obs, r, done, _ = self.step(rnd_act) # turn, throttle
                cum_rew += r
                if self.config["render"]:
                    self.engine.render()
                    time.sleep(1. / self.config["rate"])

                print(obs)

                if done: break
            print("Cumulative rew: {}".format(cum_rew))

if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
    be = BuggyEnv(config)
    be.demo()
