import gym
import torch as T
from gym import spaces

from src.envs.engines import *
from src.envs.xml_gen import *
from src.policies import LTE
import mujoco_py
from src.opt.simplex_noise import SimplexNoise

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

        self.obs_dim = self.config["state_dim"] + self.config["n_traj_pts"] * 2 + self.config["allow_latent_input"] * \
                       self.config["latent_dim"] + self.config["allow_lte"]
        self.act_dim = 2

        if self.config["allow_lte"]:
            self.lte = LTE(obs_dim=self.config["state_dim"] + self.act_dim, act_dim=self.config["state_dim"])
            self.lte.load_state_dict(T.load("agents/buggy_lte.p"), strict=False)

        self.observation_space = spaces.Box(low=-7, high=7, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

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
            engine = MujocoEngine(self.config, sim)

        return sim, engine

    def get_obs_dict(self):
        return self.engine.get_obs_dict()

    def get_state_vec(self):
        return self.engine.get_state_vec()

    def get_complete_obs_vec(self):
        complete_obs_vec = self.engine.get_complete_obs_vec()
        if self.config["allow_latent_input"]:
            complete_obs_vec += self.scaled_random_params
        return complete_obs_vec

    def get_reward(self, obs_dict, wp_visited):
        pos = obs_dict["pos"]
        cur_wp = self.engine.wp_list[self.engine.cur_wp_idx]
        dist_between_cur_wp = np.sqrt(np.square(pos[0] - cur_wp[0]) + np.square(pos[1] - cur_wp[1]))
        r = wp_visited * 3 - dist_between_cur_wp * 0.05
        return r, dist_between_cur_wp

    def step(self, act):
        self.step_ctr += 1

        # Turn, throttle
        scaled_act = [np.clip(act[0] * 0.4, -0.4, 0.4), act[1] * 0.5 + 0.5]
        done, wp_visited = self.engine.step(scaled_act)

        # Get new observation
        obs_dict = self.engine.get_obs_dict()
        complete_obs_vec = self.engine.get_complete_obs_vec()

        # calculate reward
        r, dist_to_cur_wp = self.get_reward(obs_dict, wp_visited)

        # Calculate termination
        #done = done or dist_to_cur_wp > 0.5 or self.step_ctr > self.config["max_steps"]
        done = self.step_ctr > self.config["max_steps"]

        #if self.config["render"]:
        #    self.render()

        return complete_obs_vec, r, done, {}

    def reset(self):
        # Reset variables
        self.step_ctr = 0

        # Reset simulation
        self.engine.reset()

        # Reset environment variables
        return self.engine.get_complete_obs_vec()

    def render(self, mode=None):
        self.engine.render()

    def demo(self):
        while True:
            self.noise = SimplexNoise(dim=2, smoothness=30, multiplier=1.6)
            self.reset()
            cum_rew = 0
            while True:
                zero_act = [0.0, -1.0]
                rnd_act = np.clip(self.noise(), -1, 1)
                _, r, done, _ = self.step(rnd_act) # turn, throttle
                #print(r)
                cum_rew += r
                if self.config["render"]:
                    self.engine.render()
                    time.sleep(1. / self.config["rate"])

                if done: break
            print("Cumulative rew: {}".format(cum_rew))

if __name__ == "__main__":
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))
    be = BuggyEnv(config)
    be.demo()
