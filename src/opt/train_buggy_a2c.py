import os
import random
import time
from pprint import pprint

import torch as T
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecMonitor

from src.envs import buggy_env_mujoco
from src.utils import merge_dicts


class BuggyTrajFollowerTrainer:
    def __init__(self):

        self.config = self.read_configs()
        self.env_fun = buggy_env_mujoco.BuggyEnv
        self.N_cores = 6

        self.env, self.model, self.checkpoint_callback, self.stats_path = self.setup_train()

        if self.config["train"]:
            t1 = time.time()
            try:
                self.model.learn(total_timesteps=self.config["iters"], callback=self.checkpoint_callback,
                                 log_interval=1)
            except KeyboardInterrupt:
                print("User interrupted training procedure")
            t2 = time.time()

            print("Training time: {}".format(t2 - t1))
            pprint(self.config)

            self.env.save(self.stats_path)
            self.model.save("agents/{}_SB_policy".format(self.config["session_ID"]))
            self.env.close()

        if not self.config["train"]:
            self.model = A2C.load("agents/{}_SB_policy".format(self.config["session_ID"]))

            vec_env = DummyVecEnv(env_fns=[lambda: self.env_fun(self.config)] * 1)
            monitor_env = VecMonitor(vec_env)
            normed_env = VecNormalize(venv=monitor_env, training=False, norm_obs=True, norm_reward=True, clip_reward=10.0)
            self.env = VecNormalize.load(self.stats_path, normed_env)

            N_test = 100
            total_rew = self.test_agent(deterministic=True, N=N_test)
            print(f"Total test rew: {total_rew / N_test}")

    def read_configs(self):
        with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_a2c.yaml"), 'r') as f:
            algo_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
            env_config = yaml.load(f, Loader=yaml.FullLoader)

        config = merge_dicts([algo_config, env_config])
        return config

    def setup_train(self, setup_dirs=True):
        T.set_num_threads(1)
        if setup_dirs:
            for s in ["agents", "agents_cp", "tb", "logs"]:
                if not os.path.exists(s):
                    os.makedirs(s)

        # Random ID of this session
        if self.config["default_session_ID"] is None:
            self.config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
        else:
            self.config["session_ID"] = self.config["default_session_ID"]

        #vec_env = DummyVecEnv(env_fns=[lambda : elf.env_fun(self.config)] * self.N_cores)
        vec_env = SubprocVecEnv(env_fns=[lambda : self.env_fun(self.config) for _ in range(self.N_cores)] , start_method="fork")
        monitor_env = VecMonitor(vec_env)
        normed_env = VecNormalize(venv=monitor_env, training=True, norm_obs=True, norm_reward=True)

        stats_path = "agents/{}_vecnorm.pkl".format(self.config["session_ID"])
        checkpoint_callback = CheckpointCallback(save_freq=300000,
                                                 save_path='agents_cp/',
                                                 name_prefix=self.config["session_ID"], verbose=1)

        #eval_env = gym.make('Pendulum-v1')
        #callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
        #eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)

        model = A2C(policy=self.config["policy_name"],
                    env=normed_env,
                    gamma=self.config["gamma"],
                    n_steps=self.config["n_steps"],
                    vf_coef=self.config["vf_coef"],
                    ent_coef=self.config["ent_coef"],
                    max_grad_norm=self.config["max_grad_norm"],
                    learning_rate= self.config["learning_rate"],
                    verbose=self.config["verbose"],
                    device="cpu",
                    policy_kwargs=dict(net_arch=[self.config["policy_hid_dim"], self.config["policy_hid_dim"]]))

        callback_list = CallbackList([checkpoint_callback])

        return normed_env, model, callback_list, stats_path

    def test_agent(self, deterministic=True, N=100, print_rew=True, render=True):
        total_rew = 0

        for _ in range(N):
            obs = self.env.reset()
            episode_rew = 0
            while True:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                episode_rew += self.env.get_original_reward()
                total_rew += self.env.get_original_reward()
                if render:
                    self.env.render()
                if done:
                    if print_rew:
                        print(episode_rew)
                    break
        return total_rew

if __name__ == "__main__":
    trainer = BuggyTrajFollowerTrainer()
