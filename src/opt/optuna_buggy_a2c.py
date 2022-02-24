import os
import random
import time

import optuna
import torch as T
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor, DummyVecEnv

from src.envs import buggy_env_mujoco
from src.utils import merge_dicts

def objective(trial, config):
    #config["learning_rate"] = "lambda x : x * {}".format(trial.suggest_uniform('learning_rate', 1e-4, 5e-3))
    config["learning_rate"] = trial.suggest_uniform('learning_rate', 1e-4, 5e-3)
    config["gamma"] = trial.suggest_loguniform('gamma', 0.96, 0.999)
    config["ent_coef"] = trial.suggest_loguniform('ent_coef', 0.0000001, 0.0001)
    config["max_grad_norm"] = trial.suggest_uniform('max_grad_norm', 0.3, 0.9)
    config["n_steps"] = trial.suggest_int('n_steps', 10, 50)
    config["max_steps"] = trial.suggest_int('max_steps', 300, 500)

    env, model, stats_path = setup_train(config, buggy_env_mujoco.BuggyEnv)
    model.learn(total_timesteps=config["iters"])

    #model = A2C.load("agents/{}_SB_policy".format(config["session_ID"]))
    #env = VecNormalize.load("agents/{}_vecnorm.pkl".format(config["session_ID"]), env)

    env.training = False
    avg_episode_rew = test_agent(env, model, n_steps=400, n_eval=config["n_eval"])

    env.close()
    del env
    del model

    return avg_episode_rew

def setup_train(config, env_fun, setup_dirs=True):
    T.set_num_threads(1)
    if setup_dirs:
        for s in ["agents", "agents_cp", "tb", "logs"]:
            if not os.path.exists(s):
                os.makedirs(s)

    # Random ID of this session
    if config["default_session_ID"] is None:
        config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
    else:
        config["session_ID"] = config["default_session_ID"]

    # vec_env = DummyVecEnv(env_fns=[lambda : elf.env_fun(self.config)] * self.N_cores)
    vec_env = SubprocVecEnv(env_fns=[lambda: env_fun(config) for _ in range(4)],
                            start_method="fork")
    monitor_env = VecMonitor(vec_env)
    normed_env = VecNormalize(venv=monitor_env, training=True, norm_obs=True, norm_reward=True)

    stats_path = "agents/{}_vecnorm.pkl".format(config["session_ID"])

    model = A2C(policy=config["policy_name"],
                env=normed_env,
                gamma=config["gamma"],
                n_steps=config["n_steps"],
                vf_coef=config["vf_coef"],
                ent_coef=config["ent_coef"],
                max_grad_norm=config["max_grad_norm"],
                learning_rate=config["learning_rate"],
                verbose=config["verbose"],
                device="cpu",
                policy_kwargs=dict(net_arch=[config["policy_hid_dim"], config["policy_hid_dim"]]))

    return normed_env, model, stats_path

def test_agent(env, model,n_steps=400, n_eval=100):
    total_rew = 0
    for it in range(n_eval):
        obs = env.reset()
        for i in range(n_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if hasattr(env, "get_original_reward"):
                reward = env.get_original_reward()
            total_rew += reward

    return total_rew.mean() / n_eval

def read_configs():
    with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_a2c.yaml"), 'r') as f:
        algo_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"), 'r') as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)

    config = merge_dicts([algo_config, env_config])
    return config

if __name__ == "__main__":
    configs = read_configs()

    configs["iters"] = 500000
    configs["verbose"] = False
    #configs["render"] = False
    configs["tensorboard_log"] = False

    configs["n_eval"] = 100
    N_trials = 100

    t1 = time.time()
    study = optuna.create_study(direction='maximize', study_name="buggy_opt_study", storage='sqlite:///buggy_opt.db', load_if_exists=True)

    study.optimize(lambda x : objective(x, configs), n_trials=N_trials, show_progress_bar=True)
    t2 = time.time()
    print("Time taken: ", t2-t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)