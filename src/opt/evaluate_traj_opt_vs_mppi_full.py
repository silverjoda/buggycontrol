from copy import deepcopy

from stable_baselines3 import A2C

from src.ctrl.control_buggy_mppi import ControlBuggyMPPI
from src.envs.buggy_maize_env_mujoco import BuggyMaizeEnv
from src.policies import *
from src.utils import *
from src.opt.traj_tep_optimizer import TrajTepOptimizer
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecMonitor
from tabulate import tabulate

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
T.set_num_threads(1)

class BuggyControlTester:
    def __init__(self):
        # Load RL agent(s)
        with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_a2c.yaml"), 'r') as f:
            self.algo_config = yaml.load(f, Loader=yaml.FullLoader)
        self.buggy_rl_policy = A2C.load("agents/{}_SB_policy".format(self.algo_config["default_session_ID"]))

        # Make buggy env
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_maize_env_mujoco.yaml"), 'r') as f:
            self.buggy_config = yaml.load(f, Loader=yaml.FullLoader)
        self.buggy_maize_env = BuggyMaizeEnv(self.buggy_config)
        vec_env = DummyVecEnv(env_fns=[lambda: BuggyMaizeEnv(self.buggy_config)])
        monitor_env = VecMonitor(vec_env)
        normed_env = VecNormalize(venv=monitor_env, training=False, norm_obs=True, norm_reward=True, clip_reward=10.0)
        self.buggy_maize_venv = VecNormalize.load("agents/{}_vecnorm.pkl".format(self.algo_config["default_session_ID"]), normed_env)

        # Make Mppi algo
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ctrl/configs/control_buggy_mppi.yaml"), 'r') as f:
            self.mppi_config = yaml.load(f, Loader=yaml.FullLoader)
        self.mppi_algo = ControlBuggyMPPI(self.mppi_config)

        # Load tep(s)
        self.tep = TEPMLP(obs_dim=50, act_dim=1)
        self.tep_1step = TEPMLP(obs_dim=50, act_dim=1)
        self.tep.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)
        self.tep_1step.load_state_dict(T.load("agents/full_traj_tep_1step.p"), strict=False)

        # Load trajectory optimizer
        self.traj_tep_optimizer = TrajTepOptimizer()

    def single_control_algo_evaluation(self, seed, render=False, plot=False):
        # Reset env and record trajectory
        self.buggy_maize_env.seed(seed)
        self.buggy_maize_env.reset()
        test_traj = deepcopy(self.buggy_maize_env.engine.wp_list)

        # Test buggy agent on default shortest path
        print("Testing agent on default traj")
        def_rl_agent_res = self.test_rl_agent(self.buggy_maize_env, self.buggy_maize_venv, seed, test_traj, render=render, deterministic=self.algo_config["deterministic_eval"])

        # Update the shortest path using def tep
        traj_T_sar, _ = T.tensor(self.traj_tep_optimizer.xy_to_sar(test_traj[:50]), dtype=T.float32)
        updated_traj_T_sar = self.traj_tep_optimizer.optimize_traj_with_barriers(traj_T_sar, self.tep, self.buggy_maize_env)
        updated_traj = list(self.traj_tep_optimizer.sar_to_xy(updated_traj_T_sar).detach().numpy())

        # Tack on the rest of the points from original trajectory which weren't optimized (sort of a hack for now)
        updated_traj.extend(test_traj[50:])

        # Test buggy agent on new path
        print("Testing agent on 1step traj")
        updated_traj_rl_agent_res = self.test_rl_agent(self.buggy_maize_env, self.buggy_maize_venv, seed, updated_traj, render=render, deterministic=self.algo_config["deterministic_eval"])

        # Update the shortest path using 1step tep
        traj_T_sar, _ = T.tensor(self.traj_tep_optimizer.xy_to_sar(test_traj[:50]), dtype=T.float32)
        updated_1step_traj_T_sar = self.traj_tep_optimizer.optimize_traj_with_barriers(traj_T_sar, self.tep_1step, self.buggy_maize_env)
        updated_1step_traj = list(self.traj_tep_optimizer.sar_to_xy(updated_1step_traj_T_sar).detach().numpy())

        # Tack on the rest of the points from original trajectory which weren't optimized (sort of a hack for now)
        updated_1step_traj.extend(test_traj[50:])

        # Test buggy agent on new path
        print("Testing agent on agg 1step traj")
        updated_1step_traj_rl_agent_res = self.test_rl_agent(self.buggy_maize_env, self.buggy_maize_venv, seed, updated_1step_traj, render=render, deterministic=self.algo_config["deterministic_eval"])

        # Test mppi: traj/free
        print("Testing mppi traj follower")
        mppi_traj_follower_res = self.mppi_algo.test_mppi(self.buggy_maize_env, seed=seed, test_traj=test_traj, n_samples=5,
                                                          n_horizon=5, act_std=0.5, mode='traj', render=False)

        print("Testing mppi free rew")
        mppi_free_res = self.mppi_algo.test_mppi(self.buggy_maize_env, seed=seed, test_traj=test_traj, n_samples=5,
                                                 n_horizon=5, act_std=0.5, mode='free', render=False)

        trajs = (test_traj, updated_traj, updated_1step_traj)

        if plot:
            #self.traj_tep_optimizer.plot_trajs3(test_traj,updated_traj, updated_1step_traj)
            self.buggy_maize_env.maize.plot_grid_traj3(self.buggy_maize_env.maize.dense_grid, trajs, self.buggy_maize_env.maize.get_barrier_edgepoints())

        return def_rl_agent_res, updated_traj_rl_agent_res, updated_1step_traj_rl_agent_res, mppi_traj_follower_res, mppi_free_res, trajs

    def test_system(self, render=False, plot=False):
        N_test_iters = 3
        seeds = np.arange(N_test_iters) + 1337
        #seeds = np.random.randint(0, 1000, N_test_iters)

        traj_list = []

        def_rl_agent_res_list, updated_traj_rl_agent_res_list, updated_1step_traj_rl_agent_res_list, mppi_traj_follower_res_list, mppi_free_res_list = [], [], [], [], []
        for i in range(N_test_iters):
            print(f"Test iter: {i+1}/{N_test_iters}")
            def_rl_agent_res, updated_traj_rl_agent_res, updated_1step_traj_rl_agent_res, mppi_traj_follower_res, mppi_free_res, trajs = self.single_control_algo_evaluation(seeds[i], render=render, plot=plot)
            #def_rl_agent_res, updated_traj_rl_agent_res, mppi_traj_follower_res, mppi_free_res = np.random.rand(2),np.random.rand(2),np.random.rand(2),np.random.rand(2)

            def_rl_agent_res_list.append(def_rl_agent_res)
            updated_traj_rl_agent_res_list.append(updated_traj_rl_agent_res)
            updated_1step_traj_rl_agent_res_list.append(updated_1step_traj_rl_agent_res)
            mppi_traj_follower_res_list.append(mppi_traj_follower_res)
            mppi_free_res_list.append(mppi_free_res)

            traj_list.append(trajs)

        # Calculate average rew and time taken
        def_rl_agent_rew_avg = 0
        def_rl_agent_time_taken_avg = 0

        updated_traj_rl_agent_rew_avg = 0
        updated_traj_rl_agent_time_taken_avg = 0

        updated_1step_traj_rl_agent_rew_avg = 0
        updated_1step_traj_rl_agent_time_taken_avg = 0

        mppi_traj_follower_rew_avg = 0
        mppi_traj_follower_time_taken_avg = 0

        mppi_free_rew_avg = 0
        mppi_free_time_taken_avg = 0

        for i in range(N_test_iters):
            def_rl_agent_rew_avg += def_rl_agent_res_list[i][0]
            def_rl_agent_time_taken_avg += def_rl_agent_res_list[i][1]

            updated_traj_rl_agent_rew_avg += updated_traj_rl_agent_res_list[i][0]
            updated_traj_rl_agent_time_taken_avg += updated_traj_rl_agent_res_list[i][1]

            updated_1step_traj_rl_agent_rew_avg += updated_1step_traj_rl_agent_res_list[i][0]
            updated_1step_traj_rl_agent_time_taken_avg += updated_1step_traj_rl_agent_res_list[i][1]

            mppi_traj_follower_rew_avg += mppi_traj_follower_res_list[i][0]
            mppi_traj_follower_time_taken_avg += mppi_traj_follower_res_list[i][1]

            mppi_free_rew_avg += mppi_free_res_list[i][0]
            mppi_free_time_taken_avg += mppi_free_res_list[i][1]

        # Print out results
        table = [['Def_rl_agent', def_rl_agent_rew_avg, def_rl_agent_time_taken_avg],
                 ['Updated_traj_rl_agent', updated_traj_rl_agent_rew_avg, updated_traj_rl_agent_time_taken_avg],
                 ['Updated_1step_traj_rl_agent', updated_1step_traj_rl_agent_rew_avg, updated_1step_traj_rl_agent_time_taken_avg],
                 ['Mppi_traj_follower', mppi_traj_follower_rew_avg, mppi_traj_follower_time_taken_avg],
                 ['Mppi_free', mppi_free_rew_avg, mppi_free_time_taken_avg]]
        print(tabulate(table, headers=['Alg', 'Avg rew', 'Avg time taken']))

        self.plot_multiple_trajs(traj_list)

    def test_rl_agent(self, env, venv, seed, test_traj, deterministic=True, print_rew=False, render=False):
        env.seed(seed)
        obs = env.reset()
        obs = venv.normalize_obs(obs)
        env.engine.set_trajectory(list(test_traj))

        episode_rew = 0
        step_ctr = 0
        while True:
            action, _states = self.buggy_rl_policy.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            obs = venv.normalize_obs(obs)
            episode_rew += reward
            step_ctr += 1
            if render:
                env.render()
            if done:
                if print_rew:
                    print(episode_rew)
                break

        time_taken = step_ctr * 0.01
        return episode_rew, time_taken

    def plot_multiple_trajs(self, traj_list):
        # Plot several random positional trajectories and velocities
        n_traj = len(traj_list)
        N_plot = np.minimum(5, n_traj)

        rnd_indeces = np.random.choice(np.arange(n_traj), N_plot, replace=False)

        fig, axs = plt.subplots(1, N_plot)
        for i in rnd_indeces:
            traj_def, traj_1step, traj_agg_1step = traj_list[i]
            axs[i].plot(list(zip(*traj_def))[0], list(zip(*traj_def))[1], marker="o", color="r", label='def', markersize=3)
            axs[i].plot(list(zip(*traj_1step))[0], list(zip(*traj_1step))[1], marker="o", color="g", label='1st', markersize=3)
            axs[i].plot(list(zip(*traj_agg_1step))[0], list(zip(*traj_agg_1step))[1], marker="o", color="b", label='1st_agg', markersize=3)

        fig.tight_layout()
        plt.show()

if __name__=="__main__":
    bct = BuggyControlTester()
    #bct.single_control_algo_evaluation(1337)
    bct.test_system(render=False, plot=True)
