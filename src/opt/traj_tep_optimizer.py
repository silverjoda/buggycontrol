import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import torch.nn
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
from tabulate import tabulate

from src.envs.buggy_env_mujoco import BuggyEnv
from src.policies import *
from src.utils import load_config, dist_between_wps

os.environ["CUDA_VISIBLE_DEVICES"] = ""
T.set_num_threads(1)

class TrajTepOptimizer:
    def __init__(self, policy_ID="TRN"):
        with open(os.path.join(os.path.dirname(__file__), "configs/traj_tep_optimizer.yaml"), 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy_ID = policy_ID
        self.n_dataset_pts = self.config["n_dataset_pts"]
        self.max_num_wp = self.config["max_num_wp"]

        dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/buggy_tep_dataset")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if self.config["plot_trajs"]:
            plt.ion()

        self.x_file_path = os.path.join(dir_path, "X.npy")
        self.y_file_path = os.path.join(dir_path, "Y.npy")

        #self.env, self.venv, self.sb_model = self.load_model_and_env()

    def load_model_and_env(self):
        env_config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/configs/buggy_env_mujoco.yaml"))

        # Policy + VF
        sb_model = A2C.load(f"agents/{self.policy_ID}_SB_policy")

        # Wrapped env
        env = BuggyEnv(env_config)
        vec_env = DummyVecEnv(env_fns=[lambda: env] * 1)
        monitor_env = VecMonitor(vec_env)
        normed_env = VecNormalize(venv=monitor_env, training=False, norm_obs=True, norm_reward=True, clip_reward=10.0)
        stats_path = f"agents/{self.policy_ID}_vecnorm.pkl"
        venv = VecNormalize.load(stats_path, normed_env)

        return env, venv, sb_model

    def make_dataset(self, render=True):
        print("Starting dataset creation")
        obs_list = []
        rew_list = []
        for i in range(self.n_dataset_pts):
            obs = self.env.reset()
            traj = [item for sublist in self.env.engine.wp_list[:self.max_num_wp] for item in sublist]
            episode_rew = self.evaluate_rollout(obs, self.env, self.venv, self.sb_model, traj=None, render=False, deterministic=self.config["deterministic_eval"])
            # #DEBUG TO SEE HOW CONSISTENT EVALUATION IS
            # traj_raw = deepcopy(self.env.engine.wp_list)
            # print("NEW TRAJ")
            # for j in range(10):
            #     det = j < 5
            #     episode_rew = self.evaluate_rollout(obs, traj=None, render=False, deterministic=self.config["deterministic_eval"])
            #     print(f"Deterministic: {det}", episode_rew)
            #     obs = self.env.reset()
            #     self.env.engine.wp_list = traj_raw
            obs_list.append(traj)
            rew_list.append(episode_rew)

            if i % 10 == 0:
                print(f"Iter: {i}/{self.n_dataset_pts}")

        obs_arr = np.array(obs_list, dtype=np.float32)
        rew_arr = np.array(rew_list, dtype=np.float32)

        np.save(self.x_file_path, obs_arr)
        np.save(self.y_file_path, rew_arr)

        return obs_arr, rew_arr

    def evaluate_rollout(self, obs, env, venv, sb_policy, traj=None, distances=None, render=False, deterministic=True):
        if traj is not None:
            env.engine.wp_list = list(traj)

        obs = venv.normalize_obs(obs[0])
        episode_rew = 0
        step_ctr = 0
        for i in range(self.config["max_steps"]):
            step_ctr += 1
            action, _states = sb_policy.predict(obs, deterministic=deterministic)
            obs, reward, _, info = env.step(action)
            obs = venv.normalize_obs(obs)
            episode_rew += reward
            if render:
                env.render()
            if env.engine.cur_wp_idx > 30:
                break
        return step_ctr * 0.01

    def get_successive_angle_representation(self, X):
        X_new = np.zeros((X.shape[0], X.shape[1] // 2))
        X_new[:, 0] = np.arctan2(X[:, 1], X[:, 0])
        for i in range(1, X_new.shape[1]):
            X_new[:, i] = np.arctan2(X[:, i * 2 + 1] - X[:, (i - 1) * 2 + 1], X[:, i * 2] - X[:, (i - 1) * 2])
        return X_new

    def get_delta_representation(self, X):
        X_new = np.zeros_like(X)
        X_new[:, :2] = X[:, :2]
        X_new[:, 2:] = X[:, 2:] - X[:, :-2]
        return X_new

    def train_tep(self):
        # Load dataset
        X = np.load(self.x_file_path, allow_pickle=True)
        Y = np.load(self.y_file_path)
        Y = np.expand_dims(Y, 1)

        # Change X to relative coordinates
        #X = self.get_delta_representation(X)

        # Change to successive angle representation
        X = self.get_successive_angle_representation(X)

        # Prepare policy and training
        tep = TEPMLP(obs_dim=X.shape[1], act_dim=1)
        #tep = TEPMLPDEEP(obs_dim=X.shape[1], act_dim=1)

        #RNN
        #tep = TEPRNN(n_waypts=X.shape[1] // 2, hid_dim=64, hid_dim_2=32, num_layers=1, bidirectional=False)
        #tep = TEPRNN2(n_waypts=X.shape[1], hid_dim=64, hid_dim_2=32, num_layers=1, bidirectional=False)

        # emb_dim = 36
        #tep = TEPTX(n_waypts=X.shape[1] // 2, embed_dim=36, num_heads=6, kdim=36)
        tep_optim = T.optim.Adam(params=tep.parameters(),
                                    lr=self.config['policy_lr'],
                                    weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()

        for i in range(self.config["trn_iters"]):
            rnd_start_idx = np.random.randint(low=0, high=len(X) - self.config["batchsize"] - 1)
            x = X[rnd_start_idx:rnd_start_idx + self.config["batchsize"]]
            y = Y[rnd_start_idx:rnd_start_idx + self.config["batchsize"]]

            x_T = T.tensor(x, dtype=T.float32)
            y_T = T.tensor(y, dtype=T.float32)

            y_ = tep(x_T)
            tep_loss = lossfun(y_, y_T)
            tep_loss.backward()
            tep_optim.step()
            tep_optim.zero_grad()

            if i % 50 == 0:
                print(
                    "Iter {}/{}, policy_loss: {}".format(i, self.config['trn_iters'], tep_loss.data))
        print("Done training, saving model")
        if not os.path.exists("agents"):
            os.makedirs("agents")
        T.save(tep.state_dict(), "agents/full_traj_tep.p")

    def train_tep_1step_grad(self):
        raise NotImplementedError
        self.env, self.venv, self.sb_model = self.load_model_and_env()

        # Load pretrained tep
        tep = TEPMLP(obs_dim=50, act_dim=1)
        tep.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

        # Core dataset
        X = np.load(self.x_file_path, allow_pickle=True)[:1000]
        Y = np.load(self.y_file_path)[:1000]

        # Change to successive angle representation
        X = T.tensor(self.get_successive_angle_representation(X), dtype=T.float32)
        Y = T.tensor(Y, dtype=T.float32)
        Y = Y.unsqueeze(1)

        # Make updated dataset
        X_ud = X.clone().detach()
        Y_ud = Y.clone().detach()

        # Prepare policy and training
        policy_optim = T.optim.Adam(params=tep.parameters(),
                                    lr=self.config['policy_lr'],
                                    weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()

        n_epochs = 3
        n_data = len(X)
        for ep in range(n_epochs):
            # Do single epoch
            rnd_indeces = np.arange(n_data)
            np.random.shuffle(rnd_indeces)
            n_reps = 10
            for _ in range(n_reps):
                for i in range(0, n_data, self.config["batchsize"] // 2):
                    # Halfbatch from core dataset
                    x_c = X[i:i + self.config["batchsize"] // 2]
                    y_c = Y[i:i + self.config["batchsize"] // 2]

                    # Halfbatch from ud dataset
                    x_ud = X_ud[rnd_indeces[i:i + self.config["batchsize"] // 2]]
                    y_ud = Y_ud[rnd_indeces[i:i + self.config["batchsize"] // 2]]

                    # Combine datasets
                    x = T.concat((x_c, x_ud), dim=0)
                    y = T.concat((y_c, y_ud), dim=0)

                    y_ = tep(x)
                    policy_loss = lossfun(y_, y)

                    total_loss = policy_loss
                    total_loss.backward()

                    policy_optim.step()
                    policy_optim.zero_grad()

            print("Epoch: {}, Iter {}, policy_loss: {}".format(ep, i, policy_loss.data))

            # Update ud dataset
            for t_idx in range(10):
                x_ud_traj = T.clone(X[t_idx]).detach()
                x_ud_traj.requires_grad = True
                X_ud[t_idx] = self.perform_grad_update_full_traj(x_ud_traj, tep, use_hessian=False)

                # Annotate the new X_ud
                obs = self.env.reset()
                Y_ud[t_idx] = self.evaluate_rollout(obs, X_ud[t_idx].detach().numpy())

                if t_idx % 1 == 0:
                    print(f"Dataset updating: {t_idx}")

        print("Done training, saving model")
        if not os.path.exists("agents"):
            os.makedirs("agents")
        T.save(tep.state_dict(), "agents/full_traj_tep_1step.p")

    def train_tep_1step_grad_aggregated(self):
        env, venv, sb_policy = self.load_model_and_env()

        # Load pretrained tep
        tep = TEPMLP(obs_dim=50, act_dim=1)
        tep.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

        # Core dataset
        N_traj = 2000
        X = np.load(self.x_file_path, allow_pickle=True)[:N_traj]
        Y = np.load(self.y_file_path)[:N_traj]

        # Change to successive angle representation
        X = T.tensor(self.get_successive_angle_representation(X), dtype=T.float32) # 1000, 50
        Y = T.tensor(Y, dtype=T.float32).unsqueeze(1) # 1000, 1

        # Make dataset into list of tensors
        X_list = [x for x in X]
        Y_list = [y for y in Y]

        # Prepare policy and training
        tep_optim = T.optim.Adam(params=tep.parameters(),
                                    lr=self.config['policy_lr'],
                                    weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()

        # Epoch consists of training tep and performing grad update on subset of original dataset
        for ep in range(self.config["n_epochs"]):
            n_data = len(X_list)

            # ===== TRAIN TEP ======
            for i in range(0, self.config["tep_iters"]):
                # Get random minibatch
                rnd_indeces = np.random.choice(np.arange(n_data), self.config["batchsize"], replace=False)
                x = T.stack([X_list[ri] for ri in rnd_indeces])
                y = T.stack([Y_list[ri] for ri in rnd_indeces])

                # Train tep on new dataset
                y_ = tep(x)
                tep_loss = lossfun(y_, y)

                total_loss = tep_loss
                total_loss.backward()

                tep_optim.step()
                tep_optim.zero_grad()

                if i % 10 == 0:
                    print("Epoch: {}, Iter {}, policy_loss: {}".format(ep, i, tep_loss.data))

            print("Epoch: {}, finished training tep".format(ep))

            # ===== UPDATE TRAJ ======
            # Update random trajectories from initial dataset and add to dataset
            rnd_indeces = np.random.choice(np.arange(N_traj), self.config["n_traj_update"], replace=False)
            x_list = T.stack([X_list[ri] for ri in rnd_indeces])
            for x_traj in x_list:
                x_ud_traj = T.clone(x_traj).detach()
                x_ud_traj.requires_grad = True
                x_ud_traj = self.optimize_traj(x_ud_traj, tep)
                X_list.append(x_ud_traj)

                #self.plot_trajs2(x_traj[:50], x_ud_traj)

                # Annotate the new X_ud
                obs = env.reset()
                rew = self.evaluate_rollout(obs, env, venv, sb_policy, self.sar_to_xy(x_ud_traj).detach().numpy())
                Y_list.append(T.tensor([rew]))

            print("Epoch: {}, finished updating dataset".format(ep))

        # Finished training, saving
        print("Done training, saving model")
        if not os.path.exists("agents"):
            os.makedirs("agents")
        T.save(tep.state_dict(), "agents/full_traj_tep_1step.p")

    def test_tep(self, env, venv, sb_policy):
        tep = TEPMLP(obs_dim=50, act_dim=1)
        tep.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

        N_eval = 100
        render = True
        for i in range(N_eval):
            obs = env.reset()
            time_taken = self.evaluate_rollout(obs, env, venv, sb_policy=sb_policy, render=render, deterministic=self.config["deterministic_eval"])

            # Make tep prediction
            traj = env.engine.wp_list
            traj_sar, distances = self.xy_to_sar(traj[:50])
            traj_T_sar = T.tensor(traj_sar, dtype=T.float32)
            tep_pred = tep(traj_T_sar)

            print(f"Time taken: {time_taken}, time taken predicted: {tep_pred}")

    def test_tep_full(self):
        tep_def = TEPMLP(obs_dim=50, act_dim=1)
        tep_def.load_state_dict(T.load("agents/full_traj_tep.p"), strict=False)

        tep_agg = TEPMLP(obs_dim=50, act_dim=1)
        tep_agg.load_state_dict(T.load("agents/full_traj_tep_1step.p"), strict=False)

        # Errors on def traj
        tep_err_def = 0
        tep_agg_err_def = 0

        # Errors on 1 step traj
        tep_err_1step = 0
        tep_agg_err_1step = 0

        # Errors on 1 step agg traj
        tep_err_1step_agg = 0
        tep_agg_err_1step_agg = 0

        avg_time_taken = 0
        avg_time_1step = 0
        avg_time_agg_1step = 0

        failure_def = 0
        failure_1step = 0
        failure_agg_1step = 0

        render = False
        plot = self.config["plot_trajs"]
        for i in range(self.config["n_eval"]):
            obs = self.env.reset()
            self.env.engine.wp_list = self.env.engine.wp_list[:50]
            traj = deepcopy(self.env.engine.wp_list)
            traj_sar, distances = self.xy_to_sar(traj[:50])
            traj_T_sar = T.tensor(traj_sar, dtype=T.float32, requires_grad=True)

            # Make tep prediction on default traj
            tep_def_pred = tep_def(traj_T_sar)[0].data
            tep_agg_pred = tep_agg(traj_T_sar)[0].data

            # GT time taken ============================
            time_taken = self.evaluate_rollout(obs, self.env, self.venv, self.sb_model, traj=traj, render=render,
                                               deterministic=self.config["deterministic_eval"])
            # ==========================================

            # DEF TEP ==================================
            # Make trajectory update
            traj_T_sar_ud = self.optimize_traj(traj_T_sar, tep_def)
            traj_T_ud = self.sar_to_xy(traj_T_sar_ud, distances)
            self.env.reset()
            self.env.engine.wp_list = list(traj_T_ud.detach().numpy())

            # Rollout on corrected traj
            time_taken_1step = self.evaluate_rollout(obs, self.env, self.venv, self.sb_model, traj=traj_T_ud.detach().numpy(), render=render, deterministic=self.config["deterministic_eval"])

            # Make tep prediction on updated traj
            tep_def_pred_1step = tep_def(traj_T_sar_ud)[0].data
            tep_agg_pred_1step = tep_agg(traj_T_sar_ud)[0].data
            # ==========================================

            # AGG TEP ==================================
            # Make trajectory update
            traj_T_sar_ud_agg = self.optimize_traj(traj_T_sar, tep_agg)
            traj_T_ud_agg = self.sar_to_xy(traj_T_sar_ud_agg, distances)
            self.env.reset()
            self.env.engine.wp_list = list(traj_T_ud_agg.detach().numpy())

            # Rollout on corrected traj
            time_taken_agg_1step = self.evaluate_rollout(obs, self.env, self.venv, self.sb_model, traj=traj_T_ud_agg.detach().numpy(),
                                                     render=render, deterministic=self.config["deterministic_eval"])

            # Make tep prediction on updated traj
            tep_def_pred_agg_1step = tep_def(traj_T_sar_ud)[0].data
            tep_agg_pred_agg_1step = tep_agg(traj_T_sar_ud)[0].data
            # ==========================================

            print(f"Time taken: {time_taken}, time predicted using def tep: {tep_def_pred}, time predicted using agg tep: {tep_agg_pred}")
            print(f"Time taken after n step update using def tep: {time_taken_1step}, time predicted after n step update using def tep: {tep_def_pred_1step}, time predicted after n step update using agg tep: {tep_agg_pred_1step}")
            print(f"Time taken after n step update using agg tep: {time_taken_agg_1step}, time predicted after n step agg update using def tep: {tep_def_pred_agg_1step}, time predicted after n step update using agg tep: {tep_agg_pred_agg_1step}")
            print("------------------------------------------------------------------------------------------------------------")

            # Plot before and after trajectories
            if plot:
                self.plot_trajs3(traj[:50], traj_T_ud.detach().numpy(), traj_T_ud_agg.detach().numpy())
                time.sleep(2.8)

            # Errors on def traj
            tep_err_def += np.abs(time_taken - tep_def_pred)
            tep_agg_err_def += np.abs(time_taken - tep_agg_pred)

            # Errors on 1 step traj
            tep_err_1step += np.abs(time_taken_1step - tep_def_pred_1step)
            tep_agg_err_1step += np.abs(time_taken_1step - tep_agg_pred_1step)

            # Errors on 1 step agg traj
            tep_err_1step_agg += np.abs(time_taken_agg_1step - tep_def_pred_agg_1step)
            tep_agg_err_1step_agg += np.abs(time_taken_agg_1step - tep_agg_pred_agg_1step)

            # Avg time taken
            avg_time_taken += time_taken
            avg_time_1step += time_taken_1step
            avg_time_agg_1step += time_taken_agg_1step

            # failure rates
            failure_def += (time_taken > 4.7)
            failure_1step += (time_taken_1step > 4.7)
            failure_agg_1step += (time_taken_agg_1step > 4.7)

        # Errors on def traj
        tep_err_def /= self.config["n_eval"]
        tep_agg_err_def /= self.config["n_eval"]

        # Errors on 1 step traj
        tep_err_1step /= self.config["n_eval"]
        tep_agg_err_1step /= self.config["n_eval"]

        # Errors on 1 step agg traj
        tep_err_1step_agg /= self.config["n_eval"]
        tep_agg_err_1step_agg /= self.config["n_eval"]

        avg_time_taken /= self.config["n_eval"]
        avg_time_1step /= self.config["n_eval"]
        avg_time_agg_1step /= self.config["n_eval"]

        # Print out results
        table = [['Avg time taken', avg_time_taken, avg_time_1step, avg_time_agg_1step],
                 [f'Failures (out of {self.config["n_eval"]})', failure_def, failure_1step, failure_agg_1step],
                 ['Default TEP', tep_err_def, tep_err_1step, tep_err_1step_agg],
                 ['Aggreg TEP', tep_agg_err_def, tep_agg_err_1step, tep_agg_err_1step_agg]]
        print(tabulate(table, headers=['',  'Def traj', '1 step traj', '1 step agg traj']))

    def plot_trajs(self, traj_xy, traj_T_sar_ud):
        if not hasattr(self, 'ax'):
            self.figure, self.ax = plt.subplots(figsize=(14, 6))
        #traj_T = self.sar_to_xy(traj_T_sar).detach().numpy()
        traj_T = traj_xy
        traj_T_ud = self.sar_to_xy(traj_T_sar_ud).detach().numpy()

        line1, = self.ax.plot(list(zip(*traj_T))[0], list(zip(*traj_T))[1], marker="o", color="r", markersize=3)
        line2, = self.ax.plot(list(zip(*traj_T_ud))[0], list(zip(*traj_T_ud))[1], marker="o", color="b", markersize=3)
        #self.ax.scatter([4, 6, 17], [.5, .5, 0], s=200, c=['r', 'r', 'w'])
        plt.grid()
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])

        # ax.quiver(traj_reshaped[:, 0],
        #           traj_reshaped[:, 1],
        #           grad_reshaped[:, 0],
        #           grad_reshaped[:, 1],
        #           width=0.001,
        #           color=[1, 0, 0])
        #
        # ax.quiver(traj_reshaped[:, 0],
        #           traj_reshaped[:, 1],
        #           scaled_grad_reshaped[:, 0],
        #           scaled_grad_reshaped[:, 1],
        #           width=0.001,
        #           color=[0, 0, 1])

        # x, y = list(zip(*[t.detach().numpy() for t in traj_T.reshape((len(traj_T) // 2, 2))]))
        # line1.set_xdata(x)
        # line1.set_ydata(y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        self.ax.clear()

    def plot_trajs2(self, traj_xy, traj_T_sar_ud):
        if not hasattr(self, 'ax'):
            self.figure, self.ax = plt.subplots(figsize=(14, 6))
        #traj_T = self.sar_to_xy(traj_T_sar).detach().numpy()
        traj_T = self.sar_to_xy(traj_xy).detach().numpy()
        traj_T_ud = self.sar_to_xy(traj_T_sar_ud).detach().numpy()

        line1, = self.ax.plot(list(zip(*traj_T))[0], list(zip(*traj_T))[1], marker="o", color="r", markersize=3)
        line2, = self.ax.plot(list(zip(*traj_T_ud))[0], list(zip(*traj_T_ud))[1], marker="o", color="b", markersize=3)
        #self.ax.scatter([4, 6, 17], [.5, .5, 0], s=200, c=['r', 'r', 'w'])
        plt.grid()
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])

        # ax.quiver(traj_reshaped[:, 0],
        #           traj_reshaped[:, 1],
        #           grad_reshaped[:, 0],
        #           grad_reshaped[:, 1],
        #           width=0.001,
        #           color=[1, 0, 0])
        #
        # ax.quiver(traj_reshaped[:, 0],
        #           traj_reshaped[:, 1],
        #           scaled_grad_reshaped[:, 0],
        #           scaled_grad_reshaped[:, 1],
        #           width=0.001,
        #           color=[0, 0, 1])

        # x, y = list(zip(*[t.detach().numpy() for t in traj_T.reshape((len(traj_T) // 2, 2))]))
        # line1.set_xdata(x)
        # line1.set_ydata(y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        self.ax.clear()

    def plot_trajs3(self, traj_def, traj_1step, traj_agg_1step):
        if not hasattr(self, 'ax'):
            self.figure, self.ax = plt.subplots(figsize=(14, 6))

        line1, = self.ax.plot(list(zip(*traj_def))[0], list(zip(*traj_def))[1], marker="o", color="r", label='def', markersize=3)
        line2, = self.ax.plot(list(zip(*traj_1step))[0], list(zip(*traj_1step))[1], marker="o", color="g", label='1st', markersize=3)
        line3, = self.ax.plot(list(zip(*traj_agg_1step))[0], list(zip(*traj_agg_1step))[1], marker="o", color="b", label='1st_agg', markersize=3)
        self.ax.legend()

        plt.grid()
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])

        self.figure.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        self.ax.clear()

    def optimize_traj(self, traj, tep):
        mse_loss = torch.nn.MSELoss()
        traj_xy = self.sar_to_xy(traj.detach())
        traj_opt = T.clone(traj).detach()
        traj_opt.requires_grad = True

        #optimizer = T.optim.SGD(params=[traj_opt], lr=0.03, momentum=.0)
        optimizer = T.optim.Adam(params=[traj_opt], lr=self.config["traj_opt_rate"])
        #optimizer = T.optim.LBFGS(params=[traj_opt], lr=0.03)

        for i in range(self.config["n_step_opt"]):
            # etp loss
            tep_loss = tep(traj_opt)

            # Last point loss
            traj_opt_xy = self.sar_to_xy(traj_opt)
            last_pt_loss = mse_loss(traj_opt_xy[-1], traj_xy[-1])
            loss = tep_loss + last_pt_loss * 10
            loss.backward()

            # For LBFGS
            def closure():
                optimizer.zero_grad()
                loss = tep(traj_opt)
                loss.backward()
                return loss

            optimizer.step()
            #optimizer.step(closure) # For LBFGS
            optimizer.zero_grad()

        return traj_opt

    def optimize_traj_with_barriers(self, traj, tep, env):
        # For barrier loss
        def flattened_mse(p1, p2):
            a = T.tensor(-1e2)
            c = T.tensor(.2)
            x = p1 - p2
            loss = (T.abs(a - 2.) / (a)) * (T.pow((T.square(x / c) / T.abs(a - 2)) + 1, 0.5 * a) - 1.)
            return loss

        # For LBFGS
        def closure():
            optimizer.zero_grad()
            loss = tep(traj_opt)
            loss.backward()
            return loss

        mse_loss = torch.nn.MSELoss()
        traj_xy = self.sar_to_xy(traj.detach())
        traj_opt = T.clone(traj).detach()
        traj_opt.requires_grad = True

        #optimizer = T.optim.SGD(params=[traj_opt], lr=0.03, momentum=.0)
        optimizer = T.optim.Adam(params=[traj_opt], lr=self.config["traj_opt_rate"])
        #optimizer = T.optim.LBFGS(params=[traj_opt], lr=0.03)

        # Get barriers edge points (4 per barrier)
        edgepoints = env.maize.get_barrier_edgepoints()

        for i in range(self.config["n_step_opt"]):
            # etp loss
            tep_loss = tep(traj_opt)

            # Last point loss
            traj_opt_xy = self.sar_to_xy(traj_opt)
            last_pt_loss = mse_loss(traj_opt_xy[-1], traj_xy[-1]) * self.config["traj_last_pt_coeff"]

            # Barrier losses
            barrier_loss_list = []
            for xy_idx, xy in enumerate(traj_opt_xy):
                # Find closes edge point
                closest_edgept, dist = self.find_closest_edgepoint(xy.detach(), edgepoints)

                # If edge point close enough then apply loss (doesn't apply to inital n pts)
                if dist < 0.3 and xy_idx > 5:
                    barrier_loss_list.append(-flattened_mse(xy, T.tensor(closest_edgept)) * self.config["traj_barrier_coeff"])

            barrier_loss_sum = 0
            if len(barrier_loss_list) > 0:
                barrier_loss_sum = T.stack(barrier_loss_list).sum()

            loss = tep_loss + last_pt_loss + barrier_loss_sum
            loss.backward()

            optimizer.step()
            #optimizer.step(closure) # For LBFGS
            optimizer.zero_grad()

        return traj_opt

    def find_closest_edgepoint(self, pt, edgepoints):
        distances = []
        for ep in edgepoints:
            distances.append(dist_between_wps(pt, ep))
        closest_ep = edgepoints[np.argmin(distances)]
        closest_dist = np.min(distances)
        return closest_ep, closest_dist

    def optimize_env_traj(self, env, tep):
        traj = T.tensor(env.engine.wp_list)
        pred_before = tep(traj)
        traj_opt = self.optimize_traj(traj, tep)
        pred_after = tep(traj_opt)

        raise NotImplementedError
        # TODO: THIS IS MOST LIKELY WRONG
        env.engine.reset_trajectory(list(traj_opt.detach().numpy()))
        return pred_after.data - pred_before

    def xy_to_sar(self, X):
        X = np.concatenate((np.zeros(2)[np.newaxis, :], np.array(X)))
        X_new = np.arctan2(X[1:, 1] - X[:-1, 1], X[1:, 0] - X[:-1, 0])

        distances = np.sqrt(np.square(X[1:] - X[:-1]).sum(axis=1))

        return X_new, distances

    def sar_to_xy(self, X, distances=None):
        if distances is None:
            distances = T.ones(len(X), dtype=T.float32) * 0.175
        else:
            distances = T.tensor(distances, dtype=T.float32)
        #wp_dist = 0.17
        pd_x = T.cumsum(T.cos(X) * distances, dim=0).unsqueeze(1)
        pd_y = T.cumsum(T.sin(X) * distances, dim=0).unsqueeze(1)
        traj_T = T.concat((pd_x, pd_y), dim=1)
        return traj_T

if __name__ == "__main__":
    tm = TrajTepOptimizer(policy_ID="TRN")
    env, venv, sb_model = tm.load_model_and_env()
    tm.env = env
    tm.venv = venv
    tm.sb_model = sb_model
    #tm.make_dataset(render=False)
    #tm.train_tep()
    #tm.train_tep_1step_grad_aggregated()
    #tm.test_tep(env, venv, sb_model)
    tm.test_tep_full()



