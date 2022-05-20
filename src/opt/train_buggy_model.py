import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as T

from src.policies import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""
T.set_num_threads(2)

class SourceDiscriminator:
    def __init__(self):
        super().__init__()

class NeuralDiscriminator(SourceDiscriminator):
    def __init__(self):
        super().__init__()

class LSHDiscriminator(SourceDiscriminator):
    def __init__(self):
        super().__init__()

class ModelDataset:
    def __init__(self, use_real_data=False):
        if use_real_data:
            self.X_trn, self.Y_trn, self.X_val, self.Y_val = self.load_real_dataset()
        else:
            self.X_trn, self.Y_trn, self.X_val, self.Y_val = self.load_mujoco_dataset()
        self.W_lhs = self.get_lhs_weights()

    def get_lhs_weights(self):
        # Check LHS here
        k = 8
        D = 3
        A = np.random.randn(k, D)
        X = self.X_trn.reshape((self.X_trn.shape[0] * self.X_trn.shape[1], self.X_trn.shape[2]))[:, 2:]
        Z = np.sign((A @ X.T).T)
        Zt = tuple(map(tuple, Z))

        HS = {}
        for zt in Zt:
            if zt not in HS:
                HS[zt] = 1
            else:
                HS[zt] += 1

        W_lhs = []
        for zt in Zt:
            W_lhs.append(1./np.power(HS[zt], 1/3))
        return np.array(W_lhs)

    def load_real_dataset(self):
        x_data_list = []
        y_data_list = []
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/buggy_real_dataset/")
        for i in range(100):
            fp_X = os.path.join(dataset_dir, "X_{}.pkl".format(i))
            fp_Y = os.path.join(dataset_dir, "Y_{}.pkl".format(i))
            if os.path.exists(fp_X):
                # self.data_dict_list.extend(pickle.load(open(file_path, "rb"), encoding='latin1'))
                x_data_list.append(pickle.load(open(fp_X, "rb")))
                y_data_list.append(pickle.load(open(fp_Y, "rb")))

        traj_len = 400
        x_traj_list = []
        y_traj_list = []
        for x, y in zip(x_data_list, y_data_list):
            # TODO: REMOVE THE x[0], it's due to an extra dimension which should go away with new data
            for traj_idx in range(0, len(x[0]) - traj_len, traj_len):
                x_traj_list.append(x[0,traj_idx:traj_idx + traj_len])
                y_traj_list.append(y[0,traj_idx:traj_idx + traj_len])

        # Make tensor out of loaded list
        X = np.stack(x_traj_list)
        Y = np.stack(y_traj_list)

        X_sym = np.copy(X)
        Y_sym = np.copy(Y)

        # Correct sym datasets. # -, y_vel, ang_vel, turn, -
        X_sym[:, :, np.array([1, 2, 3])] *= -1
        Y_sym[:, :, np.array([1, 2])] *= -1

        X = np.concatenate((X, X_sym), axis=0)
        Y = np.concatenate((Y, Y_sym), axis=0)

        n_traj = len(X)
        assert n_traj > 100
        print("Loaded dataset with {} trajectories".format(n_traj))

        split_pt = int(n_traj * 0.9)
        X_trn = X[:split_pt]
        Y_trn = Y[:split_pt]
        X_val = X[split_pt:]
        Y_val = Y[split_pt:]

        return X_trn, Y_trn, X_val, Y_val

    def load_mujoco_dataset(self):
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/buggy_mujoco_dataset/")
        X = np.load(os.path.join(dataset_dir, "X.npy"), encoding='latin1', allow_pickle=True)
        Y = np.load(os.path.join(dataset_dir, "Y.npy"), encoding='latin1', allow_pickle=True)

        X_sym = np.copy(X)
        Y_sym = np.copy(Y)

        # Correct sym datasets. # vel[0], vel[1], ang_vel[2]
        X_sym[:,:, np.array([1, 2, 3])] *= -1
        Y_sym[:, :, np.array([1, 2])] *= -1

        X = np.concatenate((X, X_sym), axis=0)
        Y = np.concatenate((Y, Y_sym), axis=0)

        n_traj = len(X)
        assert n_traj > 10
        print("Loaded dataset with {} trajectories".format(n_traj))

        split_pt = int(n_traj * 0.9)
        X_trn = X[:split_pt]
        Y_trn = Y[:split_pt]
        X_val = X[split_pt:]
        Y_val = Y[split_pt:]

        return X_trn, Y_trn, X_val, Y_val

    def get_random_batch(self, batchsize, tensor=True):
        X_contig = self.X_trn.reshape((self.X_trn.shape[0] * self.X_trn.shape[1], self.X_trn.shape[2]))
        Y_contig = self.Y_trn.reshape((self.Y_trn.shape[0] * self.Y_trn.shape[1], self.Y_trn.shape[2]))

        rnd_indeces = np.random.choice(np.arange(len(X_contig)), batchsize, replace=False)
        x = X_contig[rnd_indeces]
        y = Y_contig[rnd_indeces]
        if tensor:
            x = T.tensor(x, dtype=T.float32)
            y = T.tensor(y, dtype=T.float32)
        return x, y

    def get_random_batch_weighted(self, batchsize, tensor=True):
        X_contig = self.X_trn.reshape((self.X_trn.shape[0] * self.X_trn.shape[1], self.X_trn.shape[2]))
        Y_contig = self.Y_trn.reshape((self.Y_trn.shape[0] * self.Y_trn.shape[1], self.Y_trn.shape[2]))

        rnd_indeces = np.random.choice(np.arange(len(X_contig)), batchsize, replace=False)
        x = X_contig[rnd_indeces]
        y = Y_contig[rnd_indeces]
        w = self.W_lhs[rnd_indeces]
        if tensor:
            x = T.tensor(x, dtype=T.float32)
            y = T.tensor(y, dtype=T.float32)
            w = T.unsqueeze(T.tensor(w, dtype=T.float32), 1)
        return x, y, w

    def get_val_dataset(self, tensor=True):
        X = self.X_val.reshape((self.X_val.shape[0] * self.X_val.shape[1], self.X_val.shape[2]))
        Y = self.Y_val.reshape((self.Y_val.shape[0] * self.Y_val.shape[1], self.Y_val.shape[2]))
        if tensor:
            X = T.tensor(X, dtype=T.float32)
            Y = T.tensor(Y, dtype=T.float32)
        return X, Y

class ModelTrainer:
    def __init__(self, config, mujoco_dataset, real_dataset):
        self.config = config
        self.mujoco_dataset = mujoco_dataset
        self.real_dataset = real_dataset

    def filter_mujoco_dataset(self, dataset, discriminator, threshold):
        pass

    def train(self, dataset, model_name="buggy_lte", pretrained_model_path=None):
        self.policy = LTE(obs_dim=5, act_dim=3, hid_dim=128)

        if pretrained_model_path is not None:
            self.policy.load_state_dict(T.load(pretrained_model_path), strict=False)

        optim = T.optim.Adam(params=self.policy.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()
        for i in range(self.config['iters']):
            X, Y = dataset.get_random_batch(self.config['batchsize'])
            #X, Y, W = self.dataset.get_random_batch_weighted(self.config['batchsize'])
            Y_ = self.policy(X)
            loss = lossfun(Y_, Y)
            #loss = T.mean((W * (Y_ - Y) ** 2))
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                X_val, Y_val = dataset.get_val_dataset()
                Y_val_ = self.policy(X_val)
                loss_val = lossfun(Y_val_, Y_val)
                print("Iter {}/{}, loss: {}, loss_val: {}".format(i, self.config['iters'], loss.data, loss_val.data))
        print("Done training, saving model")
        T.save(self.policy.state_dict(), f"agents/{model_name}.p")

    def train_fused(self, mujoco_dataset, real_dataset, policy_name="buggy_lte", pretrained_model_path=None):
        self.policy = LTE(obs_dim=5, act_dim=3, hid_dim=128)
        if pretrained_model_path is not None:
            self.policy.load_state_dict(T.load(pretrained_model_path), strict=False)

        data_discriminator = LTE(obs_dim=5, act_dim=2, hid_dim=128)
        data_discriminator.load_state_dict(T.load("agents/data_discriminator.p"), strict=False)

        # ALGO:
        # 1) Sample equal sized batches from mujoco and real datasets
        # 2) Use discriminator to decide which data points from mujoco dataset will get a mujoco label and which will get a real label (or maybe just throw them out)
        # 3) Apply supervised loss

        # Or maybe...
        # 1) Use discriminator to filter out mujoco dataset: Remove states which are too close to real states
        # 2) Train on joint dataset with labels from corresponding dataset

        # Filter out mujoco dataset
        self.filter_mujoco_dataset(mujoco_dataset, data_discriminator, threshold=0.5)

        optim = T.optim.Adam(params=self.policy.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()
        for i in range(self.config['iters']):
            X_mujo, Y_mujo = mujoco_dataset.get_random_batch(self.config['batchsize'] // 2)
            X_real, Y_real = real_dataset.get_random_batch(self.config['batchsize'] // 2)
            #X, Y, W = self.dataset.get_random_batch_weighted(self.config['batchsize'])

            X = np.concatenate((X_mujo, X_real))
            Y = np.concatenate((Y_mujo, Y_real))

            Y_ = self.policy(X)
            loss = lossfun(Y_, Y)
            #loss = T.mean((W * (Y_ - Y) ** 2))
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                X_val_mujo, Y_val_mujo = mujoco_dataset.get_val_dataset()
                X_val_real, Y_val_reals = real_dataset.get_val_dataset()
                X_val = np.concatenate((X_val_mujo, Y_val_mujo))
                Y_val = np.concatenate((X_val_real, Y_val_reals))
                Y_val_ = self.policy(X_val)
                loss_val = lossfun(Y_val_, Y_val)
                print("Iter {}/{}, loss: {}, loss_val: {}".format(i, self.config['iters'], loss.data, loss_val.data))
        print("Done training, saving model")
        T.save(self.policy.state_dict(), "agents/{policy_name}.p")

    def train_lin(self):
        policy = LIN(state_dim=3, act_dim=2)
        optim = T.optim.Adam(params=policy.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()
        for i in range(self.config['iters']):
            X, Y, W = self.dataset.get_random_batch_weighted(self.config['batchsize'])

            # Extract components from batch
            X_s = X[:, 2:5]
            A_s = X[:, 5:7]
            Y_s = Y[:, 2:5]

            next_state = policy(X_s, A_s)
            #loss = lossfun(next_state, Y_s)
            loss = T.mean((W * (next_state - Y_s) ** 2))
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                with T.no_grad():
                    X_val, Y_val = self.dataset.get_val_dataset()
                    X_val_s = X_val[:, 2:5]
                    A_val_s = X_val[:, 5:7]
                    Y_val_s = Y_val[:, 2:5]
                    next_state_val = policy(X_val_s, A_val_s)
                    next_state_loss_val = lossfun(next_state_val, Y_val_s)

                    total_val_loss_val = next_state_loss_val
                print("Iter {}/{}, loss: {}, loss_val: {}".format(i, self.config['iters'], loss.data, total_val_loss_val.data))
        print("Done training, saving model")
        T.save(policy.state_dict(), "agents/buggy_lin.p")

    def train_linmod(self):
        policy = LINMOD(state_dim=3, act_dim=2, state_enc_dim=12, act_enc_dim=4, hid_dim=32, extra_hidden=False)
        optim = T.optim.Adam(params=policy.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        mse_lossfun = T.nn.MSELoss()
        for i in range(self.config['iters']):
            X, Y, W = self.dataset.get_random_batch_weighted(self.config['batchsize'])
            lossfun = lambda x,y : T.mean((W * (x - y) ** 2))

            # Extract components from batch
            X_s = X[:, 2:5]
            A_s = X[:, 5:7]
            Y_s = Y[:, 2:5]

            next_state_dec, state_dec, act_dec = policy(X_s, A_s)
            next_state_loss = mse_lossfun(next_state_dec, Y_s)
            state_recon_loss = mse_lossfun(state_dec, X_s)
            act_recon_loss = mse_lossfun(act_dec, A_s)
            total_loss = next_state_loss + act_recon_loss + state_recon_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                with T.no_grad():
                    X_val, Y_val = self.dataset.get_val_dataset()
                    X_val_s = X_val[:, 2:5]
                    A_val_s = X_val[:, 5:7]
                    Y_val_s = Y_val[:, 2:5]
                    next_state_dec_val, state_dec_val, act_dec_val = policy(X_val_s, A_val_s)
                    next_state_loss_val = mse_lossfun(next_state_dec_val, Y_val_s)
                    state_recon_loss_val = mse_lossfun(state_dec_val, X_val_s)
                    act_recon_loss_val = mse_lossfun(act_dec_val, A_val_s)
                    total_val_loss_val = next_state_loss_val + act_recon_loss_val + state_recon_loss_val
                print("Iter {}/{}, loss: {}, loss_val: {}".format(i, self.config['iters'], total_loss.data, total_val_loss_val.data))
        print("Done training, saving model")
        T.save(policy.state_dict(), "agents/buggy_linmod_hybrid.p")

    def train_linmod_hybrid(self):
        policy = LINMOD_HYBRID(state_dim=3, act_dim=2, state_enc_dim=12, act_enc_dim=4, hid_dim=32,
                                              extra_hidden=False)
        optim = T.optim.Adam(params=policy.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()
        for i in range(self.config['iters']):
            X, Y = self.dataset.get_random_batch(self.config['batchsize'])

            # Extract components from batch
            X_s = X[:, 2:5]
            A_s = X[:, 5:7]
            Y_s = Y[:, 2:5]

            next_state, _ = policy(X_s, A_s)
            loss = lossfun(next_state, Y_s)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                with T.no_grad():
                    X_val, Y_val = self.dataset.get_val_dataset()
                    X_val_s = X_val[:, 2:5]
                    A_val_s = X_val[:, 5:7]
                    Y_val_s = Y_val[:, 2:5]
                    next_state_val, _ = policy(X_val_s, A_val_s)
                    next_state_loss_val = lossfun(next_state_val, Y_val_s)

                    total_val_loss_val = next_state_loss_val
                print("Iter {}/{}, loss: {}, loss_val: {}".format(i, self.config['iters'], loss.data, total_val_loss_val.data))
        print("Done training, saving model")
        T.save(policy.state_dict(), "agents/buggy_linmod_hybrid.p")

    def train_data_discriminator(self):
        discrim = MLP(5, 2, hid_dim=128)
        optim = T.optim.Adam(params=discrim.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        dataset_ratio = len(self.real_dataset.X_trn) / len(self.mujoco_dataset.X_trn)
        #lossfun = T.nn.CrossEntropyLoss(weight=T.tensor([dataset_ratio, 1 - dataset_ratio]))
        lossfun = T.nn.CrossEntropyLoss()

        self.plot_umap()

        n_iters = 1000
        for i in range(n_iters):
            # Get minibatch from both datasets
            x_mujoco, _ = self.mujoco_dataset.get_random_batch(self.config["batchsize"] // 2, tensor=True)
            x_real, _ = self.real_dataset.get_random_batch(self.config["batchsize"] // 2, tensor=True)

            # Fuse minibatch
            x_fused = T.concat((x_mujoco, x_real), dim=0)
            y_fused = T.zeros(self.config["batchsize"], dtype=T.long)
            y_fused[self.config["batchsize"] // 2 :] = 1

            # Forward pass
            y_ = discrim(x_fused)

            # Loss
            loss = lossfun(y_, y_fused)

            # Update
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            if i % 100 == 0:
                x_eval_mujoco, _ = self.mujoco_dataset.get_val_dataset(tensor=True)
                y_mujoco = discrim(x_eval_mujoco)
                eval_mujoco_loss = lossfun(y_mujoco, T.zeros(len(x_eval_mujoco), dtype=T.long))

                x_eval_real, _ = self.real_dataset.get_val_dataset(tensor=True)
                y_real = discrim(x_eval_real)
                eval_real_loss = lossfun(y_real, T.zeros(len(x_eval_real), dtype=T.long))

                print(f"Iter: {i}/{n_iters}, trn accuracy: {loss.data}, eval mujoco loss: {eval_mujoco_loss.data}, eval real loss: {eval_real_loss.data}")

        print("Done training, saving model")
        T.save(discrim.state_dict(), f"agents/data_discriminator.p")

        # Use the last eval values for evaluation
        plt.hist(T.softmax(y_mujoco, dim=1)[:, 0].detach().numpy())
        plt.hist(T.softmax(y_real, dim=1)[:, 1].detach().numpy())
        plt.show()

    def evaluate_trained_model(self, dataset, model_name="buggy_lte"):
        # Load trained model
        dynamics_model = MLP(5, 3, hid_dim=128)
        model_path = os.path.join(os.path.dirname(__file__), f"agents/{model_name}.p")
        dynamics_model.load_state_dict(T.load(model_path), strict=False)

        # Define evaluation trajectory set
        X_val, Y_val = T.tensor(dataset.X_val, dtype=T.float32), T.tensor(dataset.Y_val, dtype=T.float32)
        n_traj, traj_len, _ = X_val.shape

        # == For each trajectory in evaluation set, make rollouts of various length and compare using MSE ==

        # Make velocities rollout of model given actions
        pred_vel = []
        vel_cur = X_val[:, 0:1, :3]
        for i in range(traj_len):
            obs = T.concat((vel_cur, X_val[:, i:i+1, 3:5]), dim=2)
            vel_cur = dynamics_model(obs)
            pred_vel.append(vel_cur)

        pred_vel = T.concat(pred_vel, 1).detach().numpy()
        vel_val = X_val[:, :, :3].detach().numpy()

        # Turn predicted velocities and gt velocities into positional trajectories
        t_delta = 0.01
        gt_pos_val = np.zeros_like(vel_val, dtype=np.float32)
        pred_pos_val = np.zeros_like(vel_val, dtype=np.float32)

        for t in range(traj_len - 1):
            gt_pos_val[:, t + 1, :] = gt_pos_val[:, t, :] + vel_val[:, t, :] * t_delta
            pred_pos_val[:, t + 1, :] = pred_pos_val[:, t, :] + pred_vel[:, t, :] * t_delta

        # Calculate mse
        vel_mse_30 = []
        vel_mse_60 = []
        vel_mse_150 = []

        pos_mse_30 = []
        pos_mse_60 = []
        pos_mse_150 = []
        for t in range(traj_len - 1):
            vel_mse = np.mean(np.square(vel_val[:, t, :] - pred_vel[:, t, :]))
            pos_mse = np.mean(np.square(gt_pos_val[:, t, :] - pred_pos_val[:, t, :]))
            if t < 150:
                vel_mse_150.append(vel_mse)
                pos_mse_150.append(pos_mse)
                if t < 60:
                    vel_mse_60.append(vel_mse)
                    pos_mse_60.append(pos_mse)
                    if t < 30:
                        vel_mse_30.append(vel_mse)
                        pos_mse_30.append(pos_mse)

        # Print out mse statistics for various model rollout lengths
        vel_mse_30_mean, vel_mse_30_std = np.mean(vel_mse_30), np.std(vel_mse_30)
        vel_mse_60_mean, vel_mse_60_std = np.mean(vel_mse_60), np.std(vel_mse_60)
        vel_mse_150_mean, vel_mse_150_std = np.mean(vel_mse_150), np.std(vel_mse_150)

        pos_mse_30_mean, pos_mse_30_std = np.mean(pos_mse_30), np.std(pos_mse_30)
        pos_mse_60_mean, pos_mse_60_std = np.mean(pos_mse_60), np.std(pos_mse_60)
        pos_mse_160_mean, pos_mse_160_std = np.mean(pos_mse_150), np.std(pos_mse_150)

        print(f"Vel mse 30 mean: {vel_mse_30_mean}, std: {vel_mse_30_std}")
        print(f"Vel mse 60 mean: {vel_mse_60_mean}, std: {vel_mse_60_std}")
        print(f"Vel mse 150 mean: {vel_mse_150_mean}, std: {vel_mse_150_std}")

        print(f"Pos mse 30 mean: {pos_mse_30_mean}, std: {pos_mse_30_std}")
        print(f"Pos mse 60 mean: {pos_mse_60_mean}, std: {pos_mse_60_std}")
        print(f"Pos mse 150 mean: {pos_mse_160_mean}, std: {pos_mse_160_std}")

        # Plot several random positional trajectories and velocities
        N_plot = 5
        rnd_indeces = np.random.choice(np.arange(n_traj), N_plot, replace=False)

        fig, axs = plt.subplots(3, N_plot)
        for i in range(N_plot):
            axs[0, i].set(xlabel=f'x-pos, traj: {i}', ylabel='y-pos')
            axs[0, i].plot(gt_pos_val[rnd_indeces[i], :, 0], gt_pos_val[rnd_indeces[i], :, 1], 'tab:blue', label='Gt')
            axs[0, i].plot(pred_pos_val[rnd_indeces[i], :, 0], pred_pos_val[rnd_indeces[i], :, 1], 'tab:red', label='Pred')

            axs[1, i].set(xlabel=f'Time, traj: {i}', ylabel='x-vel')
            axs[1, i].plot(np.arange(traj_len), vel_val[rnd_indeces[i], :, 0], 'tab:blue', label='Gt')
            axs[1, i].plot(np.arange(traj_len), pred_vel[rnd_indeces[i], :, 0], 'tab:red', label='Pred')

            axs[2, i].set(xlabel=f'Time, traj: {i}', ylabel='y-vel')
            axs[2, i].plot(np.arange(traj_len), vel_val[rnd_indeces[i], :, 1], 'tab:blue', label='Gt')
            axs[2, i].plot(np.arange(traj_len), pred_vel[rnd_indeces[i], :, 1], 'tab:red', label='Pred')

        fig.tight_layout()
        plt.show()
        exit(0)

    def plot_umap(self):
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import umap
        reducer = umap.UMAP()

        data_mujoco = self.mujoco_dataset.X_trn.reshape((-1, 5))
        data_mujoco = data_mujoco[np.random.choice(np.arange(len(data_mujoco)), size=3000, replace=False)]
        data_real = self.real_dataset.X_trn.reshape((-1, 5))
        data_real = data_real[np.random.choice(np.arange(len(data_real)), size=3000, replace=False)]
        data_joint = np.concatenate((data_mujoco, data_real), axis=0)
        indeces = np.array(["mujoco"] * len(data_mujoco) + ["real"] * len(data_real))
        df = pd.DataFrame(data_joint, index=indeces, columns=['velx', 'vely', 'ang_vel', 'turn', 'throttle'])
        scaled_data = StandardScaler().fit_transform(df)

        embedding = reducer.fit_transform(scaled_data)
        print(embedding.shape)

        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[sns.color_palette()[x] for x in df.index.map({"mujoco": 0, "real": 1})])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the buggy dataset', fontsize=24)
        plt.show()
        exit()

if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_model.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #real_dataset = ModelDataset(use_real_data=False)
    mujoco_dataset = ModelDataset(use_real_data=False)
    model_trainer = ModelTrainer(config, mujoco_dataset, mujoco_dataset)
    #model_trainer.plot_umap()
    #exit()

    # Train
    if config["train"]:
        pretrained_model_path = f"agents/buggy_lte_mujoco.p"

        model_trainer.train(mujoco_dataset, "buggy_lte_mujoco", pretrained_model_path=None)
        #model_trainer.train_data_discriminator()
        #model_trainer.evaluate_trained_model(mujoco_dataset, model_name="buggy_lte_mujoco")