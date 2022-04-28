import pickle
import os
from src.policies import *
import numpy as np
import torch as T
from collections import Counter

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

        # TODO: AFTER RE-RECORDING THE DATASET REMOVE THE X[0] PART (IT WILL ONLY HAVE 2 DIMS)
        traj_len = 300
        x_traj_list = []
        y_traj_list = []
        for x, y in zip(x_data_list, y_data_list):
            for traj_idx in range(0, len(x[0]) - traj_len, traj_len):
                x_traj_list.append(x[0, traj_idx:traj_idx + traj_len])
                y_traj_list.append(y[0, traj_idx:traj_idx + traj_len])

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

    def train(self, dataset, policy_name="buggy_lte", pretrained_model_path=None):
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
        T.save(self.policy.state_dict(), f"agents/{policy_name}.p")

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
        discrim = MLP(3 + 3, 2)
        optim = T.optim.Adam(params=discrim.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.CrossEntropyLoss()

        n_iters = 1000
        for i in range(n_iters):
            # Get minibatch from both datasets
            x_mujoco = self.mujoco_dataset

            # TODO: Cont here

            # Forward pass

            # Loss

            # Update
            pass


if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_model.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    mujoco_dataset = ModelDataset(use_real_data=False)
    real_dataset = ModelDataset(use_real_data=True)
    model_trainer = ModelTrainer(config, mujoco_dataset, real_dataset)

    # Train
    if config["train"]:
        pretrained_model_path = f"agents/buggy_lte.p"

        model_trainer.train_data_discriminator()
        #model_trainer.train_linmod()
        #model_trainer.train_lin()
        #model_trainer.train_linmod_hybrid()
        #model_trainer.train(mujoco_dataset, "buggy_real_lte", pretrained_model_path=None)

