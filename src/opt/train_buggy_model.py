import pickle
import os
from src.policies import *
import numpy as np
import torch as T
from collections import Counter

class ModelDataset:
    def __init__(self):
        self.X_trn, self.Y_trn, self.X_val, self.Y_val = self.load_mujoco_dataset()
        self.W_lhs = self.get_lhs_weights()

    def get_lhs_weights(self):
        # Check LHS here
        k = 7
        D = 5
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
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/dataset/")
        for i in range(100):
            fp_X = os.path.join(dataset_dir, "X_{}.pkl".format(i))
            fp_Y = os.path.join(dataset_dir, "Y_{}.pkl".format(i))
            if os.path.exists(fp_X):
                # self.data_dict_list.extend(pickle.load(open(file_path, "rb"), encoding='latin1'))
                x_data_list.append(pickle.load(open(fp_X, "rb")))
                y_data_list.append(pickle.load(open(fp_Y, "rb")))

        # Make tensor out of loaded list
        X_raw = np.concatenate(x_data_list)
        Y = np.concatenate(y_data_list)

        # Turn throttle and turn into real estimated values
        X = np.copy(X_raw)
        throttle_queue = [0] * 20
        turn_queue = [0] * 15
        for i in range(len(X_raw)):
            throttle_queue.append(X_raw[i, 0])
            turn_queue.append(X_raw[i, 1])
            del throttle_queue[0]
            del turn_queue[0]
            X[i, 0] = np.mean(throttle_queue)
            X[i, 1] = np.mean(turn_queue)

        # Condition the data
        X[X[:, 0] < 0.05, 0] = 0
        X[np.abs(X[:, 1]) < 0.01, 1] = 0
        X[np.abs(X[:, 2]) < 0.03, 2] = 0
        X[np.abs(X[:, 3]) < 0.03, 3] = 0
        X[np.abs(X[:, 4]) < 0.01, 4] = 0

        n_data_points = len(X)
        assert n_data_points > 100
        print("Loaded dataset with {} points".format(n_data_points))
        return X, Y

    def load_mujoco_dataset(self):
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/buggy_mujoco_dataset/")
        X = np.load(os.path.join(dataset_dir, "X.npy"))
        Y = np.load(os.path.join(dataset_dir, "Y.npy"))

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
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def train(self):
        self.policy = MLP(obs_dim=7, act_dim=5, hid_dim=256)
        optim = T.optim.Adam(params=self.policy.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()
        for i in range(self.config['iters']):
            X, Y = self.dataset.get_random_batch(self.config['batchsize'])
            Y_ = self.policy(X)
            loss = lossfun(Y_, Y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                X_val, Y_val = self.dataset.get_val_dataset()
                Y_val_ = self.policy(X_val)
                loss_val = lossfun(Y_val_, Y_val)
                print("Iter {}/{}, loss: {}, loss_val: {}".format(i, self.config['iters'], loss.data, loss_val.data))
        print("Done training, saving model")
        T.save(self.policy.state_dict(), "agents/buggy_lte.p")

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

if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_model.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = ModelDataset()
    model_trainer = ModelTrainer(config, dataset)

    # Train
    if config["train"]:
        model_trainer.train_linmod()
        #model_trainer.train_lin()
        #model_trainer.train_linmod_hybrid()
        #model_trainer.train()

