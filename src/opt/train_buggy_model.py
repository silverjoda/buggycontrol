import pickle
import os
from src.policies import *
import numpy as np
import torch as T

class ModelDataset:
    def __init__(self):
        self.X, self.Y = self.load_dataset()

    def load_dataset(self):
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

    def get_random_batch(self, batchsize, tensor=True):
        rnd_indeces = np.random.choice(np.arange(len(self.X)), batchsize, replace=False)
        x = self.X[rnd_indeces]
        y = self.Y[rnd_indeces]
        if tensor:
            x = T.tensor(x, dtype=T.float32)
            y = T.tensor(y, dtype=T.float32)
        return x, y

class ModelTrainer:
    def __init__(self, config, dataset, policy):
        self.config = config
        self.dataset = dataset
        self.policy = policy

    def train(self):
        optim = T.optim.Adam(params=self.policy.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        lossfun = T.nn.MSELoss()
        for i in range(self.config['iters']):
            X, Y = self.dataset.get_random_batch(self.config['batchsize'])
            Y_ = self.policy(X)
            loss = lossfun(Y_, Y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 50 == 0:
                print("Iter {}/{}, loss: {}".format(i, self.config['iters'], loss.data))
        print("Done training, saving model")
        T.save(self.policy.state_dict(), "agents/buggy_lte.p")

if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_model.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = ModelDataset()
    policy = MLP(obs_dim=5, act_dim=3, hid_dim=256)
    model_trainer = ModelTrainer(config, dataset, policy)

    # Train
    if config["train"]:
        model_trainer.train()

