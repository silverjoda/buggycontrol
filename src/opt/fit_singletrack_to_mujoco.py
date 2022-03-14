import pickle
import os
from src.policies import *
import numpy as np
import torch as T

from src.ctrl.model import make_singletrack_model, make_bicycle_model
from src.ctrl.simulator import make_simulator

import cma

class ModelDataset:
    def __init__(self):
        # Run buggy_env_dataset_gatherer first
        self.X_trn, self.Y_trn, self.X_val, self.Y_val = self.load_mujoco_dataset()

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

    def f_wrapper_bicycle(self):
        def f(w):
            # Generate new model
            model = make_bicycle_model(w)
            simulator = make_simulator(model)

            # Get batch of data
            x, y = self.dataset.get_random_batch(batchsize=self.config["batchsize"], tensor=False)

            def extract_sim_state(state):
                vx, vy, vangz, _, _ = state[0:5]
                return np.array([0,0,0,vx,vy,vangz]).reshape(-1, 1)

            total_loss = 0
            ctr = 0.
            for i in range(len(x)):
                simulator.reset_history()

                # Slip angle, velocity, yaw rate, x, y, phi
                sim_state = extract_sim_state(x[i])

                # If velocity too small, discard
                if abs(sim_state[3]) < 0.05 or abs(sim_state[5]) < 0.01:
                    continue

                sim_state_next = extract_sim_state(y[i])
                sim_act = np.array([x[i][5] * 0.38, (x[i][6] + 1.0001) * 0.499]).reshape((-1, 1))
                simulator.x0 = sim_state
                sim_state_next_pred = simulator.make_step(sim_act)

                loss = np.mean(np.square(sim_state_next_pred[0:3] - sim_state_next[0:3]))
                total_loss += loss
                ctr += 1

            return total_loss / ctr

        return f

    def f_wrapper_singletrack(self):
        def f(w):
            # Generate new model
            model = make_singletrack_model(w)
            simulator = make_simulator(model)

            # Get batch of data
            x, y = self.dataset.get_random_batch(batchsize=self.config["batchsize"], tensor=False)

            def extract_sim_state(state):
                vx, vy, vangz, _, _ = state[0:5]
                b = np.arctan2(vy, vx)
                v = np.sqrt(np.square(np.array([vx, vy]).sum()))
                r = vangz
                return np.array([b,v,r,0,0,0]).reshape(-1, 1)

            total_loss = 0
            for i in range(len(x)):
                simulator.reset_history()

                # Slip angle, velocity, yaw rate, x, y, phi
                sim_state = extract_sim_state(x[i])
                sim_state_next = extract_sim_state(y[i])
                sim_act = np.array([x[i][5] * 0.38, (x[i][6] + 1.001) * 0.5]).reshape((-1, 1))
                simulator.x0 = sim_state
                sim_state_next_pred = simulator.make_step(sim_act)

                loss = np.mean(np.square(sim_state_next_pred[0:3] - sim_state_next[0:3]))
                total_loss += loss

            return total_loss

        return f

    def train_singletrack(self):
        init_params = [2, 2, 0.14, 0.16, 0.04, 1, 6.9, 1.8, 0.1, 1, 15, 1.7, -0.5]
        es = cma.CMAEvolutionStrategy(init_params, 0.5)
        f = self.f_wrapper_singletrack()

        it = 0
        try:
            while not es.stop():
                it += 1
                if it > config["iters"]:
                    break
                X = es.ask()
                es.tell(X, [f(x) for x in X])
                es.disp()

        except KeyboardInterrupt:
            print("User interrupted process.")

        return es.result.fbest

    def train_bicycle(self):
        init_params = [0.164, 0.16, 0.53, 0.28, 4.0, 0.12, 29, 26, 0.08, 0.16, 42, 161, 0.6, 90.1, 1.8, -0.25]
        es = cma.CMAEvolutionStrategy(init_params, 0.5)
        f = self.f_wrapper_bicycle()

        it = 0
        try:
            while not es.stop():
                it += 1
                if it > config["iters"]:
                    break
                X = es.ask()
                es.tell(X, [f(x) for x in X])
                es.disp()

        except KeyboardInterrupt:
            print("User interrupted process.")

        return es.result.fbest

if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/fit_singletrack_to_mujoco.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = ModelDataset()
    model_trainer = ModelTrainer(config, dataset)

    # Train
    if config["train"]:
        model_trainer.train_bicycle()

