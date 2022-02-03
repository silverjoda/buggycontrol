import torch as T
import pickle
import os
from policies import *

class ModelDataset:
    def __init__(self):
        X, Y = self.load_dataset()

    def load_dataset(self):
        pass

class ModelTrainer:
    def __init__(self, config, dataset, policy):
        self.config = config
        self.dataset = dataset
        self.policy = policy

    def train(self):
        pass

    def evaluate(self):
        pass

if __name__=="__main__":
    import yaml

    with open(os.path.join(os.path.dirname(__file__), "configs/train_buggy_model.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = ModelDataset()
    policy = MLP(obs_dim=5, act_dim=3, hid_dim=128)
    model_trainer = ModelTrainer(config, dataset, policy)

    # Train and evaluate
    if config["train"]:
        model_trainer.train()

    # Evaluate visually
