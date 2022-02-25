import torch.nn as nn
import torch.nn.functional as F
import torch as T
from torch.nn.utils import weight_norm


class SLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(SLP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.act_dim, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        return out

class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(MLP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc2 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc3 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

        T.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

    def forward(self, x):
        fc1 = T.relu(self.fc1(x))
        fc2 = T.relu(self.fc2(fc1))
        out = self.fc3(fc2)
        return out

class LTE(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(LTE, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc2 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc3 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

    def forward(self, x):
        fc1 = T.tanh(self.fc1(x))
        fc2 = T.tanh(self.fc2(fc1))
        out = self.fc3(fc2)
        return out

    def predict_next_vel(self, o):
        o_T = T.tensor(o).unsqueeze(0)
        y_T = self.forward(o_T)
        return y_T.detach().numpy()[0]

class RNN(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=64):
        super(RNN, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.rnn_1 = T.nn.LSTM(self.obs_dim, self.hid_dim, self.hid_dim)
        self.fc3 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

    def forward(self, x):
        fc1 = T.tanh(self.fc1(x))
        rnn_1 = self.rnn_1(fc1, None)
        fc2 = self.fc2(rnn_1)
        return fc2

class TEPMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(TEPMLP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc2 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc3 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

        T.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

        self.nonlin = F.relu

    def forward(self, x):
        fc1 = self.nonlin(self.fc1(x))
        fc2 = self.nonlin(self.fc2(fc1) + fc1)
        out = self.fc3(fc2)
        return out

class TEPRNN(nn.Module):
    def __init__(self, n_waypts, hid_dim=32, hid_dim_2=6):
        super(TEPRNN, self).__init__()
        self.n_waypts = n_waypts
        self.hid_dim = hid_dim
        self.hid_dim_2 = hid_dim_2

        self.fc1 = T.nn.Linear(2, 4, bias=True)
        self.rnn = T.nn.LSTM(input_size=4, hidden_size=self.hid_dim, num_layers=2, bias=True, batch_first=True, bidirectional=False)
        self.fc2 = T.nn.Linear(self.hid_dim, self.hid_dim_2, bias=True)
        self.fc3 = T.nn.Linear(self.n_waypts * self.hid_dim_2, 1, bias=True)

        self.nonlin = F.relu

    def forward(self, x):
        x_reshaped = T.reshape(x, (len(x), self.n_waypts, 2))
        fc1 = self.nonlin(self.fc1(x_reshaped))
        rnn1 = self.rnn(fc1)
        fc2 = self.nonlin(self.fc2(rnn1))
        rnn1_reshaped = T.reshape(fc2, (len(x), self.n_waypts * self.hid_dim_2))
        out = self.fc3(rnn1_reshaped)
        return out

class TEPTX(nn.Module):
    def __init__(self, n_waypts, embed_dim, num_heads, kdim):
        super(TEPTX, self).__init__()
        self.n_waypts = n_waypts

        self.fc1 = T.nn.Linear(2, embed_dim, bias=True)
        self.fc_key = T.nn.Linear(embed_dim, kdim)
        self.fc_value = T.nn.Linear(embed_dim, kdim)

        self.tx1 = T.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=32)
        self.tx2 = T.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=32)

        self.fc_final = T.nn.Linear(self.n_waypts * kdim, 1)

        T.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

    def forward(self, x):
        fc1 = T.relu(self.fc1(x))
        fc2 = T.relu(self.fc2(fc1))
        out = self.fc3(fc2)
        return out
