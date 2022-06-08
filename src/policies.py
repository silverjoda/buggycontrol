import torch.nn as nn
import torch.nn.functional as F
import torch as T
from torch.nn.utils import weight_norm
import numpy as np

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

    def predict_next(self, o):
        o_T = T.tensor(o)
        y_T = self.forward(o_T)
        return y_T.detach().numpy()

class LTE(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(LTE, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.non_linearity = T.relu # T.tanh

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
        fc1 = self.non_linearity(self.fc1(x))
        fc2 = self.non_linearity(self.fc2(fc1))
        out = self.fc3(fc2)
        return out

    def predict_next_vel(self, o):
        o_T = T.tensor(o)
        y_T = self.forward(o_T)
        return y_T.detach().numpy()

class LIN(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(LIN, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        # Linear system
        self.A = T.nn.Parameter(T.randn((self.state_dim, self.state_dim), requires_grad=True))
        self.B = T.nn.Parameter(T.randn((self.state_dim, self.act_dim), requires_grad=True))

    def forward(self, state, act):
        return (self.A @ state.T).T + (self.B @ act.T).T

class LINMOD(nn.Module):
    def __init__(self, state_dim, act_dim, state_enc_dim, act_enc_dim, hid_dim=64, extra_hidden=False):
        super(LINMOD, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_enc_dim = state_enc_dim
        self.act_enc_dim = act_enc_dim
        self.hid_dim = hid_dim
        self.extra_hidden=extra_hidden
        self.non_linearity = T.nn.Tanh()

        # State encoder
        fc_state_enc_layers = []
        fc_state_enc_layers.append(T.nn.Linear(self.state_dim, self.state_enc_dim, bias=True))
        fc_state_enc_layers.append(self.non_linearity)
        if extra_hidden:
            fc_state_enc_layers.append(T.nn.Linear(self.state_enc_dim, self.state_enc_dim, bias=True))
            fc_state_enc_layers.append(self.non_linearity)
        self.fc_state_encoder = nn.Sequential(*fc_state_enc_layers)

        # State decoder
        fc_state_dec_layers = []
        if extra_hidden:
            fc_state_dec_layers.append(T.nn.Linear(self.state_enc_dim, self.state_enc_dim, bias=True))
            fc_state_dec_layers.append(self.non_linearity)
        fc_state_dec_layers.append(T.nn.Linear(self.state_enc_dim, self.state_dim, bias=True))
        self.fc_state_decoder = nn.Sequential(*fc_state_dec_layers)

        # Action encoder
        fc_act_enc_layers = []
        fc_act_enc_layers.append(T.nn.Linear(self.act_dim, self.act_enc_dim, bias=True))
        fc_act_enc_layers.append(self.non_linearity)
        if extra_hidden:
            fc_act_enc_layers.append(T.nn.Linear(self.act_enc_dim, self.act_enc_dim, bias=True))
            fc_act_enc_layers.append(self.non_linearity)
        self.fc_act_encoder = nn.Sequential(*fc_act_enc_layers)

        # Action decoder
        fc_act_dec_layers = []
        if extra_hidden:
            fc_act_dec_layers.append(T.nn.Linear(self.act_enc_dim, self.act_enc_dim, bias=True))
            fc_act_dec_layers.append(self.non_linearity())
        fc_act_dec_layers.append(T.nn.Linear(self.act_enc_dim, self.act_dim, bias=True))
        self.fc_act_decoder = nn.Sequential(*fc_act_dec_layers)

        # Linear system
        self.A = T.nn.Parameter(T.randn((self.state_enc_dim, self.state_enc_dim), requires_grad=True))
        self.B = T.nn.Parameter(T.randn((self.state_enc_dim, self.act_enc_dim), requires_grad=True))

        #T.nn.init.xavier_uniform_(self.fc1.weight)
        #self.fc1.bias.data.fill_(0.01)

    def encode_state(self, state):
        return self.fc_state_encoder(state)

    def decode_state(self, state_enc):
        return self.fc_state_decoder(state_enc)

    def encode_act(self, act):
        return self.fc_act_encoder(act)

    def decode_act(self, act_enc):
        return self.fc_act_decoder(act_enc)

    def forward_encoded_state(self, state_enc, act_enc):
        return (self.A @ state_enc.T).T + (self.B @ act_enc.T).T

    def forward(self, state, act):
        state_enc = self.encode_state(state)
        act_enc = self.encode_act(act)
        next_state_enc = self.forward_encoded_state(state_enc, act_enc)
        next_state = self.decode_state(next_state_enc)
        act_dec = self.decode_act(act_enc)
        state_dec = self.decode_state(state_enc)
        return next_state, state_dec, act_dec

class LINMOD_HYBRID(nn.Module):
    def __init__(self, state_dim, act_dim, state_enc_dim, act_enc_dim, hid_dim=64, extra_hidden=False):
        super(LINMOD_HYBRID, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_enc_dim = state_enc_dim
        self.act_enc_dim = act_enc_dim
        self.hid_dim = hid_dim
        self.extra_hidden=extra_hidden
        self.non_linearity = T.nn.Tanh() # T.tanh

        # State encoder
        fc_state_enc_layers = []
        fc_state_enc_layers.append(T.nn.Linear(self.state_dim, self.state_enc_dim, bias=True))
        fc_state_enc_layers.append(self.non_linearity)
        if extra_hidden:
            fc_state_enc_layers.append(T.nn.Linear(self.state_enc_dim, self.state_enc_dim, bias=True))
            fc_state_enc_layers.append(self.non_linearity)
        self.fc_state_encoder = nn.Sequential(*fc_state_enc_layers)

        # Action encoder
        fc_act_enc_layers = []
        fc_act_enc_layers.append(T.nn.Linear(self.act_dim, self.act_enc_dim, bias=True))
        fc_act_enc_layers.append(self.non_linearity)
        if extra_hidden:
            fc_act_enc_layers.append(T.nn.Linear(self.act_enc_dim, self.act_enc_dim, bias=True))
            fc_act_enc_layers.append(self.non_linearity)
        self.fc_act_encoder = nn.Sequential(*fc_act_enc_layers)

        # Linear system
        self.A = T.nn.Parameter(T.randn((self.state_enc_dim + self.state_dim, self.state_enc_dim + self.state_dim), requires_grad=True))
        self.B = T.nn.Parameter(T.randn((self.state_enc_dim + self.state_dim, self.act_enc_dim + self.act_dim), requires_grad=True))

    def encode_state(self, state):
        return self.fc_state_encoder(state)

    def encode_act(self, act):
        return self.fc_act_encoder(act)

    def forward_encoded_state(self, state_hyb, act_hyb):
        return (self.A @ state_hyb.T).T + (self.B @ act_hyb.T).T

    def forward(self, state, act):
        # Encode state and action
        state_enc = self.encode_state(state)
        act_enc = self.encode_act(act)

        # Make hybrid state
        state_hyb = T.concat([state, state_enc], dim=1)
        act_hyb = T.concat([act, act_enc], dim=1)

        next_state_hyb = self.forward_encoded_state(state_hyb, act_hyb)

        next_state = next_state_hyb[:, :self.state_dim]
        next_state_latent = next_state_hyb[:, self.state_dim:]

        return next_state, next_state_latent

class RNN(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=64):
        super(RNN, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.rnn_1 = T.nn.LSTM(self.obs_dim, self.hid_dim, self.hid_dim)
        self.fc3 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

    def forward(self, x, h=None):
        fc1 = T.tanh(self.fc1(x))
        rnn_1, _ = self.rnn_1(fc1, h)
        fc2 = self.fc2(rnn_1)
        return fc2

class TEPMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=256, n_hidden=1):
        super(TEPMLP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc2 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc3 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

        self.make_initialization()

        self.nonlin = T.tanh

    def forward(self, x):
        fc1 = self.nonlin(self.fc1(x))
        fc2 = self.nonlin(self.fc2(fc1) + fc1)
        out = self.fc3(fc2)
        return out

    def make_initialization(self):
        T.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

class TEPMLPDEEP(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=256, n_hidden=1):
        super(TEPMLPDEEP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc2 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc3 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc4 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc5 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc6 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc7 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

        T.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc5.weight)
        self.fc5.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc6.weight)
        self.fc6.bias.data.fill_(0.01)

        T.nn.init.xavier_uniform_(self.fc7.weight)
        self.fc7.bias.data.fill_(0.01)

        self.nonlin = F.tanh

    def forward(self, x):
        fc1 = self.nonlin(self.fc1(x))
        fc2 = self.nonlin(self.fc2(fc1))
        fc3 = self.nonlin(self.fc3(fc2))
        fc4 = self.nonlin(self.fc4(fc3))
        fc5 = self.nonlin(self.fc5(fc4))
        fc6 = self.nonlin(self.fc6(fc5))
        fc7 = self.fc7(fc6)
        return fc7

class TEPRNN(nn.Module):
    def __init__(self, n_waypts, hid_dim=32, hid_dim_2=6, num_layers=1, bidirectional=False):
        super(TEPRNN, self).__init__()
        self.n_waypts = n_waypts
        self.hid_dim = hid_dim
        self.hid_dim_2 = hid_dim_2
        self.bidirectional = bidirectional

        self.fc1 = T.nn.Linear(2, 6, bias=True)
        self.rnn = T.nn.LSTM(input_size=6, hidden_size=self.hid_dim, num_layers=num_layers, bias=True, batch_first=True, bidirectional=bidirectional)
        self.fc2 = T.nn.Linear(self.hid_dim * (1 + bidirectional), self.hid_dim_2, bias=True)
        self.fc3 = T.nn.Linear(self.hid_dim_2, 1, bias=True)

        self.nonlin = F.tanh

    def forward(self, x):
        x_reshaped = T.reshape(x, (len(x), self.n_waypts, 2))
        fc1 = self.nonlin(self.fc1(x_reshaped))
        rnn1, _ = self.rnn(fc1)
        fc2 = self.nonlin(self.fc2(rnn1))
        fc2_sum = T.sum(fc2, 1)
        fc3 = self.fc3(fc2_sum)
        #rnn1_reshaped = T.reshape(fc2, (len(x), self.n_waypts * self.hid_dim_2))
        #out = self.fc3(rnn1_reshaped)
        return fc3

class TEPRNN2(nn.Module):
    def __init__(self, n_waypts, hid_dim=64, hid_dim_2=6, num_layers=1, bidirectional=False):
        super(TEPRNN2, self).__init__()
        self.n_waypts = n_waypts
        self.hid_dim = hid_dim
        self.hid_dim_2 = hid_dim_2
        self.bidirectional = bidirectional

        self.fc1 = T.nn.Linear(1, 4, bias=True)
        self.rnn = T.nn.LSTM(input_size=4, hidden_size=self.hid_dim, num_layers=num_layers, bias=True, batch_first=True)
        self.fc2 = T.nn.Linear(self.hid_dim, 1, bias=True)

        self.nonlin = F.tanh

    def forward(self, x):
        fc1 = self.nonlin(self.fc1(x.unsqueeze(2)))
        rnn1, _ = self.rnn(fc1)
        fc2 = self.fc2(rnn1[:, -1, :])
        return fc2

class TEPTX(nn.Module):
    def __init__(self, n_waypts, embed_dim, num_heads, kdim):
        super(TEPTX, self).__init__()
        self.n_waypts = n_waypts
        self.embed_dim = embed_dim
        self.kdim = kdim

        self.fc_emb = T.nn.Linear(2, embed_dim, bias=True)
        self.fc_key = T.nn.Linear(embed_dim, kdim)
        self.fc_val = T.nn.Linear(embed_dim, kdim)

        self.tx1 = T.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=kdim, batch_first=True)
        self.tx2 = T.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=kdim, batch_first=True)

        self.fc_final = T.nn.Linear(embed_dim, 1)

        self.nonlin = F.tanh

        # T.nn.init.xavier_uniform_(self.fc3.weight)
        # self.fc3.bias.data.fill_(0.01)

    def get_pos_emb(self, batch_size, seq_len, d):
        p_emb = []
        for i in range(seq_len):
            emb_vec = []
            for j in range(d):
                if j % 2 == 0:
                    emb_vec.append(np.sin(i / (10000. ** ((2. * j) / d))))
                else:
                    emb_vec.append(np.cos(i / (10000. ** ((2. * j) / d))))
            p_emb.append(emb_vec)
        p_emb_T = T.tensor(p_emb, dtype=T.float32)
        p_emb_T = T.tile(p_emb_T, (batch_size, 1, 1))
        return p_emb_T

    def forward(self, x):
        # Reshape and turn to embedding
        x_reshaped = T.reshape(x, (len(x), self.n_waypts, 2))
        emb = self.nonlin(self.fc_emb(x_reshaped))

        # Get positional enc and add to emb
        pos_emb = self.get_pos_emb(batch_size=x.shape[0], seq_len=x.shape[1] // 2, d=self.embed_dim)
        emb = emb + pos_emb

        # Multi head attention layer 1
        key1 = self.fc_key(emb)
        val1 = self.fc_val(emb)
        attn_1, _ = self.tx1(emb, key1, val1, need_weights=False)

        # Multi head attention layer 2
        key2 = self.fc_key(attn_1)
        val2 = self.fc_val(attn_1)
        attn_2, _ = self.tx2(attn_1, key2, val2, need_weights=False)

        attn_2_summed = T.sum(attn_2, dim=1)
        out = self.fc_final(attn_2_summed)

        return out
