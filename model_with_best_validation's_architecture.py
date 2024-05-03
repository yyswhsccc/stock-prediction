import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_hidden, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = n_hidden // num_heads
        assert n_hidden % num_heads == 0, "Hidden size must be divisible by number of heads"

        self.query = nn.Linear(n_hidden, n_hidden)
        self.key = nn.Linear(n_hidden, n_hidden)
        self.value = nn.Linear(n_hidden, n_hidden)
        self.fc = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attn_weights, V)

        weighted_values = weighted_values.transpose(1, 2).contiguous()
        weighted_values = weighted_values.view(batch_size, -1, self.num_heads * self.head_dim)

        output = self.fc(weighted_values)
        return output
class StockPredictor(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(StockPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )
        self.attention = MultiHeadAttention(n_hidden * 2)
        self.linear1 = nn.Linear(in_features=n_hidden * 2, out_features=64)
        self.norm1 = nn.LayerNorm(64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.residual_layer1 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(in_features=64, out_features=64)  
        self.residual_layer2 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(in_features=64, out_features=32)
        self.linear5 = nn.Linear(in_features=32, out_features=1)

    def forward(self, sequences):
        lstm_out, _ = self.lstm(sequences)
        attn_output = self.attention(lstm_out[:, -1, :])
        y_pred = self.linear1(attn_output)
        y_pred = self.norm1(y_pred)

        residual1 = self.residual_layer1(y_pred)
        y_pred = self.linear2(y_pred + residual1)

        residual2 = self.residual_layer2(y_pred)
        y_pred = self.linear3(y_pred + residual2)

        y_pred = self.linear4(y_pred)
        y_pred = self.linear5(y_pred)
        return y_pred

import torch.nn.init as init

def init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

model = StockPredictor(
    n_features=16,
    n_hidden=128,
    seq_len=seq_length,
    n_layers=3
)

model.apply(init_weights)
