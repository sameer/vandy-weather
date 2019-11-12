import torch
from torch import nn
from sklearn import preprocessing

from config import WINDOW_SIZE, DEVICE, DTYPE


class Scaler():
    def __init__(self):
        self.scalerX = preprocessing.StandardScaler()
        self.scalerY = preprocessing.StandardScaler()
    
    def fit(self, X, y):
        X = torch.from_numpy(self.scalerX.fit_transform(X))
        y = torch.from_numpy(self.scalerY.fit_transform(y))
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return (X, y)
    
    def invert(self, predicted_y):
        return self.scalerY.inverse_transform(predicted_y.cpu())


def into_windows(data, window_size: int = WINDOW_SIZE):
    data_X = torch.empty((len(data) - window_size, window_size))
    data_y = torch.empty((len(data) - window_size, 1))
    for i in range(window_size, len(data)):
        data_X[i - window_size] = data[i-window_size:i]
        data_y[i - window_size] = data[i]
    return (data_X, data_y)


class WeatherLSTM(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 51, output_dim: int = 1, bidirectional: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = int(bidirectional) + 1
        self.lstm = nn.LSTM(1, hidden_dim, 2)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def initialize(self):
        '''
        hidden_state / cell_state: (num_layers * num_directions, window_size, hidden_size)
        '''
        shape = (2 * self.bidirectional, WINDOW_SIZE, self.hidden_dim)
        self.hidden = (torch.zeros(shape, dtype=DTYPE, device=DEVICE), torch.zeros(shape, dtype=DTYPE, device=DEVICE))
    
    def forward(self, inp):
        '''
        inp: (num_samples, window_size, num_features)
        out: (num_samples, prediction (1), num_features)
        '''
        out, self.hidden = self.lstm(inp, self.hidden)
        # Pull the last row as the output.
        out = self.linear(out[:, -1, :])
        return out
