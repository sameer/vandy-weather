import torch
from torch import nn

from config import WINDOW_SIZE, DEVICE, DTYPE


class WeatherLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, bidirectional: bool = False, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = int(bidirectional) + 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    # def initialize(self):
    #     '''
    #     hidden_state / cell_state: (num_layers * num_directions, window_size, hidden_size)
    #     '''
    #     shape = (2 * self.bidirectional, WINDOW_SIZE - 1, self.hidden_dim)
    #     self.hidden = (torch.zeros(shape, dtype=DTYPE, device=DEVICE), torch.zeros(shape, dtype=DTYPE, device=DEVICE))
    
    def forward(self, inp):
        '''
        inp: (num_samples, window_size, num_features)
        out: (num_samples, prediction (1), num_features)
        '''
        out, _ = self.lstm(inp)
        # Pull the last row as the output.
        out = self.linear(out[:, -1, :])
        return out
