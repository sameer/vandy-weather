from typing import NamedTuple, OrderedDict, BinaryIO, List
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing

from preprocessing import read_from_tar, TORCH_FILENAME


class WeatherLSTM(nn.Module):
    def __init__(self, window_size: int, input_dim: int = 1, hidden_dim: int = 51, bidirectional: bool = False):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = int(bidirectional) + 1
        self.lstm = nn.LSTM(1, hidden_dim, 2)
        self.linear = nn.Linear(hidden_dim, 1)
    
    def initialize(self):
        '''
        hidden_state / cell_state: (num_layers * num_directions, window_size, hidden_size)
        '''
        shape = (2 * self.bidirectional, self.window_size, self.hidden_dim)
        self.hidden = (torch.zeros(shape), torch.zeros(shape))
    
    def forward(self, inp):
        '''
        inp: (num_samples, window_size, num_features)
        out: (num_samples, prediction (1), num_features)
        '''
        # outputs = []
        # for sample in input.chunk(input.size(0), dim=0):
        #     output, (self.hidden_state, self.cell_state) = self.lstm(sample, (self.hidden_state, self.cell_state))
        #     output = self.linear(output)
        #     outputs += [output]
        # print(outputs[0].shape)
        # outputs = torch.stack(outputs, 1).squeeze(2)
        # print(outputs.shape)
        out, hidden = self.lstm(inp, self.hidden)
        print(out.shape)
        
        out = self.linear(out[:, -1, :])
        print(out.shape)
        return out


if __name__ == '__main__':
    if not os.path.isfile(f'{TORCH_FILENAME}.tar.xz'):
        print('Run preprocessing script first')
        exit()
    '''
    t = np.arange(0, len(weather))
    linear_component = np.polyfit(t, thermometer, 1)[0]
    thermometer_without_linear = thermometer - linear_component * t
    amplitudes = np.fft.fft(thermometer_without_linear)
    freqs = np.fft.fftfreq(len(amplitudes))
    indices = np.argsort(-amplitudes)
    t = np.arange(0, len(weather) + 100000)
    restored = np.zeros(len(t))
    for i in indices[: 21]:
        amplitude = np.absolute(amplitudes[i]) / len(t)
        phase = np.angle(amplitudes[i])
        restored += amplitude * np.cos(2 * np.pi * freqs[i] * t + phase)
    restored += linear_component * t
    plt.plot(t, restored)
    plt.plot(np.arange(0, len(thermometer)), thermometer)
    plt.show()
    '''
    torch_tar, torch_binary = read_from_tar(TORCH_FILENAME)
    data = torch.load(torch_binary)
    torch_tar.close()

    # Needed to obtain reproducible results for debugging
    np.random.seed(2)
    torch.manual_seed(2)

    WINDOW_SIZE = 15
    time = data[-1000:,1]
    thermometer = torch.from_numpy(data[-1000:,5]).float()
    
    thermometer_X = torch.empty((len(thermometer) - WINDOW_SIZE, WINDOW_SIZE))
    thermometer_y = torch.empty((len(thermometer) - WINDOW_SIZE, 1))
    for i in range(WINDOW_SIZE, len(thermometer)):
        thermometer_X[i - WINDOW_SIZE] = thermometer[i-WINDOW_SIZE:i]
        thermometer_y[i - WINDOW_SIZE] = thermometer[i]
    thermometer_X = thermometer_X.reshape((thermometer_X.shape[0], thermometer_X.shape[1], 1))

    model = WeatherLSTM(WINDOW_SIZE)
    # model.double() # Cast floats to double

    loss_func = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)#torch.optim.LBFGS(model.parameters(), lr=0.7)
    for t in range(10):
        model.initialize()
        def step_closure():
            optimizer.zero_grad()
            out = model(thermometer_X)
            loss = loss_func(out, thermometer_y)
            print(f'Iteration {t}, Loss: {loss.item()}')
            loss.backward()
            return loss
        optimizer.step(step_closure)

    with torch.no_grad():
        plt.plot(time[-1000 + WINDOW_SIZE:], thermometer_y)
        plt.plot(time[-1000 + WINDOW_SIZE:], model(thermometer_X))
        plt.show()
