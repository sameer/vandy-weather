import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing

from sanitize_data import read_from_tar, TORCH_FILENAME
from model import WeatherLSTM
from config import WINDOW_SIZE, DEVICE, DTYPE


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

    SAMPLES = 100

    time = data[-SAMPLES:,1]
    thermometer = torch.from_numpy(data[-SAMPLES:,5]).float()

    thermometer_X = torch.empty((len(thermometer) - WINDOW_SIZE, WINDOW_SIZE))
    thermometer_y = torch.empty((len(thermometer) - WINDOW_SIZE, 1))
    for i in range(WINDOW_SIZE, len(thermometer)):
        thermometer_X[i - WINDOW_SIZE] = thermometer[i-WINDOW_SIZE:i]
        thermometer_y[i - WINDOW_SIZE] = thermometer[i]
    scalerX = preprocessing.StandardScaler()
    scalerY = preprocessing.StandardScaler()
    thermometer_X = torch.from_numpy(scalerX.fit_transform(thermometer_X))
    thermometer_y = torch.from_numpy(scalerY.fit_transform(thermometer_y))
    thermometer_X = thermometer_X.reshape((thermometer_X.shape[0], thermometer_X.shape[1], 1))
    
    thermometer_X, thermometer_y = (thermometer_X.to(DEVICE).type(DTYPE), thermometer_y.to(DEVICE).type(DTYPE))

    model = WeatherLSTM()
    model.to(DEVICE)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)#torch.optim.LBFGS(model.parameters(), lr=0.7)
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
        plt.plot(time[-SAMPLES + WINDOW_SIZE:], scalerY.inverse_transform(thermometer_y.cpu()))
        plt.plot(time[-SAMPLES + WINDOW_SIZE:], scalerY.inverse_transform(model(thermometer_X).cpu()))
        plt.show()
