import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from sanitize_data import read_from_tar, TORCH_FILENAME
from weather_format import WeatherDataset, WeatherRow
from model import WeatherLSTM
from config import WINDOW_SIZE, DEVICE, DTYPE, TRAIN_END, VALIDATE_END, BATCH_SIZE, HIDDEN_DIM


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


    time = data[:TRAIN_END,1]
    TARGET_FEATURES = [3] + list(range(5, 10)) + list(range(11, 13)) + + list(range(14, 15)) + list(range(16,18))
    training_data = WeatherDataset(torch.from_numpy(data[:TRAIN_END, TARGET_FEATURES]).to(DEVICE, dtype=DTYPE))
    loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    model = WeatherLSTM(input_dim=len(TARGET_FEATURES), hidden_dim=HIDDEN_DIM, output_dim=len(TARGET_FEATURES))
    model.to(DEVICE, dtype=DTYPE)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)#torch.optim.LBFGS(model.parameters(), lr=0.7)
    for epoch in range(10):
        for step, batch in enumerate(loader):
            def step_closure():
                optimizer.zero_grad()
                # model.initialize()
                out = model(batch[:,:-1,:])
                loss = loss_func(out, batch[:,-1,:])
                print(f'Epoch {epoch}, Step {step} Loss: {loss.item()}')
                loss.backward()
                return loss
            optimizer.step(step_closure)

    with torch.no_grad():
        model = model.cpu()

        feature_names = list(WeatherRow.__annotations__.keys())
        validation_data = WeatherDataset(torch.from_numpy(data[TRAIN_END:VALIDATE_END,TARGET_FEATURES]).to(device=torch.device('cpu'), dtype=DTYPE), training_data.scaler)

        validation_results = [model(validation_data[idx][:-1,:].reshape((1, WINDOW_SIZE-1, len(TARGET_FEATURES))))[0,:] for idx in range(len(validation_data))]

        for i, feature in enumerate(TARGET_FEATURES):
            plt.title(feature_names[feature])
            plt.plot(data[TRAIN_END: VALIDATE_END - WINDOW_SIZE, 1], [validation_data[idx][-1, i] for idx in range(len(validation_data))])
            plt.plot(data[TRAIN_END: VALIDATE_END - WINDOW_SIZE, 1], [validation_results[idx][i] for idx in range(len(validation_data))])
            plt.savefig(f'validate-{feature_names[feature]}.png')

            error = sum([abs(validation_data[idx][-1, i] - validation_results[idx][i]) for idx in range(len(validation_data))])

            print();
            plt.clf()
