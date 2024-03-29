import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from sanitize_data import read_from_tar, TORCH_FILENAME
from weather_format import WeatherDataset, WeatherRow
from model import WeatherLSTM
from config import WINDOW_SIZE, DEVICE, DTYPE, TRAIN_END, VALIDATE_END, BATCH_SIZE, HIDDEN_DIM, TOTAL_POINTS, REPRODUCIBLE

import pandas as pd
from scipy import misc
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D


import os

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
    if REPRODUCIBLE:
        np.random.seed(2)
        torch.manual_seed(2)


    time = data[:TRAIN_END,1]
    TARGET_FEATURES = [3] + list(range(5, 10)) + list(range(11, 13)) + list(range(14, 15)) + list(range(16,18))
    training_data = WeatherDataset(torch.from_numpy(data[:TRAIN_END, TARGET_FEATURES]).to(DEVICE, dtype=DTYPE))
    validation_data = WeatherDataset(torch.from_numpy(data[TRAIN_END:VALIDATE_END,TARGET_FEATURES]).to(DEVICE, dtype=DTYPE), training_data.scaler)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=(VALIDATE_END-TRAIN_END) // 8, shuffle=False)

    model = WeatherLSTM(input_dim=len(TARGET_FEATURES), hidden_dim=HIDDEN_DIM, output_dim=len(TARGET_FEATURES))
    model.to(DEVICE, dtype=DTYPE)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)#torch.optim.LBFGS(model.parameters(), lr=0.7)
    previous_validation_loss = float('inf')
    for epoch in range(100):
        for step, batch in enumerate(train_loader):
            def step_closure():
                optimizer.zero_grad()
                # model.initialize()
                out = model(batch[:,:-1,:])
                loss = loss_func(out, batch[:,-1,:])
                print(f'Epoch {epoch+1}, Step {step+1} Loss: {loss.item()}')
                loss.backward()
                return loss
            optimizer.step(step_closure)
        with torch.no_grad():
            print('Evaluating model against validation set')
            model.eval()
            current_validation_loss = 0.0
            for batch in validation_loader:
                out = model(batch[:,:-1,:])
                current_validation_loss += loss_func(out, batch[:,-1,:]).item() * len(batch)
            current_validation_loss = current_validation_loss / (VALIDATE_END - TRAIN_END)
            model.train()
            should_stop_early = current_validation_loss > previous_validation_loss
            if should_stop_early:
                print(f'Stopping early, current validation loss {current_validation_loss} compared to previous validation loss {previous_validation_loss}')
            else:
                print(f'Current validation loss is {current_validation_loss}, down from previous {previous_validation_loss}')
            previous_validation_loss = current_validation_loss
            if should_stop_early:
                break

    print('Done training, now testing')
    with torch.no_grad():
        model.eval()

        feature_names = list(WeatherRow.__annotations__.keys())

        test_data = WeatherDataset(torch.from_numpy(data[VALIDATE_END:TOTAL_POINTS,TARGET_FEATURES]).to(device=DEVICE, dtype=DTYPE), training_data.scaler)

        test_results = [] #[model(test_data[idx][:-1,:].reshape((1, WINDOW_SIZE-1, len(TARGET_FEATURES))))[0,:] for idx in range(len(test_data))]

        print('Running model on test dataset')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=(TOTAL_POINTS-VALIDATE_END) // 8, shuffle=False)
        for step, batch in enumerate(test_loader):
            test_batch_results = test_data.scaler.inverse_transform(model(batch[:,:-1,:]).cpu().numpy())
            for i in range(len(test_batch_results)):
                test_results.append(test_batch_results[i])
            print(f'{step*test_loader.batch_size * 100.0 / len(test_data)}% done')

        '''

        print('Plotting test actual and predicted')
        for i, feature in enumerate(TARGET_FEATURES):
            plt.title(feature_names[feature])
            plt.plot(data[VALIDATE_END: TOTAL_POINTS - WINDOW_SIZE, 1], data[VALIDATE_END+WINDOW_SIZE: TOTAL_POINTS, TARGET_FEATURES[i]])
            plt.plot(data[VALIDATE_END: TOTAL_POINTS - WINDOW_SIZE, 1], [test_results[idx][i] for idx in range(len(test_data))])
            # plt.plot(data[VALIDATE_END: TOTAL_POINTS - WINDOW_SIZE, 1], [np.average(data[idx+VALIDATE_END:idx+VALIDATE_END+WINDOW_SIZE-1, feature], axis=0) for idx in range(len(test_results))])
            plt.savefig(f'test-{feature_names[feature]}.png')
            plt.clf()
        '''

        for i, feature in enumerate(TARGET_FEATURES):
            error = sum([abs(data[idx+VALIDATE_END+WINDOW_SIZE-1, feature] - test_results[idx][i]) for idx in range(len(test_results))])
            avg_error = sum([abs(data[idx+VALIDATE_END+WINDOW_SIZE-1, feature] - np.average(data[idx+VALIDATE_END:idx+VALIDATE_END+WINDOW_SIZE-1, feature], axis=0)) for idx in range(len(test_results))]);
            last_error = sum([abs(data[idx+VALIDATE_END+WINDOW_SIZE-1, feature] - data[idx+VALIDATE_END+WINDOW_SIZE-2, feature]) for idx in range(len(test_results))]);

            print("The error for {} was {}".format(feature_names[feature], error/len(test_data)));
            print("Average error: {}. This is {}% better than the average metric".format(avg_error/len(test_data), avg_error/error*100-100));
            print("Last error: {}. This is {}% better than the last value metric".format(last_error/len(test_data), last_error/error*100-100));

        samples = []
        colors = []
        feature = TARGET_FEATURES[1];
        thresh = 0.8507940004638544;
        sampling_rate = 100

        for i in range(len(test_data)):
            if i % sampling_rate == 0:
                samples.append(data[VALIDATE_END+i:VALIDATE_END+i+WINDOW_SIZE+1,:].reshape(-1))
                colors.append('b' if abs(data[VALIDATE_END+WINDOW_SIZE+i-1, feature] - test_results[i][feature]) < thresh else 'r')

        df = pd.DataFrame.from_records(samples, coerce_float=True)
        print("SAMPLES LENGTH IS {}".format(len(samples)))
        iso = manifold.Isomap(n_neighbors=6, n_components=3);
        iso.fit(df);

        isomap = iso.transform(df);

        #2 isomap components
        fig = plt.figure();
        ax = fig.add_subplot(111)
        ax.set_title("ISO transformation 2D")

        ax.scatter(isomap[:,0], isomap[:,1], marker='.', c=colors)
        plt.savefig('2disomap.png')
        plt.clf()


        #3 isomap components
        fig = plt.figure();
        ax = Axes3D(fig)
        ax.set_title("ISO transformation 3D");

        ax.scatter(isomap[:,0], isomap[:,1], isomap[:,2], marker='.', c=colors)
        plt.title(feature_names[feature]);
        plt.savefig('isomap.png');
        plt.clf();


        print("SUCCESS");

