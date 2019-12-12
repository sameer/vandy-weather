import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from sanitize_data import read_from_tar, TORCH_FILENAME
from weather_format import WeatherDataset, WeatherRow
from model import WeatherLSTM
from config import WINDOW_SIZE, DEVICE, DTYPE, TRAIN_END, VALIDATE_END, BATCH_SIZE, HIDDEN_DIM, TOTAL_POINTS, REPRODUCIBLE


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

        model_errors = []
        last_errors = []
        avg_errors = []

        for i, feature in enumerate(TARGET_FEATURES):
            error = sum([abs(data[idx+VALIDATE_END+WINDOW_SIZE-1, feature] - test_results[idx][i]) for idx in range(len(test_results))])
            avg_error = sum([abs(data[idx+VALIDATE_END+WINDOW_SIZE-1, feature] - np.average(data[idx+VALIDATE_END:idx+VALIDATE_END+WINDOW_SIZE-1, feature], axis=0)) for idx in range(len(test_results))]);
            last_error = sum([abs(data[idx+VALIDATE_END+WINDOW_SIZE-1, feature] - data[idx+VALIDATE_END+WINDOW_SIZE-2, feature]) for idx in range(len(test_results))]);

            print("The error for {} was {}".format(feature_names[feature], error/len(test_data)));
            print("Average error: {}. This is {}% better than the average metric".format(avg_error/len(test_data), avg_error/error*100-100));
            print("Last error: {}. This is {}% better than the last value metric".format(last_error/len(test_data), last_error/error*100-100));
            model_errors.append(error)
            last_errors.append(last_error)
            avg_errors.append(avg_error)

        usable_features = [feature_names[feature] for i,feature in enumerate(TARGET_FEATURES)];
        y_pos = np.arange(len(usable_features));

        fig, ax = plt.subplots()

        for i in range(len(model_errors)):
            max_val = max(model_errors[i], last_errors[i], avg_errors[i]);
            model_errors[i] = model_errors[i] / max_val;
            avg_errors[i] = avg_errors[i] / max_val;
            last_errors[i] = last_errors[i] / max_val;

        plt.bar(y_pos+0.25, model_errors, width=0.25, alpha=0.8, color='b', label='LSTM Model');
        plt.bar(y_pos, avg_errors, width=0.25, alpha=0.8, color='g', label='Average Model');
        plt.bar(y_pos-0.25, last_errors, width=0.25, alpha=0.8, color='r', label='Error Model');

        ticklabels = [usable_features[i][0]+usable_features[i][-1] for i in range(len(usable_features))]
        plt.xticks(y_pos+0.375, ticklabels)
        plt.xlabel('Feature')
        plt.ylabel('Relative L1 Error')
        plt.title('Prediction Method Errors')
        plt.legend()
        plt.tight_layout()

        plt.savefig('errors.png');
        plt.clf()
