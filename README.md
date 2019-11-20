# vandy-weather

Predict weather at Vanderbilt using a neural network.

## Dataset

Graciously provided by Ed Mansouri, founder of WeatherSTEM for [Vanderbilt WeatherSTEM station](https://davidson.weatherstem.com/vanderbilt).

## Model

An LSTM RNN (long short-term memory recurrent neural network) is used for multivariate, multi time-step prediction of weather statistics (i.e. temperature, humidity, barometric pressure). All input data has to be standardized to improve learning.

This is implemented with PyTorch for an LSTM and [scikit-learn for standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler).

## Observations

* Weather Vane is useless as a prediction input and as a prediction output. It appears to be effectively random when you only have observations from one station.
    * Same goes for anemometer, but interestingly 10-minute wind gust *IS* predictable!
* UV radiation is just UV index. The prediction graph looks fine but it may be impossible to predict with small windows. There are cases where many timesteps have the same value and the next is suddenly different. The LSTM may conclude that predicting the same UV index is always best.

## Demo

Try it out on [Google Colab](https://colab.research.google.com/github/sameer/vandy-weather/blob/master/vandy_weather.ipynb).

## Code breakdown

* sanitize_data.py: cleans up the input measurements and gets rid of bad sensor data for use in PyTorch
* model.py: defines classes related to the prediction model
* config.py: various runtime configuration, like number of hidden units and whether to run on GPU or CPU
* main.py: loads PyTorch format data, standardizes it, runs training, and outputs validation plots
* vandy_weather.ipynb: runs sanitize_data.py and main.py, for use on Google Colab
