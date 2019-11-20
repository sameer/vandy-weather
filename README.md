# vandy-weather

Forecast weather at Vanderbilt using a neural network suited for time series prediction.

## Dataset
Data for [Vanderbilt WeatherSTEM station](https://davidson.weatherstem.com/vanderbilt) 
graciously provided by Ed Mansouri, founder of WeatherSTEM,. 

### Caveats

* Includes observations from `2017-05-15 13:28:39` through `2019-09-19 13:55:15`
* There are some large time gaps in sensor measurement: ~ 1 month or less
* Measurements do not always occur regularly :sometimes every 2min, sometimes every 1min
* Last few sensor columns have many NULL values due to the recent addition of new sensors to the station for: `turf_temperature,club_level_temperature,club_level_humidity,field_level_temperature,field_level_humidity`
* dataset is NOT sorted by the time column, since it was obtained from a SQL dump and converted using [mysqldump-to-csv](https://github.com/jamesmishra/mysqldump-to-csv)
* id column does not increment by a specific amount

## Model

An LSTM RNN (long short-term memory recurrent neural network) is used for multivariate, multi time-step prediction of weather statistics (i.e. temperature, humidity, barometric pressure). All input data has to be standardized to improve learning.

This is implemented with PyTorch for an LSTM and [scikit-learn for standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler).

### Layers

* 2-layer LSTM with 51 hidden units
* Linear layer to convert from hidden dimension to output dimension

## Observations

* Weather Vane is useless as a prediction input and as a prediction output. It appears to be effectively random when you only have observations from one station.
    * Same goes for anemometer, but interestingly 10-minute wind gust *IS* predictable!
* UV radiation is just [UV index](https://www.weather.gov/ilx/uv-index). The prediction graph looks fine but it may be impossible to predict with small windows. There are cases where many timesteps have the same value and the next is suddenly different. The LSTM may conclude that predicting the same UV index is always best.

## Demo

Try it out on [Google Colab](https://colab.research.google.com/github/sameer/vandy-weather/blob/master/vandy_weather.ipynb).

## Code breakdown

* sanitize_data.py: cleans up the input measurements and gets rid of bad sensor data for use in PyTorch
* model.py: defines classes related to the prediction model
* config.py: various runtime configuration, like number of hidden units and whether to run on GPU or CPU
* main.py: loads PyTorch format data, standardizes it, runs training, and outputs validation plots
* vandy_weather.ipynb: runs sanitize_data.py and main.py, for use on Google Colab

## Resources

* PyTorch repo example: https://github.com/pytorch/examples/tree/master/time_sequence_prediction
* Link to list of helpful links: https://www.reddit.com/r/pytorch/comments/7w01gd/pytorch_for_time_series_forecasting/
* scikit-learn docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
* Insight on multivariate + multi timestep learning (not in PyTorch): https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
* colah's post on gaining LSTM intuition: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

