from pandas import Series
import matplotlib
import numpy
import math
import warnings
import sys
# Ignoring package warnings
warnings.filterwarnings("ignore")
from numpy import log
# Switching plot backend to png format as the linux distro is headless
matplotlib.use('agg',warn=False,force=True)
import os
# Ignoring Tensorflow debugging information
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from matplotlib import pyplot
from pandas import concat
from pandas import DataFrame
from pandas import read_csv
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import MinMaxScaler
from pandas import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.anomaly import Anomaly
import time
import datetime


def load():

	'''
	This function is used to load the data file into series
	and dataframe format
	:param data: Dataset in Series Format
	:param df: Dataset in Data Frame Format
	:return: Data in Series, Data in Frame Format
	'''

	# loading data into a series
	data = Series.from_csv('testdata4.csv',header=0,parse_dates=[0],index_col=0)

	# converting series to a data frame
	values = DataFrame(data.values)
	print('The dataset has been loaded')
	print('\n')
	# print(data.head(5))
	return data, values


def to_supervised(data_frame):

	'''
	This function is used to convert to make it supervised model
	where the output of datapoint 1 acts as input for datapoint 2
	:param dataframe: it is the dataframe after concatenation
	:param train: training dataset
	:param test: testing dataset
	:param train_x: train dataset from t-1 column
	:param train_y: train dataset from t column
	:param test_x: test dataset from t-1 column
	:param test_y: test dataset from t column
	:return: train of t-1 and t, test of t-1 and t and concatenated df and train, test - 66/34%
	'''

	# concatenating the previous output and current output
	dataframe_supervised = concat([data_frame.shift(1),data_frame], axis=1)
	# setting input for first observation as 0 as there is no previous observation
	dataframe_supervised.fillna(0, inplace=True)
	# print('Converted dataframe from time series to supervised')
	# print(dataframe_supervised.head(5))
        # print('\n')

	# splitting the data to test and train datasets although \
	# training is not required
	total_data = dataframe_supervised.values
	train_split = int(len(total_data)*0.66)
	train, test = total_data[1:train_split], total_data[train_split:]

	# splitting training data based on column t-1 and t
	train_x, train_y = train[:,0], train[:,1]
	test_x, test_y = test[:,0], test[:,1]

	return train_x, train_y, test_x, test_y, dataframe_supervised, train, test


def differencing(data_series, interval=1):

	'''
	This function is used to make dataset stationary by differencing
	t and t-1 observation.
	:param diff: list of differenced values
	:return: stationary series
	'''

	diff = []

	# differencing t and t-1 to make the data stationary
	for i in range(interval, len(data_series)):
		value = data_series[i] - data_series[i-interval]
		diff.append(value)
	return Series(diff)


def inverse_difference(history, yhat, interval=1):

	'''
	This is a helper function is used to inverse the differenced
	value back to original observation value .
	:return: original value of the time step
	'''

	return yhat + history[-interval]


def inverted_dataset(data_series, differenced_data):

	'''
	This function is used to revert the differenced dataset to the
	original values dataset
	:return: dataset with the original values
	'''

	inverted = []

	# inverting the difference in order to restore to original values
	for i in range(len(differenced_data)):
		value = inverse_difference(data_series, differenced_data[i], len(data_series)-i)
		inverted.append(value)

	inverted = Series(inverted)
	return inverted


def scaled_dataset(data_series):

        '''
        This function is used to scale the values for LSTM network to range [-1,1]
        :param scaler: scaler model to scale the dataset
        :param scaled_series: dataset scaled to [-1,1]
        :param inverted_scaled_series: dataset reverted to original values
        :return: scaled dataset and inverted to original datset
        '''

        df = data_series.values
        df = df.reshape(len(df),1)

	# scaler model
        scaler = MinMaxScaler(feature_range=(-1,1))

	# fitting the data onto the scalar model
	scaler = scaler.fit(df)
        scaled_df = scaler.transform(df)
        scaled_series = Series(scaled_df[:,0])
        inverted_df = scaler.inverse_transform(scaled_df)

	# inverting the scaled values to original
	inverted_scaled_series = Series(inverted_df[:,0])

        return scaled_df


def scaler_function(train,test):

	'''
	This is a helper function for scaling the input and is similar to
	scaled_dataset function.
	:param scaler: it is the scalar model
	:param train_scaled: scaled train dataset
	:param test_scaled: scaled test dataset
	:return: scaled model and scaled train dataset and scaled test dataset
	'''

	# scaler model
	scaler = MinMaxScaler(feature_range=(-1,1))
	scaler = scaler.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])

	# train dataset scaled
	train_scaled = scaler.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])

	# test dataset scaled
	test_scaled = scaler.transform(test)

	return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):

	'''
	This is a helper function to inverse the scaled values
	:return: scaled dataset inverted
	'''

	new_row = [ x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))

	# inverting the scaled data
	inverted = scaler.inverse_transform(array)

	return inverted[0, -1]


def persistence_forecast(train_x, train_y, test_x, test_y):

	'''
	This function is used to get the baseline by showing the persistence plot for given dataset
	by using the persistence model function below.
	:param predictions: store the baseline predictions using persistence model
	:return: Persistence Plot Figure
	'''

	# walk forward validation
	predictions = []
	for i in test_x:
		yhat = persistence_model(i)
		predictions.append(yhat)
	test_score = mean_squared_error(test_y, predictions)
	rmse = math.sqrt(test_score)
	print('The baseline model i.e., Persistence Model has been built')
	print('Test RMSE: %.3f' % rmse)
	print('\n')

	# plotting predictions and expected results
	pyplot.plot(test_y)
	pyplot.plot(predictions,'r--')
	pyplot.show()
	pyplot.savefig('Persistence Plot')


def persistence_model(x):

	'''
	This is the persistence model where the output of a datapoint is the input of the same datapoint
	:return: prediction x for given timestamp
	'''
	# return input as the output for making it baseline
	return x


def autocorrelation_check(data_series, dataframe):

	'''
	This function is used to check the autocorrelation between the t-1 and t+1 at every datapoint
	:return: Results based on the test performed
	'''

	print(' 1 - Lag Plot | 2 - Pearson Correlation Test | 3 - Autocorrelation Plot ')
	print('\n')
	which_test = input('Which of the above tests do you want to do? ')
	print('\n')

	if which_test == 1:
		lag_plot(data_series)
		pyplot.show()
		pyplot.savefig('Lag Plot')
		print('Plotted Log Plot')
	        print('\n')

	elif which_test == 2:
		result = dataframe.corr()
		print('Pearson Correlation Test')
		print(result)
	        print('\n')

	elif which_test == 3:
		autocorrelation_plot(data_series)
		pyplot.show()
		pyplot.savefig('Autocorrelation Plot')
		print('Plotted Autocorrelation Plot')
	        print('\n')

	else:
		print('Check the number your entered number')
	        print('\n')


def autoregression_autotrain(dataseries):

	'''
	The function is used build autoregression model but has the capability to utilize the historical data
	to autotrain and give the predictions instead of retraining at every step.
	:param data_values: values from data series
	:param train: train dataset
	:param test: test dataset
	:param model: created model for AR
	:param model_fit: trained model
	:param window: optimal lag
	:param coef: list of coefficients in the trained model
	:param history:  history from the prior trained model
	:param predictions: predictions made using the prior trained model {yhat = b0 + b1*x1 +..+ bn*xn}
	:return: AR Plot and RMSE of the model
	'''

	data_values = dataseries.values

	train_size = int(len(data_values)*0.66)

	# splitting the dataset into train and test dataset
	train, test = data_values[:train_size], data_values[train_size:]

	model = AR(train)

	# fitting the model using train dataset
	model_fit = model.fit()
	window = model_fit.k_ar
	coef = model_fit.params
	history = train[len(train)-window:]
	history = [history[i] for i in range(len(history))]
	predictions = list()
	for t in range(len(test)):
		length = len(history)
		lag = [history[i] for i in range(length-window,length)]
		yhat = coef[0]
		for d in range(window):
			yhat += coef[d+1] * lag[window-d-1]
		obs = test[t]
		predictions.append(yhat)
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat,obs))
	        print('\n')

	# RMSE used to measure the quality of model
	error = mean_squared_error(test, predictions)
	rmse = math.sqrt(error)
	print('AutoRegression Model with Auto Train has been built')
	print('Test RMSE: %.3f' % rmse)
        mean_test = numpy.mean(test)
        error_percent = ((rmse/mean_test)*(100))
	print('Error Percentage: %.3f' % error_percent)
        print('\n')

	pyplot.plot(test)
	pyplot.plot(predictions, 'r--')
	pyplot.show()
	pyplot.savefig('AR-AutoTrain Plot')


def autoregression_retrain(dataseries):

	'''
	data_values: values loaded from series format
	:param train, test: train and test datasets
	:param model: created model for AR
	:param model_fit: trained model
	:param predictions: predictions made from AR
	:return: AR Plot and RMSE
	'''

	data_values = dataseries.values
	train, test = data_values[1:len(data_values)-10], data_values[len(data_values)-10:]
	model = AR(train)
	model_fit = model.fit()
	# Lag is the optimal lag used and coefficients are from the trained model
	print('Lag: %s' % model_fit.k_ar)
	print('Coefficients: %s' % model_fit.params)
	predictions  = model_fit.predict(start = len(train), end = len(train)+len(test)-1, dynamic = False)
	for i in range(len(predictions)):
		print('predicted = %f, expected = %f' % (predictions[i], test[i]))
	error = mean_squared_error(test,predictions)
	rmse = math.sqrt(error)
	print('Test RMSE: %.3f' %rmse)
        print('\n')

	pyplot.plot(test)
	pyplot.plot(predictions, 'r--')
	pyplot.show()
	pyplot.savefig('AR-Retrain Plot')


def arima_model(data_series, p, d, q):

	'''
	This function is the ARIMA model built for forecasting on the time series dataset.
	:param df: values from data
	:param train: training dataset
	:param test: testing dataset
	:param predictions: predicted values
	:param model: Creating model for ARIMA
	:param model_fit: Model fit on train data
	:return: RMSE and ARIMA Plot
	'''

	# fit model using (p, d, q) paramters set for running the ARIMA model
	df = data_series.values
	size = int(len(df)*0.66)
	train = df[1:size]
	test = df[size:]
	history = [x for x in train]
	predictions = []
	for t in range(len(test)):
		model = ARIMA(history, order=(p,d,q))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		# this is the arima model summary
		# print(model_fit.summary())
		print('predicted=%f, expected=%f' % (yhat,obs))
	error = mean_squared_error(test, predictions)
	rmse = math.sqrt(error)
	print('---------------------------------')
	print('Test RMSE: %.3f' % rmse)
	mean_test = numpy.mean(test)
	error_percent = ((rmse/mean_test)*(100))
	print('Error Percent: %.3f' % error_percent)
	print('---------------------------------')
        print('\n')

	# plot residual errors
	residuals = DataFrame(model_fit.resid)
	residuals.plot()
	pyplot.show()
	# line plot - residual errors
	pyplot.savefig('Fit Plot 1')
	residuals.plot(kind='kde')
	pyplot.show()
	# density plot - residual errors
	pyplot.savefig('Fit Plot 2')
	# residual error distribution
	print(residuals.describe())

	# arima predictionplot
	pyplot.plot(test)
	pyplot.plot(predictions, 'r--')
	pyplot.show()
	pyplot.savefig('ARIMA Plot')


def which_model(data_series):

	'''
	This function is used for ease of building AR, I, MA, ARMA or ARIMA Model based on the preference of the user
	:return: The required model and RMSE of the model with the set value
	'''

	print('1 - AR Model | 2 - I Model | 3 - MA Model | 4 - ARMA Model | 5 - ARIMA Model | 6 - All Models with Grid Search')
        print('\n')
	model_type = input('Enter the number of the desired model? ')
        print('\n')

	if model_type == 1:
		print('The default p value set is 1')
	        print('\n')
		arima_model(data_series, 1, 0, 0)

	elif model_type == 2:
		print('The default d value set is 1')
	        print('\n')
		arima_model(data_series, 0, 1, 0)

	elif model_type == 3:
		print('The default q value set is 1')
		print('\n')
		arima_model(data_series, 0, 0, 1)

	elif model_type == 4:
		print('The default p, q value set is 1,1')
	        print('\n')
		arima_model(data_series, 1, 0, 1)

	elif model_type == 5:
		print('The default p, d, q value set is 5,1,1')
	        print('\n')
		arima_model(data_series, 10, 1, 0)

	elif model_type == 6:
		print('This model runs for given range of p, d, q values and finally gives the best combination of p,d,q values to build arima model')
		print('It also shows all the AR, MA, I, ARMA, ARIMA RMSE values')
	        print('\n')
		grid_arima(data_series)


def grid_arima_model(dataset, arima_order):

	'''
	This function is a sub function used for the grid search for different p,d,q values
	:return: Returns the RMSE of the test p,d,q values
	'''

	# splitting the dataset to test and train
        train_size = int(len(dataset) * 0.66)
        train, test = dataset[0:train_size], dataset[train_size:]
        history = [x for x in train]

        predictions = list()

	# RMSE for difference p,d,q values
        for t in range(len(test)):
                model = ARIMA(history, order=arima_order)
                model_fit = model.fit(disp=0)
                yhat = model_fit.forecast()[0]
                predictions.append(yhat)
                history.append(test[t])

        error = mean_squared_error(test, predictions)
	error = math.sqrt(error)
        return error


def evaluate_models(df, p_values, d_values, q_values):

	'''
	This is function used for grid search and for understanding ARIMA model check above function for ARIMA
	:return: This returns the best p,d,q value for the dataset for user to modify their ARIMA model
	'''

	dataset = df.astype('float32')

	# storing the best score till current iteration and comparing to update the best score
        best_score, best_cfg = float("inf"), None
        for p in p_values:
                for d in d_values:
                        for q in q_values:
                                order = (p,d,q)
                                try:
                                        mse = grid_arima_model(dataset, order)
                                        if mse < best_score:
                                                best_score, best_cfg = mse, order
                                        print('%s MSE=%.3f' % (order,mse))
                                	print('\n')
			        except:
                                        continue
        print('Best Combination of p,d,q is %s MSE=%.3f' % (best_cfg, best_score))
        print('\n')


def grid_arima(data_series):

	'''
	This function sets the list of p,d,q values you want to test ARIMA model on the given dataset
	:return: Check ARIMA model for all possible combinations of p,d,q values
	'''

	warnings.filterwarnings("ignore")

	# autoregression value (p)
	p_values = [5, 6 ,7]

	# differencing value (d)
	d_values = range(0, 1)

	# moving average value (q)
	q_values = range(0, 1)
	df = data_series.values

	# evaluating different models by using all p,d,q values
	evaluate_models(df, p_values, d_values, q_values)


def adf_test(data_series):

	'''
	This function can be used to do Augumented Dickey-Fuller Test
	p-value is used to determine if the data is stationary or not
	:return: Whether the dataset is stationary or not
	'''

	df = data_series.values

	# log transform
	log_df = log(df)
	result = adfuller(df)
	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])
	print('Critical Values:')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key,value))
	        print('\n')


	# data stationarity test
	if result[1] > 0.05:

		print('The data has a unit root and is non stationary')
	        print('\n')
		log_result = adfuller(log_df)

		if log_result[1] > 0.05:
			print('The log transform of data still is non stationary')
		        print('\n')

		else:
			print('The log transform of data is stationary')
			print('p-value: %f' % log_result[1])
		        print('\n')

	else:

		print('The data does not have a unit root and is stationary')
		print('Differencing is required in the ARIMA model')
	        print('\n')


def acf(data_series):

	'''
	This function is used to do the Auto-Correlation Function Plot
	:return: ACF Plot
	'''

	# acf plot
	plot_acf(data_series, lags=20)
	pyplot.show()
	pyplot.savefig('ACF Plot')


def pacf(data_series):

	'''
	This function is used to do the Partial Auto-Correlation Function Plot
	:return: PACF Plot
	'''

	# pacf plot
	plot_pacf(data_series, lags=20)
	pyplot.show()
	pyplot.savefig('PACF Plot')


def fit_lstm(train, batch_size, nb_epoch, neurons):

	'''
	This function is used to fit data to the LSTM Model
	model: sequential model used from Keras
	:return: model fit with the given training data
	'''

	X, Y = train[:, 0:-1], train[:,-1]
	X = X.reshape(X.shape[0], 1, X.shape[1])

	# sequential model
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape =(batch_size, X.shape[1], X.shape[2]), stateful = True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X,Y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model


def forecast_lstm(model, batch_size, X):

	'''
	This function is used to forecast the data using the trained model
	:return: data forecast
	'''

	X = X.reshape(1,1, len(X))

	# forecast from the trained model
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


def dataset(data, look_back = 10):

	'''
	This function is used to manipulate the data for lstm_two function
	:return: data for lstm_two
	'''

	data_X = []
	data_Y = []
	for i in range(len(data)-look_back-1):
		x = data[i:(i+look_back), 0]
		data_X.append(x)
		data_Y.append(data[i+look_back,0])
	return numpy.array(data_X), numpy.array(data_Y)


def lstm(data_series):

	'''
	This function is lstm built with no memory between batches and stateless
	value_mean: mean of performance data used to normalize RMSE
	diff_values: differenced values
	supervised_values: changing time series to supervised
	:param lstm_model: lstm model on which data is fit
	:return: Basic LSTM Model
	'''

	df = data_series.values
	value_mean = numpy.mean(df)

	# train set size
	df_train_size = int(len(df)*0.85)

	# differenced values
	diff_values = differencing(df,1)

	# converting time series to supervised
	train_x, train_y, test_x, test_y, dataframe_supervised, train, test = to_supervised(diff_values)
	supervised_values = dataframe_supervised.values

	# train and test datasets
	train_size = int(len(supervised_values)*0.85)
	train, test = supervised_values[:train_size], supervised_values[train_size:]

	# scaling data
	scaler, train_scaled, test_scaled = scaler_function(train, test)

	# number of times experiment repeated to reduce randomness in result
	repeats = 2
	error_scores = []

	for r in range(repeats):
		lstm_model = fit_lstm(train_scaled, 1, 50, 4)
		train_reshaped = train_scaled[:,0].reshape(len(train_scaled),1,1)
		lstm_model.predict(train_reshaped, batch_size = 1)

		predictions = []
		for i in range(len(test_scaled)):

			X, Y = test_scaled[i,0:-1], test_scaled[i,-1]
			yhat = forecast_lstm(lstm_model, 1, X)

			yhat = invert_scale(scaler, X, yhat)

			yhat = inverse_difference(df, yhat, len(test_scaled)+1-i)

			# predictions made to compare with test dataset
			predictions.append(yhat)
			expected = df[len(train)+i+1]
			print('Predicted=%f, Expected=%f' % (yhat,expected))

		# root mean square error
		rmse = math.sqrt(mean_squared_error(df[df_train_size:], predictions))

		# error percentage
		error_percent = ((rmse/value_mean)*100)
		print('%d. Test RMSE: %.3f' % (r+1, rmse))
		print('    Error Percent: %.3f' % error_percent)
		print('\n')
		error_scores.append(rmse)

	# comprehensive results from the predictions
	results = DataFrame()
	results['rmse'] = error_scores
	print(results.describe())
	print('\n')
	pyplot.plot(results, 'r--')
	pyplot.show()


def lstm_two(data_series, data_frame):

	'''
	This is a stacked LSTM model with memory between batches.
	:param scaler: scaled model for scaling data
	:param model: sequential model for the lstm
	:return: Stacked LSTM model with memory between batches for time series forecasting
	'''

	numpy.random.seed(7)

	# load dataset
	dataframe = read_csv('testdata4.csv', usecols=[1], engine='python', skipfooter=3)
	df = dataframe.values
	df = df.astype('float32')

	# test dataset
	size_train = int(len(df)*0.67)
	test_dataset = df[size_train:]

	# scaler model
	scaler = MinMaxScaler(feature_range=(0,1))
	df = scaler.fit_transform(df)

	# train and test dataset
	train_size = int(len(df)*0.67)
	train, test = df[0:train_size,:], df[train_size:len(df),:]

	# mean value of test dataset to normalize RMSE
	value_mean = numpy.mean(test_dataset)

	# the number of timesteps to lookback
	look_back = 10

	train_X, train_Y = dataset(train, look_back)
	test_X, test_Y = dataset(test, look_back)

	train_X = numpy.reshape(train_X, (train_X.shape[0], train_X.shape[1],1))
	test_X = numpy.reshape(test_X, (test_X.shape[0], test_X.shape[1],1))

	# setting batch size for the model
	batch_size = 1

	# sequential model for LSTM
	model = Sequential()

	# stacked LSTM
	model.add(LSTM(4, batch_input_shape=(batch_size,look_back,1), stateful=True, return_sequences=True))
	model.add(LSTM(4, batch_input_shape=(batch_size,look_back,1), stateful=True))
	model.add(Dense(1))

	# using adam optimizer
	model.compile(loss='mean_squared_error', optimizer='adam')


	# epochs for LSTM
	epochs = 10
	print('The current LSTM model runs for ' + str(epochs) + ' epochs')
	print('\n')
	for i in range(epochs):
		print('\t Running Epoch '+str(i+1)+'/'+str(epochs))
		print('\t -------------------')
		model.fit(train_X, train_Y, epochs = 1, batch_size=1, verbose=1, shuffle=False)
		model.reset_states()
		print('\n')

	# score, accuracy = model.evaluate(test_X, test_Y, batch_size = batch_size)

	# predictions made for train and test data
	predict_train = model.predict(train_X, batch_size=batch_size)
	predict_test = model.predict(test_X,batch_size=batch_size)

	predict_train = scaler.inverse_transform(predict_train)
	train_Y = scaler.inverse_transform([train_Y])
	predict_test = scaler.inverse_transform(predict_test)
	test_Y = scaler.inverse_transform([test_Y])

	# test dataset predictions error
	test_error = math.sqrt(mean_squared_error(train_Y[0], predict_train[:,0]))
	error_percent = (test_error/value_mean)*100
	print('RMSE: %.3f' % (test_error))
	print('Error Percent: %.3f' %(error_percent))

	# print('Test Accuracy: %.3f' % (accuracy))
	# print('Score: %.3f' % (score))

	plot_train = numpy.empty_like(df)
        plot_train[:,:] = numpy.nan
        plot_train[look_back:len(predict_train)+look_back, :] = predict_train

	plot_test = numpy.empty_like(df)
	plot_test[:,:] = numpy.nan
	plot_test[len(predict_train)+(look_back*2)+1:len(df)-1, :] = predict_test

	# plotting predictions against actual data
	pyplot.plot(scaler.inverse_transform(df))
	pyplot.plot(plot_test)
	pyplot.show()
	pyplot.savefig('LSTM2')


def anomaly_detection():

	detector = AnomalyDetector('m.csv')
	anomalies = detector.get_anomalies()
	score = detector.get_all_scores()
	anomaly_list = []

	for i in anomalies:
		anomaly = (str(i))
		anomaly = anomaly.split(' ')
		epoch_time = anomaly[2]
		time = datetime.datetime.utcfromtimestamp(float(epoch_time)/1000.)

		format = "%Y-%m-%d %H:%M:%S"
		anomaly_time = time.strftime(format)
		print('Anomaly detected on ' +  anomaly_time)

	'''
	anom_score = []

	for (timestamp,value) in score.iteritems():
		t_str = time.strftime('%Y-%m-%d', time.localtime(timestamp))
		anom_score.append([t_str, value])
	# print(anom_score)
	'''

def plot(data_series):

	'''
	This function is used to plot the dataset in different styles
	:return: Plots in different styles
	'''

	print('1 - Line Plot | 2 - Dot Plot | 3 - Histogram')
        print('\n')
	plot_type = input('Enter the desired plot type by above numbers: ')
        print('\n')

	if plot_type == 1:
		# line Plot
		pyplot.plot(data_series)
		pyplot.show()
		pyplot.savefig('LinePlot')

	elif plot_type == 2:
		# dot Plot
		pyplot.plot(data_series, 'r--')
		pyplot.show()
		pyplot.savefig('DotPlot')

	elif plot_type == 3:
		# histogram
		pyplot.hist(data_series)
		pyplot.show()
		pyplot.savefig('Histogram')

	else:
		print('Check the entered number!')


def main():

	'''
	Main Function
	:param data_file: input File stored in this variable
	:param data_series: data in series format
	:param data_frame: data in frame format
	:return: forecasting model
	'''

	# data_file = raw_input('Please enter the file name: ')
	print('\n')
	data_series, data_frame = load()
	# plot(data_series)
	# train_x, train_y, test_x, test_y, dataframe_supervised, train, test = to_supervised(data_frame)
	# persistence_forecast(train_x, train_y, test_x, test_y)
	# autocorrelation_check(data_series, dataframe_supervised)
	# autoregression_autotrain(data_series)
	# autoregression_retrain(data_series)
	# arima_model(data_series, 6, 1, 1)
	# grid_arima(data_series)
	# which_model(data_series)
	# adf_test(data_series)
	# acf(data_series)
	# pacf(data_series)
	# differenced_data = differencing(data_series)
	# inverted_data = inverted_dataset(data_series, differenced_data)
	# scaled_dataset(data_frame)
	# lstm(data_series)
	lstm_two(data_series,data_frame)
	# anomaly_detection()
	print('--Process Completed--')

if __name__ == '__main__':
	main()
