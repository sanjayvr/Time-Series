from pandas import Series
import matplotlib
import numpy
import math
# Switching plot backend to png format as the linux distro is headless
matplotlib.use('agg',warn=False,force=True)
from matplotlib import pyplot
from pandas import concat
from pandas import DataFrame
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR


def load(data_file):

	'''
	This function is used to load the data file into series, frame format
	data: Dataset in Series Format
	df: Dataset in Data Frame Format
	:return: Data in Series, Data in Frame Format
	'''

	# loading data into a series
	data = Series.from_csv(data_file,header=0,parse_dates=[0],index_col=0)

	# converting series to a data frame
	values = DataFrame(data.values)
	print(values.head(5))

	return data, values


def to_supervised(data_series, data_frame):

	'''
	This function is used to convert to meet supervised learning conditions
	dataframe: it is the dataframe after concatenation
	train: training dataset
	test: testing dataset
	train_x: train dataset from t-1 column
	train_y: train dataset from t+1 column
	test_x: test dataset from t-1 column
	test_y: test dataset from t+1 column
	:return: train of t-1 and t+1, test of t-1 and t+1 and concatenated df and train, test - 66/34%
	'''

	# concatenating the previous output and current output
	dataframe = concat([data_frame.shift(1),data_frame], axis=1)
	dataframe.columns = ['t-1','t+1']
	print(dataframe.head(5))

	# splitting the data to test and train datasets although \
	# training is not required
	total_data = dataframe.values
	train_split = int(len(total_data)*0.66)
	train, test = total_data[1:train_split], total_data[train_split:]

	# splitting training data based on column t-1 and t+1
	train_x, train_y = train[:,0], train[:,1]
	test_x, test_y = test[:,0], test[:,1]

	return train_x, train_y, test_x, test_y, dataframe, train, test


def persistence_forecast(train_x, train_y, test_x, test_y):

	'''
	This function is used to get the baseline by showing the persistence plot for given dataset
	predictions: store the baselien predictions using persistence model
	:return: Persistence Plot Figure
	'''

	# walk forward validation
	predictions = []
	for i in test_x:
		yhat = persistence_model(i)
		predictions.append(yhat)
	test_score = mean_squared_error(test_y, predictions)
	print('Test MSE: %.3f' % test_score)

	# plotting predictions and expected results
	pyplot.plot(test_y)
	pyplot.plot(predictions,'r--')
	pyplot.show()
	pyplot.savefig('Persistence Plot')


def persistence_model(x):

	'''
	This is the persistence model which results the output prediction same as input
	:return: prediction x for given timestamp
	'''

	return x


def autocorrelation_check(data_series, dataframe):

	'''
	This function is used to check the autocorrelation between the t-1 and t+1 at every datapoint
	:return: Results based on the test performed
	'''

	print(' 1 - Lag Plot | 2 - Pearson Correlation Test | 3 - Autocorrelation Plot ')
	which_test = input('Which of the above tests do you want to do? ')

	if which_test == 1:
		lag_plot(data_series)
		pyplot.show()
		pyplot.savefig('Lag Plot')

	elif which_test == 2:
		result = dataframe.corr()
		print(result)

	elif which_test == 3:
		autocorrelation_plot(data_series)
		pyplot.show()
		pyplot.savefig('Autocorrelation Plot')

	else:
		print('Check the number your entered number')


def autoregression_autotrain(dataseries):

	'''
	The function is used build autoregression model but has the capability to utilize the historical data
	to autotrain and give the predictions instead of retraining at every step.
	data_values: valuse from data series
	train: train dataset
	test: test dataset
	model: created model for AR
	model_fit: trained model
	window: optimal lag
	coef: list of coefficients in the trained model
	history:  history from the prior trained model
	predictions: predictions made using the prior trained model {yhat = b0 + b1*x1 +..+ bn*xn}
	:return: AR Plot and RMSE of the model
	'''

	data_values = dataseries.values
	train, test = data_values[600:len(data_values)-300], data_values[len(data_values)-300:]
	model = AR(train)
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
	error = mean_squared_error(test, predictions)
	rmse = math.sqrt(error)
	print('Test RMSE: %.3f' % rmse)

	pyplot.plot(test)
	pyplot.plot(predictions, 'r--')
	pyplot.show()
	pyplot.savefig('AR-AutoTrain Plot')


def autoregression_retrain(dataseries):

	'''
	data_values: values loaded from series format
	train, test: train and test datasets
	model: created model for AR
	model_fit: trained model
	predictions: predictions made from AR
	:return: AR Plot and RMSE
	'''

	data_values = dataseries.values
	train, test = data_values[1:len(data_values)-10], data_values[len(data_values)-10:]
	model = AR(train)
	model_fit = model.fit()
	# Lag is the optimal lag used and coefficients are from the trained model
	# print('Lag: %s' % model_fit.k_ar)
	# print('Coefficients: %s' % model_fit.params)
	predictions  = model_fit.predict(start = len(train), end = len(train)+len(test)-1, dynamic = False)
	for i in range(len(predictions)):
		print('predicted = %f, expected = %f' % (predictions[i], test[i]))
	error = mean_squared_error(test,predictions)
	rmse = math.sqrt(error)
	print('Test RMSE: %.3f' %rmse)

	pyplot.plot(test)
	pyplot.plot(predictions, 'r--')
	pyplot.show()
	pyplot.savefig('AR-Retrain Plot')


def plot(data_series):

	'''
	This function is used to plot the dataset in different styles
	'''

	print('1 - Line Plot | 2 - Dot Plot | 3 - Histogram')
	plot_type = input('Enter the desired plot type by above numbers: ')

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
	data_file: Input File stored in this variable
	data_series: Data in Series Format
	data_frame: Data in Frame Format
	'''

	data_file = raw_input('Please enter the file name: ')
	data_series, data_frame = load(data_file)
	# plot(data_series)
	# train_x, train_y, test_x, test_y, dataframe, train, test = to_supervised(data_series, data_frame)
	# persistence_forecast(train_x, train_y, test_x, test_y)
	# autocorrelation_check(data_series, dataframe)
	# autoregression_autotrain(data_series)
	# autoregression_retrain(data_series)

	print('--Process Completed--')

if __name__ == '__main__':
	main()
