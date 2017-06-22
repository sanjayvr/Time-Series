# importing series class used for time labeled data
from pandas import Series
# importing modules for plotting
import matplotlib
# Switching plot backend to png format as the linux distro is headless 
matplotlib.use('agg',warn=False,force=True)
from matplotlib import pyplot


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
	df = Series.to_frame(data)

	return data, df


def plot(data_series):
	'''
	This function is used to plot the dataset in different styles
	'''
	print('1 - Line Plot | 2 - Dot Plot | 3 - Histogram')
	plot_type = input('Enter the desired plot type by above numbers: ')

	if plot_type == 1:
		# Line Plot
		pyplot.plot(data_series)
		pyplot.show()
		pyplot.savefig('LinePlot')

	elif plot_type == 2:
		# Dot Plot
		pyplot.plot(data_series, 'r--')
		pyplot.show()
		pyplot.savefig('DotPlot')

	elif plot_type == 3:
		# Histogram
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
	plot(data_series)
	print('--Process Completed--')


if __name__ == '__main__':
	main()
