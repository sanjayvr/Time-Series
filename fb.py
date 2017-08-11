import matplotlib
import pandas
import numpy
from pandas import Series
from pandas import DataFrame
# Switching plot backend to png format as the linux distro is headless
matplotlib.use('agg',warn=False,force=True)
from matplotlib import pyplot
from fbprophet import Prophet

if __name__ == '__main__':

        df = pandas.read_csv('testdata5.csv')
	df['y'] = numpy.log(df['y'])

	df.head()
	m = Prophet()
	m.fit(df);
	future = m.make_future_dataframe(periods=10)
	future.tail()
	forecast = m.predict(future)
	forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
	forecast[['yhat']] = numpy.exp(forecast[['yhat']])
	print(forecast[['ds','yhat']])
	m.plot(forecast)
	pyplot.show()
	pyplot.savefig('Img')

