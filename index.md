# Introduction
*Time series datasets are datasets that have a component of time for every observation and can be used to either do analysis of the data or forecasting of future values.*

### Time Series Analysis
Time series analysis is primarily used to develop models that provide descriptions for the given dataset.

### Time Series Forecasting
Forecasting involves taking models fit on historical data and then predicting future values. 

### Components of Time Series
1. Level - Baselines value for the series
2. Trend - Increasing/Decreasing behavior of series over time
3. Seasonality - Repeated patterns over time
4. Noise 

### Suggestions for Forecasting
+ Take as much data as possible for testing, tuning and improving model
+ Prefer short time predictions for higher accuracy
+ Improve forecasts by updating the data points
+ Check if data frequency is too high
+ Identify the outlier values and missing gaps in order to improve the model


---------------------------------------------------------------------------------------------------------------------------------------

# Time Series Forecasting as Supervised Learning
*Time Series Forecasting problem can be transformed into supervised machine learning problem in order to test different standard algorithms on the dataset.*

### Supervised Learning
Supervised learning is a branch of Machine Learning that deals with training machines on datasets with labels. Supervised learning problems can be grouped into Classifcation or Regression problems.
+ Classification Problem - Classify the given input to a certain output label
+ Regression Problem - Calculate output which is a real value based on the given data. 

### Sliding Window Method
In order to convert time series data to supervised learning data we could change the output of observation 1 to input of observation 2 and output of observation 2 to input of observation 3 and so on.

Example -


**Time Series Data**

| Time | O/P  |
|:----:|:----:|
|  01  | 100  |
|  02  | 110  |
|  03  | 120  |


**Supervised Learning Data**

|  X1  | Y1   |
|:----:|:----:|
|  x   | 100  |
| 100  | 110  |
| 110  | 120  |
| 120  |  y   |


*x - The input value that has to be taken from previous output*
*y - The value that needs to be forecasted*


The intial row can be deleted as we don't have a prior observation. THe use of prior time outputs for next observation is called sliding window method and in statistics is called lag method. The number of previous time steps is called window width or lag size.

### Sliding Method with Multivariate Time Series Data
Univariate time series are dataset with single variable and multivariate time series datasets are datasets with two or more variables.

Example -


**Time Series Data**

| Time | Measure 1 | Measure 2 |
|:----:|:---------:|:---------:|
|  01  | 100       |  1        |
|  02  | 110       |  2        |
|  03  | 120       |  3        |


**Supervised Learning Data**


| X1  | X2  | X3  |  Y  |
|:---:|:---:|:---:|:---:|
| ?   | ?   | 100 |  1  |
| 100 | 1   | 110 |  2  |
| 110 | 2   | 120 |  3  |
| 120 | 3   |  ?  |  ?  |


The rows with unknown values can be removed and Y is the to be predicted column but in case we need to predict two different values then the following can be done


| X1  | X2  | Y1  |  Y2 |
|:---:|:---:|:---:|:---:|
| ?   | ?   | 100 |  1  |
| 100 | 1   | 110 |  2  |
| 110 | 2   | 120 |  3  |
| 120 | 3   |  ?  |  ?  |


Not many methods can handle prediction of multiple output values so need to choose suitable methods for the work.


### Sliding Window with Multi-Step Forecasting
Forecasting can either be one step or multi step i.e, prediction of either next time step or multiple future time steps.

Example -


**Time Series Data**

| Time | O/P  |
|:----:|:----:|
|  01  | 100  |
|  02  | 110  |
|  03  | 120  |


**Supervised Learning Data**

|  X1  | Y1   | Y2   |
|:----:|:----:|:----:|
|  x   | 100  | 110  |
| 100  | 110  | 120  |
| 110  | 120  |  ?   |
| 120  |  ?   |  ?   |

*We just use the first row to train the model and then predict both y1 and y2 values.*


---------------------------------------------------------------------------------------------------------------------------------------

# Time Series Forecasting with Python

### Load Time Series Data

```python
from pandas import Series
from pandas import DataFrame
data = Series.from_csv('xyz.csv', header = 0, parse_dates=[0], index_col=0)
values = DataFrame(data.values)
```

Useful Functions -
+ .head(n), .tail(n) - Peak the first and last n records in the series
+ .size() - Size of the series
+ .describe() - Gives Count, Mean, Standard Deviation, Median, Minimum, Maximum of the series

### Plotting Time Series

**Line Plot with Dotted Style**

```python
import matplotlib
matplotlib.use('agg',warn=False,force=True) # used to save plot on headless linux distros
from matplotlib import pyplot

pyplot.plot(data,'--') # '--' is used to style the plot to dotted line
pyplot.show()
pyplot.save('Figure_Name')
```

**Histogram**

```python
pyplot.hist(data)
```


### Establishing Baselines using Persistence Algorithm
A baseline in performance gives us an idea how other models are performing on our problem and before establishing baseline we need to
decide on the dataset split and performance measure. We can use Random Prediction algorithm or Zero Rule algorithm to form a baseline.

**Random Prediction Algorithm**
In this the prediction is just a random outcome from the training data and we always set a random number seed to make sure we always get same decidions everytime we run the algorithm. The algorithm takes all the unique output values of training dataset and randomly gives a value to the test dataset.

```python
from random import seed
from random import randrange

def random_prediction(train, test):
    output_values = [row[-1] for row in train] # storing all the train output values from the last column
    unique_output = list(set(output_values)
    predicted = []
    for row in test:
      rand_index = randrange(len(unique)) # chosing random index number
      predicted.append(unique[index]) # assigning the output value at the random index number
    return predicted
    
seed(1)
train = [[0], [1], [1], [0]]
test = [[None], [None], [None]]
predictions = random_prediction(train,test)
print (predictions)
```


**Zero Rule Algorithm**
In a classification problem for this algorithm we just assign the label with highest occurence to every instance in dataset.

```python
def zero_rule_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted
```


In a Regression problem we just use the mean of the output value observed in the training data.

```python
def zero_rule_regression(train, test):
    output_values = [rpw[-1] for row in train]
    prediction = sum(output_values) / float(len(output_values))
    predicted = [prediction for i in range(len(test))]
    return predicted
```


**Persistence Algorithm**
This is also called naive forecast. A persistence model can be implemented by -

1. Transform univariate dataset into supervised learning problem
```python
from pandas import concat
from pandas import DataFrame

values = DataFrame(data.values) # converting the data to dataframe
dataframe = concat([values.shift(1), values], axis=1) # creating Dataframe with Current Output and Previous Output
dataframe.columns = ['t-1', 't+1'] #setting column names
```

2. Split the dataset to train and test datasets.
```python
total_data = dataframe.values
train_size = int(len(total_data) * 0.66)
train, test = total_data[1:train_size], total_data[train_size:] # splitting Dataset to 66/34 % for train and test
train_x, train_y = train[:,0], train[:,1] # splitting train data based on columns t-1 and t+1
test_x, test_y = test[:,0], test[:,1] #splitting test data based on columns t-1 and t+1
```

3. Define the persistence model.
In this model we just return the input value as the prediction. So if we provide t-1 value to predict t+1 the value will be shown for t-1 value. Although it is not the right value we use this model for our baseline performance.
```python
def persistence_model(x):
    return x
```

4. Establish a baseline performance
```python
from sklearn.metrics import mean_squared_error
predictions = []
for i in test_x
    yhat = persistence_model(i)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
```

5. Review proble and plot the output
```python
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
```

*Root Mean Squared Error(RMSE)*
```python
rmse = sqrt(test_score)
```

### Autoregression
A regression model works on a model like y = b0 + b1 * x where y is the prediction, b0 and b1 are optimizing coefficients and x is input value. 


In time series, the previous time step output is used as input for the next observation and is called lag variable.

*x(t+1) = b0 + b1 * x(t-1) + b2 * x(t-2)*


An autoregression model assumes the observations at previous step are useful in prediction of the next time step and this is called correlation. If both are in same direction then it is positive correlation and if not negtive correlation.


One quick way to check for autocorrelation in the given dataset is by plotting lag plot 
```python
from pandas.tools.plotting import lag_plot
lag_plot(data_series)
```


Another quick way is to directly calculate the correlation between the observation and the lag variable using Pearson correlation coefficient. The correlation between two variables is shown with values between -1(negative correlation) to 1(positive correlation) and small values closer to zero indicate low correlation and high values above 0.5 or below -0.5 indicate high correlation.

```python
# taking the dataframe created with the lag variables
result = dataframe.corr()
print(result)
```


The more advanced way of checking the above is by using an autocorrelation plot

```python
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(data_series)
```


**AR Model using statsmodels**
```python
from statsmodels.tsa.ar_model import AR

data_values = data_series.values
train, test = data_values[1:len(data_values)-10], data_values[len(data_values)-10:] #testing only 10 observations

# train AR model
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar) # printing the chosen optimal lag
print('Coefficients: %s' % model_fit.params) # list of coefficients in the trained model

# test/forecast
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic = False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

#plot
pyplot.plot(test)
pyplot.plot(predictions, 'r--')
pyplot.show()
```

*The main drawback of the above model is that for every new observation we have to re-train the model which is not the most optimal way so instead we can create a history from the initial test and use those in the regression equation to come up with new forecasts.*

yhat = b0 + b1 * x1 + b2 * x2 + ... + bn * xn

```python
# train AR Model
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

# utilizing previously trained model
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]

# predictions
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

# RMSE
error = mean_squared_error(test, predictions)
rmse = math.sqrt(error)
print('Test RMSE: %.3f' % rmse)

#plot
pyplot.plot(test)
pyplot.plot(predictions, 'r--')
pyplot.show()
pyplot.savefig('AR-AutoTrain Plot')
```






