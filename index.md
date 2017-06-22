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

data = Series.from_csv('xyz.csv', header = 0, parse_dates=[0], index_col=0)
df = Series.to_frame(data)
```

Useful Functions -
+ .head(n), .tail(n) - Peak the first and last n records in the series
+ .size() - Size of the series
+ .describe() - Gives Count, Mean, Standard Deviation, Median, Minimum, Maximum of the series

### Plotting Time Series

**Line Plot with Dotted Style**

```python
import matplotlib
matplotlib.use('agg',warn=False,force=True) # Used to save plot on headless linux distros
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

**Random Prediction Algorith**
In this the prediction is just a random outcome from the training data and we always set a random number seed to make sure we always get same decidions everytime we run the algorithm. The algorithm takes all the unique output values of training dataset and randomly gives a value to the test dataset.

```python
from random import seed
from random import randrange

def random_prediction(train, test):
    output_values = [row[-1] for row in train] # Storing all the train output values from the last column
    unique_output = list(set(output_values)
    predicted = []
    for row in test:
      rand_index = randrange(len(unique)) # chosing random index number
      predicted.append(unique[index]) # Assigning the output value at the random index number
    return predicted
    
seed(1)
train = [[0], [1], [1], [0]]
test = [[None], [None], [None]]
predictions = random_prediction(train,test)
print (predictions)
```

**Zero Rule Algorith**
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














