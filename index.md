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


# Time Series Forecasting as Supervised Learning
*Time Series Forecasting problem can be transformed into supervised machine learning problem in order to test different standard algorithms on the dataset.*

### Supervised Learning
Supervised learning is a branch of Machine Learning that deals with training machines on datasets with labels. Supervised learning problems can be grouped into Classifcation or Regression problems.
+ Classification Problem - Classify the given input to a certain output label
+ Regression Problem - Calculate output which is a real value based on the given data. 

### Converting Time Series Data to Supervised Learning Data
In order to achieve this we could change the output of observation 1 to input of observation 2 and output of observation 2 to input of observation 3 and so on.
Example -

**Time Series Data**

| Time | O/P  |
|:----:|:----:|
|  01  | 100  |
|  02  | 110  |
|  03  | 120  |

**Supervised Learning Data**

|  I/P | O/P  |
|:----:|:----:|
|  x   | 100  |
| 100  | 110  |
| 110  | 120  |
| 120  |  y   |

*y - The value that needs to be forecasted




