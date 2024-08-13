# SOC 225: Lab 9
# Emma Bacarra
# 07/24/2024
# 5 TASKS

# Welcome to Lab 9! Today, we will continue our exploration of time series data analysis, focusing on basic models for time series forecasting and an introduction to ARIMA models. By the end of this lab, you should be able to apply simple moving averages, exponential smoothing, and use ARIMA models in base R.

# Lab Goals
# - Applying simple moving averages
# - Exponential smoothing
# - Fitting ARIMA models
# - Forecasting with ARIMA
# - Evaluating forecast accuracy
# - Visualizing forecast results

# BASIC MODELS FOR TIME SERIES FORECASTING

# SIMPLE MOVING AVERAGES

# Moving averages are used to smooth out short-term fluctuations and highlight longer-term trends or cycles. You use a few nearby data values in order, average them, and define that as a new point. 
# Let's start with a simple example of applying a moving average to a time series.

# Load necessary packages
library(ggplot2)

# First we're going to simulate a new set of time-series data to use in our examples. When generating random data, it is wise to set a seed value, which allows you to get the same "random" result each time.
set.seed(123)

# The rnorm in the function below generates a random distribution of 100 numbers with a mean of 50 and a standard deviation of 10. For more information, type ?rnorm. This is then plugged into the time_series function we used last week
time_series <- ts(rnorm(100, mean = 50, sd = 10), frequency = 12, start = c(2020, 1))

# Plot original time series. Note in our plot here that we're creating a dataframe for use in the graph within the plot call; the dataframe doesn't exist in the environment after running the call.
ggplot(data = data.frame(time = seq_along(time_series), value = as.numeric(time_series)), aes(x = time, y = value)) +
  geom_line() +
  labs(title = "Original Time Series", x = "Time", y = "Value")

# Apply simple moving average. 
window_size <- 3
# This function applies the moving average to our time series data. It'll create new values for each value in our time series besides the first and the last.
smoothed_series <- filter(time_series, rep(1/window_size, window_size), sides = 2)

# We can see how the values have changed by looking at them below
time_series
smoothed_series

# And then plotting the smoothed time series
ggplot(data = data.frame(time = seq_along(smoothed_series), value = as.numeric(smoothed_series)), aes(x = time, y = value)) +
  geom_line(color = 'blue') +
  labs(title = "Smoothed Time Series (Moving Average)", x = "Time", y = "Value")

# TASK 1 ***********************************************************************
# Using the above example as a guide, apply a moving average with a window size of 5 to the time series data and plot the results. How is this time series different than the one with a window size of 3?
window_size <- 3
smoothed_series <- filter(time_series, rep(1/window_size, window_size), sides = 2)

ggplot(data = data.frame(time = seq_along(smoothed_series), value = as.numeric(smoothed_series)), aes(x = time, y = value)) +
  geom_line(color = 'green') +
  labs(title = sprintf("Smoothed Time Series (Moving Average, Window Size = %s)", window_size), x = "Time", y = "Value")

# window size 5 is slightly smoother compared to window size 3 and the range of values is less


# EXPONENTIAL SMOOTHING

# Exponential smoothing is a technique that applies decreasing weights to past observations. This means they matter less for our smoothed points. 
# Let's apply exponential smoothing to our time series data.

# Apply exponential smoothing. Don't worry too much about the particulars of this method, but if you're interested type ?HoltWinters.
exp_smooth_series <- HoltWinters(time_series)

# Plot the smoothed time series
plot(exp_smooth_series, main = "Exponential Smoothing")

# TASK 2 ***********************************************************************
# Apply exponential smoothing to the Nottingham Castle data from last lab and plot the results. Describe any patterns you observe. Which smoothing technique fits better? Why do you think so?
data("nottem")
nottem_ts <- ts(nottem, start = c(1920, 1), frequency = 12)
exp_smooth_nottingham <- HoltWinters(nottem_ts)
plot(exp_smooth_nottingham, main = "Exponential Smoothing (Nottingham Castle Data)")


# ARIMA MODELS
install.packages("xts")
install.packages("forecast", dependencies = TRUE)
library(forecast)
# Introduction to ARIMA Models
# ARIMA (AutoRegressive Integrated Moving Average) is a popular time series forecasting model. These stats behind these models are well beyond this course, but data scientists like to use them to represent future changes in time series data.
# Let's fit an ARIMA model to our time series data.

# Fit ARIMA model
arima_model <- auto.arima(time_series)

# Print model summary
summary(arima_model)

# TASK 3 ***********************************************************************
# Fit an ARIMA model to the Nottingham Castle data as well. Don't worry too much about the output of the model summary for now, we'll go more into that next week.
arima_model_nottingham <- auto.arima(nottem_ts)
summary(arima_model_nottingham)


# FORECASTING WITH ARIMA

# Once we have fitted an ARIMA model, we can use it to forecast future values.
# Let's forecast the next 12 periods of our simulated time series.

# Forecast the next 12 periods. Type ?forecast to learn more.
forecast_values <- forecast(arima_model, h = 12)

# Plot forecast
plot(forecast_values, main = "ARIMA Forecast")

# TASK 4 ***********************************************************************
# Use the Nottingham ARIMA model from Task 3 to forecast the next 12 periods and plot the forecasted values along with the original time series. Which forecast do seems to fit better with the observed values? Why do you think so?
forecast_nottingham <- forecast(arima_model_nottingham, h = 12)
plot(forecast_nottingham, main = "ARIMA Forecast (Nottingham Castle Data)")
# this forecast follows the trend of the previous spikes, showing that it could fit in well with the data


# EVALUATING FORECAST ACCURACY

# Evaluating the accuracy of forecasts is crucial for determining the reliability of our model. One way to do this without being a fortune teller is to see how reliable your model is at predicting a subset of the data you have.
# We will split our data into training and test sets and evaluate our model's accuracy.

# Split data into training and test sets. Type ?window for more information.
train <- window(time_series, end = c(2022, 12))
test <- window(time_series, start = c(2023, 1))

# Fit ARIMA model on training data
arima_model_train <- auto.arima(train)

# Forecast on test data
forecast_test <- forecast(arima_model_train, h = length(test))

# Calculate accuracy and compare the values. A close model would have similar point values. Type ?accuracy for more information.
accuracy(forecast_test, test)

# We can also see what values were predicted by the training model using the plots we learned above. Compare these values to our observed time series plot to compare accuracy.
plot(forecast_test, main = "Training Data Forecast")
plot(time_series, main = "Time Series Data")

# TASK 5 ***********************************************************************
# Create a training and test set for the Nottingham Data. The training data should end in December 1929 and the test data should start in January 1930. Report the accuracy metrics for your model and plot the forecasted values of the training model. Which forecast was closer?
train_nottingham <- window(nottem_ts, end = c(1929, 12))
test_nottingham <- window(nottem_ts, start = c(1930, 1))

arima_model_train_nottingham <- auto.arima(train_nottingham)
forecast_test_nottingham <- forecast(arima_model_train_nottingham, h = length(test_nottingham))

accuracy(forecast_test_nottingham, test_nottingham)

plot(forecast_test_nottingham, main = "Training Data Forecast (Nottingham Castle Data)")
plot(nottem_ts, main = "Nottingham Castle Time Series Data")
# the RMSE model (root mean square error) seems like it had the best performance


# Conclusion
# In this lab, we continued our exploration of time series data analysis. We applied simple moving averages, exponential smoothing, and fitted ARIMA models to our data. Additionally, we forecasted future values, evaluated forecast accuracy, and visualized forecast results. These skills are essential for making reliable forecasts and understanding the underlying patterns in time series data.

# Be sure to SAVE THIS FILE and upload it into Canvas when you have completed all tasks. Please submit as an .R file.