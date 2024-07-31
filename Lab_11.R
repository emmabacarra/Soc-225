# SOC 225: Lab 11
# Emma Bacarra
# 07/31/2024
# 6 TASKS

# Welcome to Lab 11! Today, we will dive into the basics of data preprocessing and linear regression models. By the end of this lab, you should be able to clean a dataset, handle missing values, create new features, encode categorical variables, and build and evaluate simple linear regression models.

# Lab Goals
# -Data cleaning and preprocessing
# -Feature engineering
# -Creating dummy variables
# -Basic linear regression

# INTRODUCTION TO DATA PREPROCESSING

# Data preprocessing is a crucial step in data analysis and machine learning. It involves cleaning and transforming raw data into a format that can be easily analyzed. Common tasks include handling missing values, encoding categorical variables, and creating new features.

# Let's start with an example of handling missing values using the dplyr package. We will use a small dataset for demonstration purposes.
library(dplyr)
library(tidyr)

# Example dataset with missing values
data <- data.frame(
  var1 = c(1, 2, NA, 4, 5),
  var2 = c("A", "B", "A", "B", NA)
)

# Handling missing values
data_clean <- data %>%
  mutate(var1 = ifelse(is.na(var1), mean(var1, na.rm = TRUE), var1)) %>%
  drop_na(var2)

print(data_clean)

# TASK 1 ***********************************************************************
# Clean the following dataset by handling missing values. Replace missing values in 'age' with the median age, and drop rows with missing values in 'gender'. Print your result.

task1_data <- data.frame(
  age = c(25, 30, NA, 35, 40, NA),
  gender = c("Male", "Female", "Female", NA, "Male", "Female")
)

median_age <- median(task1_data$age, na.rm = TRUE)

task1_data %>%
  mutate(age = ifelse(is.na(age), median_age, age)) %>%
  drop_na(gender) %>% print()



# FEATURE ENGINEERING

# Feature engineering involves creating new features from existing data to improve the performance of your models. This can include combining or transforming variables.

# Here we'll create new features in a dataset by extracting information from a date column and transforming another variable.
# Dates are another variable type we haven't used too much in these labs. As always, type ?Dates for more information.
data <- data.frame(
  date = seq.Date(from = as.Date("2020-01-01"), by = "month", length.out = 12),
  value = rnorm(12, 100, 10)
)

# Creating new features
data <- data %>%
  mutate(
    month = format(date, "%m"),
    log_value = log(value)
  )

print(data)

# TASK 2 ***********************************************************************
# Create a new column 'year' by extracting the year from the 'date' column, and another column 'value_squared' by squaring the 'value' column. Print your result.

task2_data <- data.frame(
  date = seq.Date(from = as.Date("2015-01-01"), by = "quarter", length.out = 8),
  value = rnorm(8, 50, 5)
)

task2_data %>%
  mutate(
    year = format(date, "%Y"),
    value_squared = value^2
  ) %>% print()


# CREATING DUMMY VARIABLES

# Dummy variables are used to represent categorical data as numerical values. This is necessary for many machine learning algorithms that require numerical input.

# Let's create dummy variables for a categorical feature in a dataset.
data <- data.frame(
  gender = c("Male", "Female", "Female", "Male"),
  age = c(25, 30, 35, 40)
)

# We can create dummy variables using the model.matrix function. Type ?model.matrix for more information.
dummy_data <- model.matrix(~ gender - 1, data = data)

print(dummy_data)

# TASK 3 ***********************************************************************
# Create dummy variables for the 'department' column in the following dataset.

task3_data <- data.frame(
  department = c("HR", "Finance", "IT", "Marketing"),
  salary = c(50000, 60000, 55000, 65000)
)

model.matrix(~ department - 1, data = task3_data) %>% print()


# BASIC LINEAR REGRESSION

# Linear regression is a basic and commonly used type of predictive analysis. It allows us to model the relationship between a dependent variable and one or more independent variables.

# Let's build a simple linear regression model using the mtcars dataset.
data(mtcars)

# Building a linear regression model that looks at how well weight and horsepower predict the fuel economy of a car.
model <- lm(mpg ~ wt + hp, data = mtcars)

# The summary of the model shows the coeffecients, P Values, and R-squared values of our chosen model. We'll use these to intepret the effectiveness of our model. 
# To decipher a linear model, look at the variables under Coefficients. For every increase in weight by 1 unit, our model suggests a decline in fuel economy by 3.87. Likewise for horsepower, the model suggests a decline in mpg of 0.031 for every increase by 1 unit. The stars next to the summary table highlight whether the chosen coefficient is significantly different from 0 at different alpha levels, like the different tests we ran on Monday. If there's 3 stars, you likely have an important relationship!
summary(model)

# TASK 4 ***********************************************************************
# Build a simple linear regression model using the mtcars dataset. Predict 'mpg' using 'disp' and 'qsec' as independent variables. Summarize what your model is doing (Hint: type ?mtcars if you need more information about the variables used).
data(mtcars)

lm(mpg ~ disp + qsec, data = mtcars) %>% summary()


# TASK 5 ***********************************************************************
# Interpret the coefficients from the linear regression model you built in TASK 4 as best you can.

# intercept might represent when a predicted value of disp may overlap with qsec
# disp might represent the displacement of a model's predictions from the previous number
# qsec might represent the speed at which a model's predictions are changing


# TASK 6 ***********************************************************************
# Add the additional variable 'wt' to your model in TASK 4. How do the results change?

lm(mpg ~ disp + qsec + wt, data = mtcars) %>% summary()


# Conclusion
# In this lab, we explored the basics of data preprocessing and linear regression models. We learned how to clean a dataset, create new features, encode categorical variables, and build simple linear regression models. These skills are fundamental for analyzing and interpreting data in various domains.

# Be sure to SAVE THIS FILE and upload it into Canvas when you have completed all tasks. Please submit as a .R file.