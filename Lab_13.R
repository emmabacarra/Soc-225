# SOC 225: Lab 13
# Emma Bacarra
# 08/7/2024
# 6 TASKS

# Welcome to Lab 13, the Last Lab!!! Today, we will explore the basics of machine learning by building and evaluating simple linear and logistic regression models. By the end of this lab, you should be able to define machine learning, understand the concepts of linear and logistic regression, and build models using R.

# Lab Goals
# - Introduction to machine learning
# - Simple linear regression recap
# - Logistic regression
# - Practical machine learning example

# INTRODUCTION TO MACHINE LEARNING

# Machine learning is a field of artificial intelligence that uses algorithms to learn from and make predictions on data. It has applications in various domains such as healthcare, finance, and marketing.

# Machine learning builds off of statistical models. We'll review those again here, get those interpretations ready!

# SIMPLE LINEAR REGRESSION RECAP

# Linear regression is a basic and commonly used type of predictive analysis. It allows us to model the relationship between a dependent variable and one or more independent variables.

# Let's build a simple linear regression model using the diamonds dataset from the ggplot2 package.
library(ggplot2)
data(diamonds)

# Building a linear regression model to predict 'price' using 'carat' (weight) as the independent variable.
linear_model <- lm(price ~ carat, data = diamonds)

# Summary of the model to see coefficients, R-squared, and p-values. Note that as the value of 'carat' increases by 1, the price goes up by 7756.
summary(linear_model)

# TASK 1 ***********************************************************************
# Build a simple linear regression model using the iris dataset. Predict 'Sepal.Length' using 'Petal.Length' as the independent variable. Interpret your model results.
library(tidyr)
data(iris)

linear_model_iris <- lm(Sepal.Length ~ Petal.Length, data = iris)
summary(linear_model_iris)
# the Coefficients represent the slope and intercept of the line, the R-squared value represents the variations in sepal length that come from the petal lengths, and the p-value shows how closely related the two are

# LOGISTIC REGRESSION

# Logistic regression is used for classification problems where the dependent variable is categorical. It models the probability that a given input belongs to a certain category.

# We can use a logistic regression to predict whether a diamond is premium based on its characteristics.

# The 'cut' variable in the diamonds dataset is categorical with multiple levels. Let's simplify it to binary for this example (Ideal = 1, Not Ideal = 0).
diamonds$cut_binary <- ifelse(diamonds$cut == "Ideal", 1, 0)

# Building a logistic regression model to predict 'cut_binary' using 'carat' (weight) and 'depth'.
logistic_model <- glm(cut_binary ~ carat + depth, data = diamonds, family = binomial)

# Summary of the model to see coefficients, p-values, and model fit. We intrepret the coefficients of these models a little differently: instead of leading to a direct increase or decrease in the dependent variable, they represent an increase or decrease in the log odds of being in a particular category. That goes a bit beyond this class, so for now, take a general approach and suggest that a negative coefficient decreases the odds of being an 'Ideal' diamond and a positive coefficient increases the odds.
summary(logistic_model)

# TASK 2 ***********************************************************************
# Build a logistic regression model using the iris dataset. Predict 'Species' (Create a binary: setosa vs. non-setosa) using 'Sepal.Width' and 'Petal.Width' as independent variables. Interpret your model results.
iris$Species_binary <- ifelse(iris$Species == "setosa", 1, 0)

logistic_model_iris <- glm(Species_binary ~ Sepal.Width + Petal.Width, data = iris, family = binomial)
summary(logistic_model_iris)

# the Coefficients represent the differences in logarithmic odds of the dependent variable setosa being the value of 1 when a change occurs in the predictor, the p-value represents the importance of the predictor, and the signs of the coefficients represent the direction for which they are related

# EVALUATING MODEL PERFORMANCE

# For linear regression, we use metrics like R-squared and Mean Squared Error (MSE) to evaluate model performance.

# For logistic regression, we use metrics like accuracy, precision, recall, and the confusion matrix.

# Evaluating the linear regression model
# Calculate R-squared: This value suggests how much of the variation in the dependent variable is accounted for in our model.
r_squared <- summary(linear_model)$r.squared
print(paste("R-squared:", r_squared))

# Evaluating the logistic regression model
# Calculate predicted probabilities: This segment predicts whether diamonds are 'Ideal' based on the logistic model we previously made. It will produce a probability that the diamond is 'Ideal'.
predicted_probs <- predict(logistic_model, type = "response")

# Convert probabilities to binary predictions (threshold = 0.5)::If a diamond is given a probability greater than a coin flip of being ideal, we'll count it.
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)

# Create a confusion matrix: Like on Monday, we want to see which classifications the model got right and wrong.
confusion_matrix <- table(predicted_classes, diamonds$cut_binary)
print(confusion_matrix)

# TASK 3 ***********************************************************************
# Evaluate the linear regression model you built in TASK 1. Calculate and print the R-squared value.
r_squared_iris <- summary(linear_model_iris)$r.squared
print(paste("R-squared:", r_squared_iris))

# TASK 4 ***********************************************************************
# Evaluate the logistic regression model you built in TASK 2. Calculate and print the confusion matrix.
predicted_probs_iris <- predict(logistic_model_iris, type = "response")
predicted_classes_iris <- ifelse(predicted_probs_iris > 0.5, 1, 0)
confusion_matrix_iris <- table(predicted_classes_iris, iris$Species_binary)

print(confusion_matrix_iris)

# PRACTICAL MACHINE LEARNING EXAMPLE

# Let's use the diamonds dataset to build and evaluate a machine learning model.

# Load the diamonds dataset
data(diamonds)
diamonds$cut_binary <- ifelse(diamonds$cut == "Ideal", 1, 0)
# Split the dataset into training and testing sets
set.seed(123)
train_index <- sample(1:nrow(diamonds), 0.7 * nrow(diamonds))
train_data <- diamonds[train_index, ]
test_data <- diamonds[-train_index, ]

# Building a logistic regression model to predict whether a diamond is premium based on its characteristics.
ml_model <- glm(cut_binary ~ carat + depth + table + price, data = train_data, family = binomial)

# Summary of the model
summary(ml_model)

# Predicting on the test set
predictions <- predict(ml_model, newdata = test_data, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Creating a confusion matrix
confusion_matrix <- table(predicted_classes, test_data$cut_binary)
print(confusion_matrix)

# The logistic model will be our 'standard' approach. We'll now use a super simple machine learning model, the decision tree model we used last time.

# Building a decision tree model using the rpart package
library(rpart)

# Building the model
tree_model <- rpart(cut_binary ~ carat + depth + table + price, data = train_data, method = "class")

# Summary of the model
print(tree_model)

# Predicting on the test set
tree_predictions <- predict(tree_model, newdata = test_data, type = "class")

# Creating a confusion matrix
tree_confusion_matrix <- table(tree_predictions, test_data$cut_binary)
print(tree_confusion_matrix)

# TASK 5 ***********************************************************************
# Which model, ml_model or tree_model, correctly predicted more of the test dataset? What were the accuracy percentages of each?
ml_accuracy <- sum(predicted_classes == test_data$cut_binary) / length(test_data$cut_binary)
tree_accuracy <- sum(tree_predictions == test_data$cut_binary) / length(test_data$cut_binary)

print(paste("ML Model Accuracy:", ml_accuracy * 100, "%"))
print(paste("Tree Model Accuracy:", tree_accuracy * 100, "%"))

# TASK 6 ***********************************************************************
# Build a decision tree model using the rpart package to predict the species of iris flowers. Evaluate the model's performance on a test dataset. Calculate and print the accuracy of the model.
set.seed(123)
train_index_iris <- sample(1:nrow(iris), 0.7 * nrow(iris))
train_data_iris <- iris[train_index_iris, ]
test_data_iris <- iris[-train_index_iris, ]

tree_model_iris <- rpart(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = train_data_iris, method = "class")

tree_predictions_iris <- predict(tree_model_iris, newdata = test_data_iris, type = "class")

tree_accuracy_iris <- sum(tree_predictions_iris == test_data_iris$Species) / length(test_data_iris$Species)
print(paste("Decision Tree Model Accuracy:", tree_accuracy_iris * 100, "%"))

# Conclusion
# In this lab, we explored the basics of machine learning by building and evaluating simple linear and logistic regression models. We also built a practical machine learning model using the diamonds dataset. These skills are foundational for further study and application of machine learning techniques.

# Be sure to SAVE THIS FILE and upload it into Canvas when you have completed all tasks. Please submit as a .R file.