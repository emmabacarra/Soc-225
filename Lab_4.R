# SOC 225: Lab 4
# Emma Bacarra
# 7/3/2024
# 5 Tasks

# Welcome to Lab 4 for SOC 225! In this lab, we'll focus on Exploratory Data Analysis (EDA). EDA involves summarizing the main characteristics of your data, often using visual methods. We'll cover basic descriptive statistics and data visualization using ggplot2.

# Lab Goals
# -- Calculating basic descriptive statistics
# -- Creating histograms, boxplots, and scatter plots with ggplot2

#DESCRIPTIVE STATISTICS

# Let's start by loading the necessary libraries and data.
library(dplyr)
# You'll need to install ggplot2 if you haven't used it before. It's super useful for making advanced graphs and charts.
install.packages("ggplot2")
library(ggplot2)

# Load the "Lab_3_Data.csv" file (used in last lab) into a data frame called data.
setwd("/Users/emmabacarra/downloads")
data <- read.csv("Lab_3_Data.csv")

# Calculating descriptive statistics helps us understand the distribution and central tendency of our data.
# We'll calculate the mean, median, mode, and range for the "Age" column below.
mean_age <- mean(data$Age, na.rm = TRUE)
median_age <- median(data$Age, na.rm = TRUE)
# Note there isn't a default mode function. This is a bit janky, but works.
mode_age <- as.numeric(names(sort(table(data$Age), decreasing = TRUE)[1]))
range_age <- range(data$Age, na.rm = TRUE)

# And view in the console here.
mean_age
median_age
mode_age
range_age

# TASK 1 ***********************************************************************
# Calculate and display the mean, median, mode, and range for the "Income" column for data rows that have an Age >= 45.
filt_data <- data[data$Age >= 45,]

mean_inc <- mean(filt_data$Income, na.rm = TRUE)
median_inc <- median(filt_data$Income, na.rm = TRUE)
range_inc <- range(filt_data$Income, na.rm = TRUE)
mode_inc <- as.numeric(names(sort(table(filt_data$Income), decreasing = TRUE)[1]))

cat(mean_inc, median_inc, range_inc, mode_inc)


# DATA VISUALIZATION WITH GGPLOT2

# Visualization is a powerful tool for EDA. We'll use the ggplot2 package to create histograms, boxplots, and scatter plots.

# HISTOGRAMS

# Histograms show the distribution of a single variable.
# Let's create a histogram of the "Age" column.
ggplot(data, aes(x = Age)) + geom_histogram(binwidth = 1)


# TASK 2 ***********************************************************************
# Create a histogram of the "Income" column. Use a binwidth of 10000. Use colors of your choice and make sure the axes are labeled and the graph has a title. 
# Creating a histogram of the Income column
ggplot(data, aes(x = Income)) + geom_histogram(binwidth = 10000, fill = "pink", color = "magenta") +
  labs(title = "Distribution of Income", x = "Income", y = "Count")



# BOXPLOTS

# Boxplots show the distribution of a variable and highlight outliers.
# Let's create a boxplot of the "Age" column.
ggplot(data, aes(x = Category, y = Age)) +
  geom_boxplot(fill = "purple", color = "black") +
  labs(title = "Boxplot of Age", y = "Age", x = "Category")

# TASK 3 ***********************************************************************
# Create a boxplot of the "Income" column by the group "Category". Remember colors and labels.
# Creating a boxplot of the Income column by Category
ggplot(data, aes(x = Category, y = Income)) + geom_boxplot(fill = "lavender", color = "purple") +
  labs(title = "Income by Category", x = "Category", y = "Income")



# SCATTER PLOTS
# Scatter plots show the relationship between two variables.
# Let's create a scatter plot of "Age" vs "Income".
ggplot(data, aes(x = Age, y = Income)) +
geom_point(color = "red") +
labs(title = "Scatter Plot of Age vs Income", x = "Age", y = "Income")

# TASK 4 ***********************************************************************
# Create a scatter plot of "Age" vs "Income" colored by "Category". (Hint: ?aes)
# Creating a scatter plot of Age vs Income colored by Category
ggplot(data, aes(x = Age, y = Income, color = Category)) + geom_point() +
  labs (title = "Age vs Income", x = "Age", y = "Income")



# TASK 5 ***********************************************************************
# In your own words, interpret the relationship between Age and Income.
# the average age is 44.98, the average income is $66125.49
# on the scatterplot, the numbers seem to spread all over, so it's not clear if there's a correlation between age and income.
# generally, the assumption however is that the income can increase with more job experience, but as this scatterplot shows,
# that's not always the case. this can be due to several different factors, including environment, economic class, the type
# of job, level of education required, etc. categorizing the data is a good start, but in order to make any inferences, it's
# probably best to get a better context of how/where this data is gathered. also, extreme outliers can throw off the numbers.

# That's it for Lab 4! You've learned how to calculate basic descriptive statistics and create various types of plots using ggplot2. These skills are crucial for understanding and presenting your data.

# Quick Question: Please rate the difficulty of this lab from 1 to 5.

# Be sure to SAVE THIS FILE and upload it to Canvas when you have completed all tasks. Please submit as a .R file.