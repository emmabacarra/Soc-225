# SOC 225: Lab 3
# Emma Bacarra
# 7/1/2024
# 7 TASKS

# Welcome to Lab 3 for SOC 225! In this lab, we'll focus more on data transformation and manipulation using the dplyr package. These skills are essential for preparing your data for analysis.

# Lab Goals
# -- Data manipulation using dplyr
# -- Filtering data
# -- Selecting specific columns
# -- Creating new variables
# -- Summarizing data
# -- Using the pipe operator %>%

# DATA TRANSFORMATION AND MANIPULATION

# The dplyr package provides a set of functions that are designed to make data manipulation tasks easier and more intuitive. We installed this package last week in Lab 2. Let's start by loading the dplyr package.
library(dplyr)

# TASK 1 ***********************************************************************
# Set your working directory to where Lab_3_Data.csv is stored and load the data file into a data frame called data.
setwd("/Users/emmabacarra/downloads")

# FILTERING DATA

# Filtering allows us to select rows that meet certain criteria. The filter() function is used for this purpose.
# For example, to filter rows where Age is greater than 30:
data_filtered <- filter(data, Age > 30)

# TASK 2 ***********************************************************************
# How many rows have an Age greater than 30? Use the nrow() function to find out.
data = read.csv("Lab_3_Data.csv")
nrow(filter(data, Age > 30)) #119


# SELECTING COLUMNS

# Sometimes we only need specific columns from our data. The select() function allows us to choose which columns to keep.
# For example, to select only the Age and Income columns:
data_selected <- select(data, Age, Income)

# TASK 3 ***********************************************************************
# What happens if you try to select the columns 'Age' and 'City' from the data?
select(data_selected, Age, City)
# Error in `select()`:
#   ! Can't select columns that don't exist.
# ✖ Column `City` doesn't exist.
# Run `rlang::last_trace()` to see where the error occurred.

# CREATING NEW VARIABLES

# We can create new variables using the mutate() function. For example, let's create a new variable called Income_per_Year that represents monthly income multiplied by 12.
data_mutated <- mutate(data, Income_per_Year = Income * 12)

# TASK 4 ***********************************************************************
# Create a new variable called Age_in_Decades that represents Age divided by 10. What are the first 5 values of this new variable?
Age_in_Decades <- mutate(data, Age_Dec = Age / 10)
head(Age_in_Decades, 5)


# SUMMARIZING DATA

# Summarizing data allows us to get aggregate statistics. The summarize() function helps us calculate summary statistics.
# For example, to calculate the mean and median Age:
summary_stats <- summarize(data, mean_age = mean(Age, na.rm = TRUE), median_age = median(Age, na.rm = TRUE))
# The 'na.rm=TRUE' argument allows the calculation to proceed even if there are missing values in the selected data. See ?summarize for more info.

# TASK 5 ***********************************************************************
# Calculate the mean and median Income. What are these values?
summarize(data, mean_income = mean(Income, na.rm = TRUE), median_income = median(Income, na.rm = TRUE))


# TASK 6 ***********************************************************************
# What is the maximum income for rows were Age > 65? (Hint: type ?summarize for more info)
summarize(data, max_inc_65p = max(filter(data, Age > 65)$Income, na.rm = TRUE))


# USING THE PIPE OPERATOR %>%

# The pipe operator %>% allows us to chain multiple functions together in a readable manner. For example, we can filter, select, and mutate in a single step:
data_piped <- data %>%
filter(Age > 30) %>%
select(Age, Income, Category) %>%
mutate(Income_per_Year = Income * 12)
#Be sure to view data_piped to see what happened
View(data_piped)

# TASK 7 ***********************************************************************
# This one may be tricky if you're newer to R. Type ?ifelse in the console if you're stuck!
# Use the pipe operator to filter rows where Income is greater than 50000, select the columns "Age" and "Income", and create a new variable "Income_Category" that categorizes Income into "High" if greater than 70000 and "Medium" otherwise. What are the last 5 rows of the resulting data frame?
pip <- data %>%
  filter(Income > 50000) %>%
  select(Age, Income) %>%
  mutate(Income_Category = ifelse(Income > 70000, "High", "Medium"))

tail(pip, 5)


# That's it for Lab 3! You've learned how to manipulate and transform data using dplyr. These skills will be essential for preparing data for analysis in your final project.

# You know the drill: Please rate the difficulty of this lab from 1 to 5.
# 1

# Be sure to SAVE THIS FILE and upload it to Canvas when you have completed all tasks. Please submit as a .R file.