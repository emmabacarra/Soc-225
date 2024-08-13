# SOC 225: Lab 2
# Emma Bacarra
# 6/26/2024
# 11 TASKS


# Welcome to Lab 2 for SOC 225! In this lab, we'll dive into data importation and cleaning. These skills are foundational for any data analysis project, as real-world data often comes in various formats and requires significant cleaning before it can be analyzed.

# Lab Goals
# --Importing data
# --Understanding data types
# --Handling missing values
# --Basic data cleaning
# --Data transformation

# IMPORTING DATA

# In this section, we'll learn how to import data into R. We'll start with importing CSV files, which are commonly used to store data.

# Download the "Lab_2_Data.csv" file from Canvas and save where you save your Lab files.
# Before we load in the data, we'll want to set your working directory in RStudio. Setting the working directory allows for easy access to files stored within a folder so that you don't need to type a long file path each and every time.

#In the code below, replace the file path in quotes with the folder that you store your Lab files.
setwd("/Users/emmabacarra/downloads")

# To read in a CSV file, we'll use the read.csv() function.

# TASK 1 ***********************************************************************
# Use the read.csv() function to import Lab_2_Data.csv. Remember you can type ?read.csv() for more information about the function. Name the imported dataframe data.
data = read.csv("Lab_2_Data.csv")


# To learn more about the imported data, we can look at the first few rows of our data using the head() function.
# If you didn't name your data "data" in Task 1, set data <- your_dataframe to make the rest of the lab go smoothly.
head(data)

# TASK 2 ***********************************************************************
# What are the column names in your dataset?
colnames(data)
# [1] "Year"                  "Brand"                 "Superbowl.Ads.Link"    "Youtube.Link"          "Funny"                 "Shows.Product.Quickly"
# [7] "Patriotic"             "Celebrity"             "Danger"                "Animals"               "Uses.Sex"              "Length"               
# [13] "Estimated.Cost"        "Youtube.Views"         "Youtube.Likes"         "TV.Viewers"   


# UNDERSTANDING DATA TYPES

# Once we've imported our data, it's important to understand the types of data we have. We can use the str() function to see the structure of our dataset.
str(data)

# TASK 3 ***********************************************************************
# How many columns are in the dataset? What are their data types?

length(colnames(data)) 
# there are 16 columns

sapply(data, class)
#      Year                 Brand    Superbowl.Ads.Link          Youtube.Link                 Funny Shows.Product.Quickly             Patriotic 
# "integer"           "character"           "character"           "character"             "logical"             "logical"             "logical" 
# Celebrity                Danger               Animals              Uses.Sex                Length        Estimated.Cost         Youtube.Views 
# "logical"             "logical"             "logical"             "logical"             "integer"             "numeric"             "integer" 
# Youtube.Likes            TV.Viewers 
#.    "numeric"             "numeric" 

# HANDLING MISSING VALUES

# Real-world data often contains missing values. These need to be handled before any analysis can be performed. In R, missing values should be stored as 'NA', which is essentially a blank in the data set. Let's start by identifying missing values.
# The is.na() function checks for missing values. We can use this in combination with the sum() function to count the number of missing values in each column.
csums = colSums(is.na(data))
csums
#Year                 Brand    Superbowl.Ads.Link          Youtube.Link                 Funny Shows.Product.Quickly             Patriotic 
# 0                     0                     0                     0                     0                     6                     1 
# Celebrity                Danger               Animals              Uses.Sex                Length        Estimated.Cost         Youtube.Views 
# 2                     2                     1                     3                     3                     1                    12 
# Youtube.Likes            TV.Viewers 
# 18                     1 

# TASK 4 ***********************************************************************
# Which columns have missing values and how many are there in each?
csums[csums > 0]
# Shows.Product.Quickly             Patriotic             Celebrity                Danger               Animals              Uses.Sex                Length 
#                     6                     1                     2                     2                     1                     3                     3 
# Estimated.Cost         Youtube.Views         Youtube.Likes            TV.Viewers 
#              1                    12                    18                     1 

# Let's double-check that last task. View(data) and scroll down to Row 249. Notice the missing value? Let's see why that didn't register as an NA.
# We can access specific cells in a dataframe using the notation below. The first number in brackets is the row, the second is the column number. Rows and column numbers both start at 1 in R.
data[248,2]

# When importing a .csv, the function we used only sets values to NA if the cell has 'NA' written in it. We can adjust this so that empty cells are also recorded as 'NA' using the code below.
data <- read.csv("Lab_2_Data.csv", na.strings = c("","NA"))

# TASK 5 ***********************************************************************
# Now what do your NA's look like?
csums = colSums(is.na(data))
csums
# Year                 Brand    Superbowl.Ads.Link          Youtube.Link                 Funny Shows.Product.Quickly             Patriotic 
# 0                     2                     1                    11                     0                     6                     1 
# Celebrity                Danger               Animals              Uses.Sex                Length        Estimated.Cost         Youtube.Views 
# 2                     2                     1                     3                     3                     1                    12 
# Youtube.Likes            TV.Viewers 
# 18                     1 


# One common way to handle missing values is to remove any rows that contain them. We can do this using the na.omit() function.
data_clean <- na.omit(data)

# Let's check the number of rows before and after cleaning.
nrow(data)
nrow(data_clean)

# TASK 6 ***********************************************************************
# How many rows were removed after cleaning?
nrow(data) - nrow(data_clean)
# 42

# BASIC DATA CLEANING

# Besides handling missing values, data often requires other cleaning steps such as removing duplicates or correcting data types.
# Let's start by removing any duplicate rows using the distinct() function from the dplyr package. We'll first need to download the package if you haven't used it before.
install.packages("dplyr")
library(dplyr)
data_clean <- distinct(data_clean)

# Next, let's correct data types. Suppose we want to make the 'Brand' column a factor instead of a character. Remember factors from last Lab? These are character values that are stored as a number. Watch.
data_clean$Brand <- as.factor(data_clean$Brand)
# Note that we can access columns of a dataframe using '$' notation.

# TASK 7 ***********************************************************************
# Verify the data type of the "Brand" column. How many levels are there?
typeof(data_clean$Brand)
# int

unique(data_clean$Brand)
# [1] E-Trade   Budweiser Bud Light Pepsi     NFL       Toyota    Doritos   Coca-Cola Hyundai   Kia      
# Levels: Bud Light Budweiser Coca-Cola Doritos E-Trade Hyundai Kia NFL Pepsi Toyota


# DATA TRANSFORMATION

# Sometimes we need to create new variables or transform existing ones. Let's create a new column that categorizes a numeric variable into different groups.
# Suppose we want to categorize how popular the ad videos were on YouTube. Let's break it down into Low, Medium, High, and Viral categories. We'll use the cut() function. Type ?cut() in the console for more information.
data_clean$Youtube.Popularity <- cut(data_clean$Youtube.Views, breaks = c(0, 5000, 50000, 1000000, 2000000000), labels = c("Low", "Medium", "High", "Viral"))

# TASK 8 ***********************************************************************
# How many ads fall into each popularity group? Hint: Use the table() function to find out.
table(data_clean$Youtube.Popularity)
# Low Medium   High  Viral 
# 45     59     81     22 

# We'll finish up by summarizing our cleaned data using the summary() function to get a sense of the distribution of each variable.
summary(data_clean)

# TASK 9 ***********************************************************************
# Based on the summary output, what is the mean and median of the "TV.Viewers" column?
summary(data_clean$TV.Viewers)[4]
# Mean   :100.48
summary(data_clean$TV.Viewers)[3]
# Median : 98.73


# TASK 10***********************************************************************
# Create a new dataframe that contains only ads who are categorized as "Medium" popularity and have an "Length" greater than the median length. Save this new dataframe as long_and_popular.
long_and_popular = subset(data_clean, Youtube.Popularity == "Medium" & Length > summary(data_clean$TV.Viewers)[4][[1]])


# TASK 11 **********************************************************************
# How many ads meet the criteria for the long_and_popular data frame?
length(long_and_popular)
# 17


# That's it for Lab 2! You've learned how to import data, understand its structure, handle missing values, perform basic cleaning, and transform data. These skills will be crucial as we move on to more advanced data manipulation and analysis techniques.

# Quick Question: Please rate the difficulty of this lab from 1 to 5.
1

# Be sure to SAVE THIS FILE and upload it to Canvas when you have completed all tasks. Please submit as a .R file.