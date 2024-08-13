# SOC 225: Lab 7
# Emma Bacarra
# 7/15/2024
# 5 TASKs

# Lab Goals:
# -Understanding how to combine datasets using dplyr
# -Performing various types of joins (inner, outer, left, right)
# -Dealing with data integrity issues after joins


# SETUP

# Before we start, make sure you have the dplyr package installed and loaded:

install.packages("dplyr")
library(dplyr)

# We will use two datasets for this lab, both of which are available on Canvas. Make sure to download and save both Lab_7_Employees and Lab_7_Departments and move the .csv to your working directory. I set my working directory below.

setwd("/Users/emmabacarra/downloads")

# Load the datasets
employee_data <- read.csv("Lab_7_Employees.csv")
department_data <- read.csv("Lab_7_Departments.csv")

# It's usually a good idea to take a look at new data after importing it to have some idea of what you are working with. Let's do that below.

View(employee_data)
View(department_data)

# INNER JOINS

# In general, joins are ways to link two datasets together by a variable they share. In base R, these are called 'merges'. We'll be going over the join method from dplyr here just because it offers a bit more flexibility, but feel free to type ?merge into the console as well.

# An inner join returns rows that have matching values in both datasets.
# Let's try an inner join
inner_join_result <- inner_join(employee_data, department_data, by = "dept_id")

# TASK 1 ***********************************************************************
# Look at the results of the inner join on the employee_data and department_data datasets and answer the following questions:
  
# How many rows are in the resulting dataset?
nrow(inner_join_result)
# Why have some rows been removed from the result?
# their ids don't match in the datasets (NA values)

# LEFT JOIN

# A left join returns all rows from the left dataset (first one listed) and matching rows from the right dataset (second one listed).

# Another example here of a left join
left_join_result <- left_join(employee_data, department_data, by = "dept_id")

# TASK 2 ***********************************************************************
# Look at the results of the left join on the employee_data and department_data datasets and answer the following questions:
  
# How many rows are in the resulting dataset?
nrow(left_join_result)
# What is the department name for Employee 40?
View(left_join_result)
left_join_result[left_join_result$employee_id == 40, "dept_name"]


# RIGHT JOIN

# A right join returns all rows from the right dataset and matching rows from the left dataset.

# Here's an example of right join
right_join_result <- right_join(employee_data, department_data, by = "dept_id")

# TASK 3 ***********************************************************************
# You know the drill for answering the following questions:
View(right_join_result)
  
# How many rows are in the resulting dataset?
nrow(right_join_result)
# Which employee names are included in the result? Why?
right_join_result$name

    
# FULL JOIN
# A full join returns all rows when there is a match in either left or right dataset.

# Last example I promise of a full join
full_join_result <- full_join(employee_data, department_data, by = "dept_id")

# TASK 4 ***********************************************************************
# Look at the results of the full join on the employee_data and department_data datasets and answer the following questions:
  
# How many rows are in the resulting dataset?
nrow(full_join_result)
# Why is the result the same as the left join for these two datasets?
# because employee_data is the left dataset, so information gets joined accordingly with that one rather than it getting joined with another
  

# DEALING WITH DATA INTEGRITY ISSUES

#Sometimes datasets have missing or inconsistent data. Remember how to solve them?

# TASK 5 ***********************************************************************
# Run another inner join between the employee and department data, but before doing so drop all rows of the employee data that include NA values (Hint: we did this in a previous lab, and there's a second method using the dplyr package instead). 
# How many rows are in the resulting joined dataframe?
na_omit <- inner_join(na.omit(employee_data), department_data, by = "dept_id")
nrow(na_omit)



# That's it for today! Remember to submit the .R file for this Lab to Canvas!