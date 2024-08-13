# SOC 225: Lab 10
# Emma Bacarra
# 07/29/2024
# 5 TASKS

# Welcome to Lab 10! Today, we will dive into basic statistical analysis techniques. By the end of this lab, you should be able to conduct t-tests, chi-squared tests, and ANOVA using the t.test(), chisq.test(), and aov() functions in R, and interpret the results.

# Lab Goals
# - Conducting t-tests
# - Performing chi-squared tests
# - Executing ANOVA
# - Interpreting the results of these statistical tests

library(dplyr)

# HYPOTHESIS TESTING

# Hypothesis testing is a fundamental method in statistics for testing assumptions about a population parameter. 
# A hypothesis test evaluates two mutually exclusive statements about a population to determine which statement is best supported by the sample data.

# Conducting a t-test
# Let's start with a simple example using a one-sample t-test. We will use the built-in dataset "mtcars" to test whether the mean mpg (miles per gallon) of cars is equal to 20.

# Load the data
data("mtcars")

# Perform a one-sample t-test
t_test_result <- t.test(mtcars$mpg, mu = 20)

# View the results
t_test_result

# The result will show you the test statistic, degrees of freedom, p-value, and confidence interval. If the p-value is less than the significance level (usually 0.05), we reject the null hypothesis.
# In this case, because the p-value is greater than the "standard" significance level of 0.05, we would be unable to reject the null hypothesis that the mean mpg of cars is not equal to 20.

# TASK 1 ***********************************************************************
# Using the above example as a guide, conduct a one-sample t-test to check if the mean weight (wt) of cars in the mtcars dataset is equal to 3.2. Explain your finding.
t.test(mtcars$wt, mu = 3.2)
# the true mean is not equal to 3.2

# CHI-SQUARED TESTS

# Chi-squared tests are used to examine the association between categorical variables. We will use the built-in "HairEyeColor" dataset to perform a chi-squared test to see if there is an association between hair color and eye color.

# Load the data
data("HairEyeColor")

# Perform a chi-squared test
chi_sq_test_result <- chisq.test(HairEyeColor[, , "Male"])

# View the results
chi_sq_test_result

# The result will show the test statistic, degrees of freedom, and p-value. If the p-value is less than the significance level, we reject the null hypothesis of independence. In this case, we would reject the null hypothesis that there is no association and support the alternative hypothesis that there is an association between eye and hair color in males.

# TASK 2 ***********************************************************************
# Using the above example as a guide, perform a chi-squared test to check the association between hair color and eye color for females in the HairEyeColor dataset. View the HairEyeColor dataset for help accessing the needed table. Explain your finding.
HairEyeColor[, , "Female"] %>% chisq.test()
# the p-value is very small, so there is an association between hair and eye colors

# ANOVA (Analysis of Variance)

# ANOVA is used to compare the means of three or more samples. We will use the "PlantGrowth" dataset to perform a one-way ANOVA to see if there is a significant difference in plant weight across different treatment groups.

# Load the data
data("PlantGrowth")

# Perform a one-way ANOVA
anova_result <- aov(weight ~ group, data = PlantGrowth)

# View the summary of the ANOVA
summary(anova_result)

# The result will show the F-statistic and p-value. If the p-value is less than the significance level, we reject the null hypothesis that the means are equal. Here, we have a p-value slightly less than 0.05, meaning we would reject the null that the treatment has no impact and support the alternative hypothesis that there is a difference between the treatment effects.

# TASK 3 ***********************************************************************
# Using the above example as a guide, perform a one-way ANOVA to test if there is a significant difference in the 'mpg' across different 'cyl' (number of cylinders) groups in the mtcars dataset. Explain your finding.
# Perform the one-way ANOVA
aov(mpg ~ factor(cyl), data = mtcars) %>% summary()
# the means are not equal because the p-value is small


# Two-Sample t-test

# A two-sample t-test compares the means of two independent groups. We will test if the mean mpg differs between automatic and manual transmission cars.

# Perform a two-sample t-test
t_test_two_sample_result <- t.test(mpg ~ am, data = mtcars)

# View the results
t_test_two_sample_result

# The p-value here again suggests that we can reject the null hypothesis that the means are the same and support the alternative hypothesis that they are different.

# TASK 4 ***********************************************************************
# Using the above example as a guide, conduct a two-sample t-test to check if the mean horsepower (hp) differs between cars with different numbers of cylinders (4 vs 6). Explain your findings.
t.test(hp ~ factor(cyl), data = mtcars, subset = (cyl %in% c(4, 6)))
# the mean horsepower doesn't differ between cars with different numbers of cylinders

# Paired t-test

# A paired t-test compares the means of two related groups. They are usually used when you take two measures from the same subject and rely on slightly different assumptions than a normal t test. We will create a quick example dataset to demonstrate this.

# Create example data
before <- c(12, 15, 14, 10, 12)
after <- c(14, 18, 15, 12, 13)

# Perform a paired t-test
t_test_paired_result <- t.test(before, after, paired = TRUE)

# View the results
t_test_paired_result

# We still check the result in a similar way, with the p-value suggesting we reject the null hypothesis that there is no difference between the groups and support the alternative hypothesis that the two groups have different means.

# TASK 5 ***********************************************************************
# Create your own example data for before and after measurements (e.g., test scores before and after a training program) that you expect would show no difference between the means. Perform a paired t-test and explain if the results are what you would expect.
# Create example data
before <- c(12, 15, 14, 10, 12)
after <- c(12, 15, 14, 11, 12)

t.test(before, after, paired = TRUE)
# the p-value is 0.379, so there is a difference between their means

# Conclusion
# In this lab, we explored the basics of statistical analysis. We learned how to conduct t-tests, chi-squared tests, and ANOVA, and interpret the results. These skills are essential for analyzing and interpreting data in various research domains. We'll continue with more analysis skills on Wednesday.

# Be sure to SAVE THIS FILE and upload it into Canvas when you have completed all tasks. Please submit as a .R file.