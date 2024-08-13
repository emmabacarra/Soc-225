# SOC 225: Lab 1
# Emma Bacarra
# 6/24/2024


# Welcome to the first Lab for SOC 225! These assignments will run through the basics of programming for data science, focusing on bringing in data, cleaning and preparing it for analysis, making exploratory tables, and some more advanced topics using data from the Internet


# Today, we're going to start at square 1; you should have R and RStudio installed on your computer. If you don't have these programs ready, follow the series of links in the Week1 Lecture Slides on Canvas.

# Lab Goals
# --Basics of R
# --Creating objects
# --Scripts
# --Libraries

# BASICS OF R

# First, note that this Lab takes the form of a .R file. These files are the basic file type used in writing programming files in R; think of them like a .docx for Microsoft Word or a .xlsx for Excel. R is the programming language that is used within R files. Up to this point of this document, the text has not been executable lines of code but rather comments. Comments are used to keep track of what lines of code do, leave notes or messages of what work needs to be continued, or describe how to complete a Lab assignment. Any line that includes a # will be a comment after that character.

# TASK 1 *****
# Make this line a comment.

# Besides just typing '#', you can also make a single line or a block of lines a comment by selecting the line and pressing 'Ctrl'+'Shift'+'C'.

# TASK 2 *****
# Make these lines

# comments please

# R is a programming language. If you haven't used a programming language before, think of it as a way to communicate with the computer in a way simpler than English, but more complex than 1's and 0's. By telling the computer what to do, we can get it to perform tasks, like data science. For starters, we'll ask it to do something very simple, an addition problem.

# To execute the line below, click your cursor on the line and hit the run button in the top right of this window, or press 'Ctrl'+'Enter'.
5+13

# Nice! R has more advanced calculations as well. A function is a set of lines of code with a reference name that the language has built into it (or that you write). To call a function, we write the reference name, and they usually take a value or two to help perform a task. Run and try to guess what some of the below functions mean.
sqrt(36)
max(12, 15, 3, -4, 22, 1, 2, 2)
abs(-3.2)
trunc(3.14159)
exp(10)

# If you see a function() you don't recognize, you can use '?' to pull up the help file that explains it in a different window. Try running the line below.
?exp

# TASK 3 *****
# Write a line of code that adds fifteen and seven and divides that total by the square root of 64

# R can act like a more advanced calculator too. Remember graphing calculators? R has you covered. These lines of code should open a graph in the same window as the help command.
curve((x))
curve((x^2))

# CREATING OBJECTS

# You learned about functions in the last section. R runs on functions, but to make them work we need to supply them with data. There are 4 basic data types in R: Characters, Logical, Numeric, and Factors.

# Characters are text, like letter, words, and sentences.
"This is a character"
"So is this"
"a"

# Logical values are either true or false. These are useful in creating and testing functions, but we can access them in a few easy ways
is.character("Is this a character?")
is.character(17)
15<2

# Numerics we've used a lot already: they're numbers!
log(17)
4*2

# Factors are unique in that they look like numbers, but often represent characters. We'll dive deeper into them when we look at importing and using data sets for analysis.

# TASK 4 ******
# Test what happens when you add two character values together. What happens? Test and explain below.
str1='Hello'
str2='world'
str1+str2
# console returns: "Error in str1 + str2 : non-numeric argument to binary operator" when running line 77 because you cant add strings together.
# however, if you wanted to print them out together, you would just concatenate them instead:
cat(str1, str2)

# So we know functions and characters, and we'll now add objects to the list. Objects allow you to store values of a variety of types and recall them for future use. At the simplest level, we can set each of our data types as objects.
a<-9
b<-"My second object"
c=is.numeric(15)

# Note that you can assign an object using either '<-' or '='. Also note 

# Objects can be much more complex. We can also create vectors, or a set of objects, of the same types. We create vectors using the notation 'c()'
vector1<-c(2,3,2,5,7,2,1)

# TASK 5 *****
#Explain what happens when you run the code below. Look closely at the saved value type.
vector2<-c(1,"12",5,32,"test")
typeof(vector1)
typeof(vector2)
# since the elements of vector2 are different types, the integers will be saved as strings

# Values saved as objects may or may not save when you close and reopen RStudio. You can also remove all objects from your environment by clicking the broom icon above the values or running the line of code below
rm(list=ls())

# A vector can be thought of as a line of values. What happens if we align multiple vectors horizontally or vertically? We get another object type called a data frame! These work almost like spreadsheets, and are the basic data storage unit in R. We'll talk much more about them on Wednesday, but for now we can make one below.
vector1<-c(1,2,3,4,5,6,7,8)
vector2<-c(11,12,13,14,15,16,17,18)
df1<-data.frame(vector1, vector2)

# Note that the data frame is shown differently in the Environment than vectors and single datum variables. To see what is looks like, click the name of the data frame or use the following command
View(df1)

# TASK 6 *****
# View df1. What values are in the second-to-last row?
# 7 and 17

# SCRIPTS

# Up till now we've been running single lines at a time. You'll note that this .R file is now more than 100 lines long. Data science analysis files are usually a series of commands that refer back to variables made earlier and run through complex functions and loops before generating output. We'll get there in the next month and a half, but for now try running the series of lines below. You can run multiple lines of code at once by highlighting it and hitting the Run button or pressing 'Ctrl'+'Enter'
test<-15
symptom<-4.3
blitz<-TRUE

while (test>12) {
  
  if(blitz==TRUE){
    
    print("Still good")
    
  }
  symptom<-symptom+5
  test=test-1
  
}

# TASK 7 *****
# What value is symptom after running the code above?
# 9.3

# LIBRARIES

# Today we've spent a lot of time running functions and testing code in R. The functions we've used today are all a part of the base R language; kinda like the Duolingo version of learning a new language, they're the parts you need to know but there are fancier tools out there. Sometimes, we'll need to use functions that are not in the base R language, and we download additional libraries to use those functions if someone has made them before. 

# Remember the concept of Tidy Data from lecture today? That idea originated here on R, and there is a whole suite of package meant to make data cleaning and manipulation easier. Let's download and load it up before we go.
install.packages("tidyverse")

# The code above downloads the tidyverse package, but we still need to load it into our environment.
library(tidyverse)

# We can see what it does by clicking the packages tab to the right and scrolling to it, or by typing the following:
help(tidyverse)

# TASK 8 *****
# Who is the Maintainer of the tidyverse package?
# hadley wickham

# That's it for today! Thanks for bearing with me as we get everyone up to speed on basic coding concepts. Next time we'll begin data loading and start to figure out how to get information ready for analysis.

# Quick Question: Please rate the difficulty of this lab from 1 to 5
1 # (i have a strong background in python and r already)

# Be sure to SAVE THIS FILE and upload it into Canvas when you have completed all 8 Tasks. Please submit as a .R file.