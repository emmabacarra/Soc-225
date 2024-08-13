# SOC 225: Lab 6
# Emma Bacarra
# 7/10/24
# 3+1 TASKS

# Welcome to Lab 6 for SOC 225! Today we will focus on best practices for data visualization and storytelling with data. We will also explore creating dashboards using Shiny in R. This lab will be helpful as an example of how to frame your Final Data Project.

# Lab Goals
# -- Understanding data visualization best practices
# -- Storytelling with data
# -- Designing effective visualizations
# -- Creating dashboards with Shiny

# DATA VISUALIZATION BEST PRACTICES

# Designing effective visualizations is crucial for clearly communicating your data insights. Below are some best practices to follow:

# 1. Know Your Audience: Tailor your visualizations to the knowledge level and interests of your audience.
# 2. Choose the Right Chart Type: Different types of data are best represented by different types of charts (e.g., bar charts for comparisons, line charts for trends).
# 3. Keep It Simple: Avoid clutter by focusing on the most important data and removing unnecessary elements.
# 4. Use Color Wisely: Use color to highlight key data points, but be mindful of colorblindness and ensure your colors have enough contrast.
# 5. Label Clearly: Ensure all axes, data points, and key elements are clearly labeled.
# 6. Tell a Story: Structure your visualizations to guide the viewer through the data, highlighting key insights and trends.

# Let's run through an example. Note which of the elements discussed above are present in the folowing graph:

library(ggplot2)
#We're going to load the MPG dataset from ggplot2, which contains fuel economy data from 1999-2008 for various cars
data(mpg)
mpg<-data.frame(mpg)
#The code to make this plot should look pretty familiar! Remember to use the help navigation panel if you have any questions!
ggplot(mpg, aes(x = displ, y = hwy, color = class)) +
  geom_point() +
  labs(title = "Highway MPG vs. Engine Displacement",
       x = "Engine Displacement (liters)",
       y = "Highway MPG",
       color = "Car Class") +
  theme_minimal()

#  STORYTELLING WITH DATA

# Storytelling with data involves using visualizations to tell a coherent story that highlights the key points of your analysis. Here are some steps to create an effective data story:

# 1. Start with a Question: Frame your analysis around a specific question or hypothesis.
# 2. Provide Context: Give background information to help the audience understand the significance of your data.
# 3. Highlight Key Insights: Use visualizations to highlight the most important findings.
# 4. Conclude with Actionable Insights: End with clear conclusions or recommendations based on your data.

# Let's walk through another simple example:
# Suppose we want to tell a story about fuel efficiency across different car classes. The chart below may help us to cleary convey to our audience the variation in fuel efficiency.

ggplot(mpg, aes(x = class, y = hwy)) +
  geom_boxplot() +
  labs(title = "Fuel Efficiency by Car Class",
       x = "Car Class",
       y = "Highway MPG") +
  theme_minimal() +
  annotate("text", x = 4, y = 40, label = "Pickups have the lowest fuel efficiency", color = "red", size = 3)

# CREATING DASHBOARDS WITH SHINY
# Shiny is an R package that allows you to build interactive web applications. It's a little more fun than working with Plotly like we did on Monday. Here, we will create a simple Shiny app to visualize data interactively.

# Install Shiny if you haven't already
install.packages("shiny")

# Remember to load the Shiny library
library(shiny)

# Here's an example of a simple Shiny app. Don't worry too much about the nomenclature here, but feel free to type ?fluidPage or ?shinyapp for more information about what the variables mean.

# This first section is setting up the user interface for our web app.
ui <- fluidPage(
  titlePanel("MPG Data Dashboard"),
  sidebarLayout(
  sidebarPanel(
    selectInput("variable", "Variable:", choices = c("displ", "hwy", "cty")),
    sliderInput("bins", "Number of bins:", min = 1, max = 50, value = 30)
  ),
  mainPanel(
    plotOutput("distPlot")
  )
)
)

# This section sets up a server function, which tells the web page what to display.
server <- function(input, output) {
  output$distPlot <- renderPlot({
    ggplot(mpg, aes_string(x = input$variable)) +
    geom_histogram(bins = input$bins) +
    labs(title = paste("Histogram of", input$variable),
      x = input$variable,
      y = "Frequency")
  })
}

# This function creates the Shiny app. It will open a pseudo-web browser tab from RStudio. There should be a button at the top of the window that also allows you to open the app in your default browser.
shinyApp(ui = ui, server = server)

# TASKS

# TASK 1 ***********************************************************************
# Create a scatter plot using ggplot2 showing the relationship between engine displacement (displ) and highway MPG (hwy). Use color to differentiate between different car classes (class).
ggplot(mpg, aes(x = displ, y = hwy, color = class)) +
  geom_point() +
  labs(title = "Engine Displacement vs Highway MPG",
       x = "Engine Displacement",
       y = "Highway MPG",
       color = "Car") +
  theme_minimal()

# TASK 2 ***********************************************************************
# Design a visualization to compare the city MPG (cty) across different car classes. Make sure to follow best practices for chart design.
ggplot(mpg, aes(x = class, y = cty, fill = class)) +
  geom_boxplot() +
  labs(title = "city mpg across different car classes",
       x = "Car",
       y = "City MPG") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# TASK 3 ***********************************************************************
# Write a short narrative (3-5 sentences) explaining the insights from your visualizations created in Task 1 and Task 2. Focus on what you can interpret from the data (I promise it's not random this time).
# The scatter plot shows that larger engine displacements have lower MPG on the highway, also roughly
# divided in groups by car type. Generally, the pickups and SUVs are the most fuel inefficient, while 
# the compact and subcompact cars seem to be the most efficient. However, it is fairly mixed, given 
# that different brands can improve the performance of cars based on their designs. The boxplot shows 
# that compact and subcompact cars have the highest MPG in the city, while pickups and SUVs are also
# still the most fuel inefficient. This means that generally, smaller cars are better than the larger
# cars at being fuel efficient because they are smaller.



# TASK 4 ***********************************************************************
# This is a tough one! Grading this one on effort, give it a shot!
# Create a simple Shiny app that allows users to explore the `mpg` dataset. The app should include:
# - Dropdown menus to select variables for the x and y axes of a scatterplot (e.g., displ, hwy, cty)
# - A plot output to display the scatterplot

# If you get stuck: Scroll down to the bottom of the lab file, the updated ui2 for the app is there.
ui2 <- fluidPage(
  titlePanel("MPG Dataset"),
  sidebarLayout(
    sidebarPanel(
      selectInput("xvar", "X-axis:", choices = c("displ", "hwy", "cty")),
      selectInput("yvar", "Y-axis:", choices = c("displ", "hwy", "cty")),
      checkboxGroupInput("classes", "Car Types:", 
                         choices = unique(mpg$class), 
                         selected = unique(mpg$class))
    ),
    mainPanel(
      plotOutput("scatterPlot")
    )
  )
)

server2 <- function(input, output) {
  filteredData <- reactive({
    mpg %>% filter(class %in% input$classes)
  })
  
  output$scatterPlot <- renderPlot({
    ggplot(filteredData(), aes_string(x = input$xvar, y = input$yvar, color = "class")) +
      geom_point() +
      labs(title = paste("Scatter Plot of", input$xvar, "vs", input$yvar),
           x = input$xvar,
           y = input$yvar,
           color = "Car Class") +
      theme_minimal()
  })
}

shinyApp(ui = ui2, server = server2)


# Be sure to SAVE THIS FILE and upload it into Canvas when you have completed all tasks. Please submit as a .R file and a .zip file containing your Shiny app.

# No quick question today! Next week we get into data analysis techniques woohoo!

# Updated ui2 code
# Thanks for giving it a try!
ui2 <- fluidPage(
  titlePanel("MPG Data Dashboard"),
  sidebarLayout(
    sidebarPanel(
      selectInput("xvar", "X-axis variable:", choices = c("displ", "hwy", "cty")),
      selectInput("yvar", "Y-axis variable:", choices = c("displ", "hwy", "cty"))
    ),
    mainPanel(
      plotOutput("scatterPlot")
    )
  )
)