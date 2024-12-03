# Here we will begin Part 2 in the course
# BUILDING NEURAL NETWORKS WITH TENSORFLOW

# Task Understanding will be our first module.

# Here we will be buiding a linear regression model neural network to predict the current price of a second
#hand car based on the number of years the car has been used, the number of kilometers traveled, its rating,
#condition, economy, top speed, horse power, torque.
# We are going to build models that once given this data, will allow us to predict the current price.

# The first feature we will look at is the horse power as X, and get prices as Y. We will train the model to
#predict a price based on the horse power.
# Our model will make these predictions based on the data provided.

# Instead of entering the price, we want to give the car a score, and based on that score we will determine if the 
#car is exspensive or not.
# For all cars below 8.5, we will consider cheap. 
# For all cars above 8.5, we will consider exspensive.

# From here we may have a different sort of task. One in which we want to say, if based on some input, or inputs,
#the car will fall under one of those categories.

#    X(hp)  |   Y(prices K$)
#   109     |   8
#   100     |   9.3
#   113     |   7.5
#   97      |   8.87
#   206     |   15.16
#   124     |   8.31
#   162     |   12.8
#   150     |   11.50
#   80      |
#   156     |