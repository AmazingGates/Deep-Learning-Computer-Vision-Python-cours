import tensorflow as tf # For our Models
import pandas as pd # For reading and processing Data
import seaborn as sns # For Visualization
import keras.layers 
from keras.layers import Normalization, Dense
import matplotlib.pyplot as plt
import keras.losses
from keras.losses import MeanSquaredError, Huber, MeanAbsoluteError


# Here we will be going over error sanctioning.

# We will see how representative this model is of our data set.


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


# So to check this out, we'll notice that for each and every output we have, we are going to compare the 
#actual output with what the model gives us.
# So our model tells us that at this point, we add this x and this y axis.



#|Y
#|___________________/
#|                  /|   Y = mX + c
#|____________.2   / |    
#|     .1      |  /  |     
#|             | /   |
#|_____________|/    | 
#|____________./3    |   
#|            /      |    
#|      .4   /       |  
#|__________/    .5  |      
#|_________/.6       |   
#|        /          |    
#|----.7 /* (2) What | we actually have is this, according to our data set, or the actual selling price.
#|____|_/___________.8
#|----|/* (1)This should mark our selling price according to our model.
#|    /
#|   /
#|  /
#| /  |                                                       X
#/------------------------------------------------------------


# If we look at the other points, we see that our model does quite well, as our model, or the actual price is this,
#See (.6), and if we extrapolate we see our model tells us our price is this (see line right above .6), so the actual
#price and the models price is actual quite similar.

# Now if we take in another point like this (see .2), we see this performs poorly, our model tells us that we should
#be around this selling price, and the actual sellingprice around this (see top line from .2).
# So this tells us that if we want to choose the best possible values for m and c, then we have to choose them
#such that we minimize these differences we have here. 
# That's the difference between Y (actual price), and the models price (Y = mX + c)
# We have to try to minimize these differences.

# By minimizing the differences, we actually sanction in this error.
# So with sanction in the model every time it makes these kinds of errors.

# So with these kinds of large errors we have here (see .8), we're trying to sanction the model.
# And when we our models work perfectly, we have less sanction, like this (see .3)

# The sanction is actually quite simple on what we'll do when we have an error.
# Let's say we have an output with an error, we'll use the formula of Ya(actual price) subtracted by
#Yp(predicted price) and square the equation. 
# The formula should look like this (Ya - Yp)squared

# Then we see clearly that if the Y actual and the Y predicted are both the same, then in that case we will 
#have a zero, because Ya and Yp are the same, so when subtracting a number from it's equal, we get zero.

# This means that we have to give that model zero sanctions for that particular prediction.
# Like in the example of (.3)

# On the other hand, if there is a very big difference between the two, then we continue with the formula
#(Ya - Yp)squared

# Let's say for example that our Ya = 4, and our Yp = 2.
# Our output will be 4, according to our formula.
# (4 - 2)squared = 4

# Anytime we make an error we amplify that error (squrae it).

# Now if we want to get an overall error, we could use what we call the mean square root error function.
# With a mean square error root function what we are basically doing is this equation, (Ya - Yp)squared, 
#but we are repeating this for each and every point, and then we are finding the average of all those errors.

# And as usual, tensorflow already has this built in so all we need to do is just make use of this function
#which has already been built.

# tf.keras.losses.MeanSquaredError - Computes the mean of squares of errors between labels and predictions.

# Here we have our loss function
# tf.keras.losses.MeanSquaredError(
#     reduction-losses utils.ReductionV2.AUTO, name="mean squared error"
#)

# Now let's look at an example of this in practice

Ya = [[0., 1.], [0., 0.]]
Yp = [[1., 1.], [1., 0.]]
# (Using "auto"/"sum_over_batch_size" reduction type.)
mse = tf.keras.losses.MeanSquaredError()
mse(Ya, Yp).numpy()
print(mse(Ya, Yp).numpy()) # Output 0.5

# This is the process we would use to sum up all these squared errors and then divide by the number of elements
#we have.  

# Now that we have imported our MeanSquareError, we will make use of this when we compile our model.

# So we will use model.compile, and we definr our loss. So our loss is a mean square error which we just define 
#MeanSquaredError

# And this does is as we compile inour model, we take into consideration the fact that our error sanctioning
#function is going to be the mean squared error. 

# Now for regression tasks apart from the mean square error, we also have the mean absolute error and
#here, everytime we do the subtraction of the Y true, or Y actual, and the Y predicted, instead of squaring
#this, what we do is we calculate the absolute value.

#   tf.keras.losses.AbsoluteError -
#                                 - Computes the mean of absolute difference between labels and predictions

# So it's quite similar to what we've seen already with a mean square error, the only difference here is we
#have the absolute of the subtraction of the difference between the Ya and the Yp.

# To better understand when to use the mean squared error or the mean absolute error, let's take this example
#we have here.
# With this example, we have the horsepower and the current price.
# Most times the horse power is positively correlated to the current price.
# That is if we increase the horsepower, generally overall we have an increase in the current price.
# And if we reduce the horsepower we have an overall reduction in the current price.



#model.compile(loss= MeanSquaredError()) # Output None
# Notice that when model.compile was run on our csv data, we have a return of None. 
# Note: Since this data is our data preparation file, this is where we are going to run it.

#model.compile(loss= MeanAbsoluteError()) # Output None
# Notice that when model.compile was run on our csv data, we have a return of None. 
# Note: Since this data is our data preparation file, this is where we are going to run it.

#model.compile(loss= Huber(delta=0.2)) # Output None
# Notice that when model.compile was run on our csv data, we have a return of None. 
# Note: Since this data is our data preparation file, this is where we are going to run it.
