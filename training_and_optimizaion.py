import tensorflow as tf # For our Models
import pandas as pd # For reading and processing Data
import seaborn as sns # For Visualization
import keras.layers 
from keras.layers import Normalization, Dense
import matplotlib.pyplot as plt
import keras.losses
from keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError


# Here we will be going over Training and Optimization.

# Recall that our model was a linear function, which we were trying to fit such that m and c are peaked
#so that the errors are minimized.

# So looking at this we could have another line drawn.
# We could have this other line drawn and then have a line that crosses that new line to connect to our 
#Y = mx + c equation.

# Essentilly, we could have an infinite number of lines that we generate.

# So in order to get this m and c, we use the method commonly used, which is the stochastic gradient descent.

# Now let's understand how this works.

# We'll just write out this formula we have. We have a weight, which has to be updated, we recall that initially
#our weights are random.

# So we randomly initialize our weights. This means that if we randomly initialize m and c, and then let's
#suppose we have m equals 0.
# So we have m equals 0, and then let's say that we have c equals 1.
# Then in that case we would have something like this line here. See graph below for m = 0,  c = 1 line

# And it's clear that this line isn't very representative of our data we have, so what we could do is update
#our m and c such that they take up values that permit us to adapt to this data set which will be given.

# The way that is done, as we've said already is with SGD (Stochastic Gradient Descent) algorithm, and this is
#how it's done.

# We have a weight and we have a weight previous (W = Wp), or in the case where we are the at the first step,
#we have the initialized weights minus a learning rate (W = Wp - LR), the rate at which we are learning 
#this data which weve been given, times the derivative (W = Wp - LR)2.

# Now in case we have no background in calculus, we shouldn't have any worries because tensorflow takes care
#of all that for us.

# So we have the LR times the derivative of the loss function, with respect to that weight (W = Wp - LR)2L/2Wp

# So with that said, if initially we have 0, and then here we have C, let's say C is 1.
# Now what happens is for each and every weight, we're going to take this (W = Wp - LR)2L/2Wp, so we want to get 
#the new values for the weights or for m.
# So we have m, it's going to be equal to it's initial value of 0.
# So the previous value is 0 minus..., (now let's pick a random learning rate)
# Learning rates are generally picked in the order 0.001, 0.1, 0.01, 0.0001, and could continue to 1 times 10 to the
#negative 6, just for example.

# So here we're randomly picking 0.1. So that's our learning rate, times the derivative of the loss.
# What's the loss? 
# We have already seen 3 loss functions in our error sanctioning module.

# So here we have in 0.1 times the rate at which the loss changes with respect to that particular weight.
# So for (m) we have (m = 0 - 0.1 x 2L/2m)
# And the same is done for (c)
# So for (c) we have (c = 1 - 0.1 x 2L/2c)

# We have now updated m and c and if we computed the error we would have Y actual (Ya) minus the new values
#for m and c, that will equal out to a new error (Ya - (m + c) = Enew).

# And so from here we expect that as we keep training, we want our loss to keep dropping to a value of 0.



#|Y      
#|             /     /    /            /
#|            /     /   Y/ = mX + c   /
#|           / .2  /  / /            /
#|     .1   /     /  / /            /
#|         /     /  / /            /
#|        /     /  / /            /
#|       /    ./3 / /            /
#|      /     /  / /            /
#|     / .4  /  / /            /
#|    /     /  / /.5          /
#|   /     /.6/ /            /
#|  /     /  / /            /
#| /   .7/  / /            /
#|/     /  / /         .8 /
#|-----/- /-/----------- /------- m = 0, c = 1 line
#|    /  / /            /
#| /-/- / /            /
#|/ /    /            /
#/ /    /            /                                         X
#/------------------------------------------------------------

# So as a recap, we see that we have our inputs, and we have our outputs.
# We have m and c, that's our model parameters.
# We pass our inputs using our initialized m and c. We get outputs. We compute a loss. That is the difference
#between what we are actually supposed to get, and what our model predicts.
# We could use a square or an absolute value function. We get this loss (Yp).
# And based on this loss we modify these values and then we repeat the same process on our training converges.
# Our training Converges simply means we've attained a point where our loss doesn't increase anymore. So if we
#have training like this we could get to a point that we keep on trianing.
# That is we keep on repeating the gradient descent step which we've seen already, where we update by doing
#W equals W minus a learning rate times 2L over DW (W = W - 0.1 x 2L/DW). Note 0.1 is just a random learning
#rate we selected for this example.

# So we repeat this step which we've seen before, and if our loss doesn't change much, or at all, then our model
#has converged and we could stop training at that point.



# X----------> Model (m,c are our models parameters) ---------> Y - Yp
#                              |                                   |
#                              |___________________________________| Repeating the same process on our model


# As usual, Tensorflow does all this for all the hard work for us.
# And so here, we have a model.fit. 
# We pass in as our inputs (X,y, ###epochs = 100, Verbose = 1), then we specify the number of ###epochs.
# Let's say 100. 
# Then we pass in our Verbose, which equals 1

# Now we understand what this X and Y means. It's Basically our data set.

# Now the ###epoch, or number of ###epochs here is specified. This is the number of times we are going to update
#our weights.
# So the number of times we are going to go through the gradient descent step.

# And for thje verbose, which has to do with outputs from trainiing step.

# Note that we will run this model in data preparation file where our csv is.

# model.fit(x,y, ###epochs = 100, verbose = 1) # Output 
##Epoch 1/100
##32/32 [==============================] - 1s 2ms/step - loss: 308520.1562
##Epoch 2/100
##32/32 [==============================] - 0s 2ms/step - loss: 308520.1875
##Epoch 3/100
##32/32 [==============================] - 0s 2ms/step - loss: 308520.1250
##Epoch 4/100
##32/32 [==============================] - 0s 3ms/step - loss: 308520.0938
##Epoch 5/100
##32/32 [==============================] - 0s 3ms/step - loss: 308520.0938
##Epoch 6/100
##32/32 [==============================] - 0s 2ms/step - loss: 308520.0625
##Epoch 7/100
##32/32 [==============================] - 0s 2ms/step - loss: 308520.0312
##Epoch 8/100
##32/32 [==============================] - 0s 3ms/step - loss: 308520.0625
##Epoch 9/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.9688
##Epoch 10/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.9688
##Epoch 11/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.8750
##Epoch 12/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.8750
##Epoch 13/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.8438
##Epoch 14/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.8125
##Epoch 15/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.7812
##Epoch 16/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.7188
##Epoch 17/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.6875
##Epoch 18/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.6562
##Epoch 19/100
##32/32 [==============================] - 0s 5ms/step - loss: 308519.5938
##Epoch 20/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.5625
##Epoch 21/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.5625
##Epoch 22/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.5625
##Epoch 23/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.5000
##Epoch 24/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.5000
##Epoch 25/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.4688
##Epoch 26/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.4375
##Epoch 27/100
##32/32 [==============================] - 0s 5ms/step - loss: 308519.4062
##Epoch 28/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.3125
##Epoch 29/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.2812
##Epoch 30/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.2812
##Epoch 31/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.2812
##Epoch 32/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.2500
##Epoch 33/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.1250
##Epoch 34/100
##32/32 [==============================] - 0s 4ms/step - loss: 308519.0938
##Epoch 35/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.0938
##Epoch 36/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.0938
##Epoch 37/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.0312
##Epoch 38/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.0625
##Epoch 39/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.0000
##Epoch 40/100
##32/32 [==============================] - 0s 3ms/step - loss: 308519.0000
##Epoch 41/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.9375
##Epoch 42/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.9375
##Epoch 43/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.8750
##Epoch 44/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.8438
##Epoch 45/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.8438
##Epoch 46/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.7188
##Epoch 47/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.7500
##Epoch 48/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.7188
##Epoch 49/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.6875
##Epoch 50/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.6562
##Epoch 51/100
##32/32 [==============================] - 0s 2ms/step - loss: 308518.6250
##Epoch 52/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.5938
##Epoch 53/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.5625
##Epoch 54/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.5000
##Epoch 55/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.5000
##Epoch 56/100
##32/32 [==============================] - 0s 3ms/step - loss: 308518.4688
##Epoch 57/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.4688
##Epoch 58/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.3750
##Epoch 59/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.3750
##Epoch 60/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.3438
##Epoch 61/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.2812
##Epoch 62/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.2812
##Epoch 63/100
##32/32 [==============================] - 0s 5ms/step - loss: 308518.2500
##Epoch 64/100
##32/32 [==============================] - 0s 4ms/step - loss: 308518.2500
##Epoch 65/100
##32/32 [==============================] - 0s 6ms/step - loss: 308518.1875
##Epoch 66/100
##32/32 [==============================] - 0s 6ms/step - loss: 308518.1250
##Epoch 67/100
##32/32 [==============================] - 0s 10ms/step - loss: 308518.1250
##Epoch 68/100
##32/32 [==============================] - 0s 7ms/step - loss: 308518.0625
##Epoch 69/100
##32/32 [==============================] - 0s 8ms/step - loss: 308518.0938
##Epoch 70/100
##32/32 [==============================] - 0s 6ms/step - loss: 308517.9688
##Epoch 71/100
##32/32 [==============================] - 0s 10ms/step - loss: 308518.0312
##Epoch 72/100
##32/32 [==============================] - 1s 15ms/step - loss: 308517.9375
##Epoch 73/100
##32/32 [==============================] - 0s 4ms/step - loss: 308517.8750
##Epoch 74/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.8750
##Epoch 75/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.8125
##Epoch 76/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.7812
##Epoch 77/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.7812
##Epoch 78/100
##32/32 [==============================] - 0s 4ms/step - loss: 308517.6875
##Epoch 79/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.7188
##Epoch 80/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.6875
##Epoch 81/100
##32/32 [==============================] - 0s 4ms/step - loss: 308517.6562
##Epoch 82/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.5938
##Epoch 83/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.5625
##Epoch 84/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.5625
##Epoch 85/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.5625
##Epoch 86/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.5000
##Epoch 87/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.5000
##Epoch 88/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.4062
##Epoch 89/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.4375
##Epoch 90/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.3438
##Epoch 91/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.3750
##Epoch 92/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.2812
##Epoch 93/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.2812
##Epoch 94/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.2188
##Epoch 95/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.2188
##Epoch 96/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.2188
##Epoch 97/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.1250
##Epoch 98/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.1562
##Epoch 99/100
##32/32 [==============================] - 0s 4ms/step - loss: 308517.0625
##Epoch 100/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.0312
##Epoch 1/100
##32/32 [==============================] - 0s 3ms/step - loss: 308517.0938
##Epoch 2/100
##32/32 [==============================] - 0s 4ms/step - loss: 308516.9688
##Epoch 3/100
##32/32 [==============================] - 0s 4ms/step - loss: 308516.9375
##Epoch 4/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.9062
##Epoch 5/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.8750
##Epoch 6/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.8750
##Epoch 7/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.8438
##Epoch 8/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.8438
##Epoch 9/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.7500
##Epoch 10/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.7812
##Epoch 11/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.6875
##Epoch 12/100
##32/32 [==============================] - 0s 3ms/step - loss: 308516.6875
##Epoch 13/100
##32/32 [==============================] - 0s 4ms/step - loss: 308516.6250
##Epoch 14/100
##32/32 [==============================] - 0s 5ms/step - loss: 308516.5625
##Epoch 15/100
##32/32 [==============================] - 0s 5ms/step - loss: 308516.5312
##Epoch 16/100
##32/32 [==============================] - 0s 7ms/step - loss: 308516.5312
##Epoch 17/100
##32/32 [==============================] - 0s 6ms/step - loss: 308516.4375
##Epoch 18/100
##32/32 [==============================] - 0s 4ms/step - loss: 308516.5000
##Epoch 19/100
##32/32 [==============================] - 0s 4ms/step - loss: 308516.4688
##Epoch 20/100
##32/32 [==============================] - 0s 5ms/step - loss: 308516.4062
##Epoch 21/100
##32/32 [==============================] - 0s 5ms/step - loss: 308516.4062
##Epoch 22/100
##32/32 [==============================] - 0s 7ms/step - loss: 308516.3438
##Epoch 23/100
##32/32 [==============================] - 0s 14ms/step - loss: 308516.3125
##Epoch 24/100
##32/32 [==============================] - 0s 7ms/step - loss: 308516.3125
##Epoch 25/100
##32/32 [==============================] - 0s 8ms/step - loss: 308516.2188
##Epoch 26/100
##32/32 [==============================] - 0s 7ms/step - loss: 308516.2188
##Epoch 27/100
##32/32 [==============================] - 0s 6ms/step - loss: 308516.1875
##Epoch 28/100
##32/32 [==============================] - 0s 5ms/step - loss: 308516.1562
##Epoch 29/100
##32/32 [==============================] - 0s 5ms/step - loss: 308516.1250
##Epoch 30/100
##32/32 [==============================] - 0s 6ms/step - loss: 308516.1250
##Epoch 31/100
##32/32 [==============================] - 0s 11ms/step - loss: 308516.0625
##Epoch 32/100
##32/32 [==============================] - 0s 10ms/step - loss: 308516.0312
##Epoch 33/100
##32/32 [==============================] - 0s 4ms/step - loss: 308515.9688
##Epoch 34/100
##32/32 [==============================] - 0s 4ms/step - loss: 308515.9688
##Epoch 35/100
##32/32 [==============================] - 0s 4ms/step - loss: 308515.9375
##Epoch 36/100
##32/32 [==============================] - 0s 4ms/step - loss: 308515.9062
##Epoch 37/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.9062
##Epoch 38/100
##32/32 [==============================] - 0s 8ms/step - loss: 308515.8438
##Epoch 39/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.7812
##Epoch 40/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.8125
##Epoch 41/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.7188
##Epoch 42/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.7188
##Epoch 43/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.6875
##Epoch 44/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.6250
##Epoch 45/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.5625
##Epoch 46/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.5312
##Epoch 47/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.5312
##Epoch 48/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.5938
##Epoch 49/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.4688
##Epoch 50/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.5000
##Epoch 51/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.4062
##Epoch 52/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.4062
##Epoch 53/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.4062
##Epoch 54/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.3125
##Epoch 55/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.2500
##Epoch 56/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.2188
##Epoch 57/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.1875
##Epoch 58/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.2188
##Epoch 59/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.1562
##Epoch 60/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.0938
##Epoch 61/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.1250
##Epoch 62/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.0938
##Epoch 63/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.0312
##Epoch 64/100
##32/32 [==============================] - 0s 3ms/step - loss: 308515.0000
##Epoch 65/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.9688
##Epoch 66/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.9062
##Epoch 67/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.9062
##Epoch 68/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.8750
##Epoch 69/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.8438
##Epoch 70/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.8438
##Epoch 71/100
##32/32 [==============================] - 0s 7ms/step - loss: 308514.7188
##Epoch 72/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.7188
##Epoch 73/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.7188
##Epoch 74/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.6562
##Epoch 75/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.6250
##Epoch 76/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.6250
##Epoch 77/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.5625
##Epoch 78/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.5312
##Epoch 79/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.5000
##Epoch 80/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.5000
##Epoch 81/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.4688
##Epoch 82/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.3750
##Epoch 83/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.4375
##Epoch 84/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.3750
##Epoch 85/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.3750
##Epoch 86/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.3125
##Epoch 87/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.2500
##Epoch 88/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.2500
##Epoch 89/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.2188
##Epoch 90/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.1875
##Epoch 91/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.1250
##Epoch 92/100
##32/32 [==============================] - 0s 3ms/step - loss: 308514.1250
##Epoch 93/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.0625
##Epoch 94/100
##32/32 [==============================] - 0s 4ms/step - loss: 308514.0312
##Epoch 95/100
##32/32 [==============================] - 0s 3ms/step - loss: 308513.9688
##Epoch 96/100
##32/32 [==============================] - 0s 3ms/step - loss: 308513.9375
##Epoch 97/100
##32/32 [==============================] - 0s 2ms/step - loss: 308513.9062
##Epoch 98/100
##32/32 [==============================] - 0s 3ms/step - loss: 308513.9375
##Epoch 99/100
##32/32 [==============================] - 0s 3ms/step - loss: 308513.8125
##Epoch 100/100
##32/32 [==============================] - 0s 2ms/step - loss: 308513.8750
#<keras.callbacks.History object at 0x0000022A09D92CD0>

# Notice that because our variable equals 1, we are able to get these kinds of outputs.

# So here in real time we are able to see the values of our loss.

# As we go from one #epoch to another, we are able to get the mean absolute errors, which is an
#absolute value of our model predictions, or the absolute value of the difference between our model
#predictions and the actual current prices.


# Now we will look into the tf.keras.Model function

#   tf.keras.Model -
#                  - Model groups layers into an object with training and inference features.

# When we get into the tf.keras.Model documentation, and we go to compile, we see we could have this
# compile(
#   optimizer = "rmsprop", loss = None, metrics = None, loss_weights = None,
#   weighted_metrics = None, run_eagerly = None, steps_per_execution = None, **kwargs
#)

# Notice in our definition of compile, we see how by default our optimizer is rms prop.
# Now note that this optimizer or those different optimizers are essentially variants of the stochastic
#gradient descent algorithm.

# So next we will look into optimizers.

#   tf.keras.optimizers.SGD -
#                           - Gradient descent (with momentum) optimizer

# As we can see, we have added Delta out of grad item, and we can see we have SGD.
# So this is the SGD we've already seen.
# So we can have this 
# tf.keras.optimizer.SGD(
#   learning_rate = 0.1, momentum = 0.0, nesterov = False, name = "SGD", **kwargs
#)

# SGD will specify the learning rate.
# We have the momentum. We could increase this parameter so that we could speed up training.
# We could also specify if we have a nester of type momentum. 
# So when we see nester of True, then we're having a nester of type momentum.

# Of all these optimizers, the most commonly used is the Adam optimizer.
# We'll see that many practitioners generally use this optimizer by default
# tf.keras.optimizers.Adam(
#   learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False 
#)

# The learning rate is 0.001.
# beta_1 is 0.9
# beta_2 is 0.999
# epsilon is 1e - 07 or 1 times 10 to the negative 7
# And the Ams grad time is set to False.


# To better understand the learning rate, let's take a look at this example.

# Here we have loss versus a weight like m or the c.



#\                | (L)            /
# \               |               /
#  \              |              /
#   \             |             /
#    \            |            /
#     \           |           /
#      \          |          /
#       \         |         /
#        \        |        /|
#         \       |       / |
#          \      |      /  |
#           \     |     /|  |
#            \    |    / |  |
#             \   |   /  |  |
#              \  |  /   |  |
#               \ | /----|--- Slope (point B)
#                \|/     |
#                 |      |
#                 |      |
#----------------------------------------------- (W)
#                 |      | Point A
#                 |     (Wi)

#       W = W - LR * 2L/2W

# Now we have the derivative of the loss with respect to the weight.

# So let's consider that this derivative is positive, we'll take a point like this (see diagram above (point A))
# We have the weight here.

# In this particular weight let's say Wi would have picked out this point and we have this derivative of
#the loss with respect to the weight or the partial derivative.
# Then let's say we have the slope (see diagram above (point b))
# The slope is positive 

# To update this weight from Wi to a new weight, we apply this formula.
# We have Wi minus a learning rate times the positive slope.

# Now if this learning rate is too small, let's say 10 to the power of negative 10, it's clear 
#that we are not going to have a great differnce between the W we obtain after this update since this going 
#to be multiplied by this positive value and W minus a very small number will give us a value very close
#to W so there's not going to be a very great change after this update.

# Now if LR is too large, let's say we take a learning rate of 10 to the power of 3.
# In that case we have W minus a thousand times this positive value.
# Then the change is going to be too brutal, and instead of going slowly, towards the point we want, we
#will end up skipping the point we want.

# And if we keep doing this, we will actually get away from the point that we want.

# Also, many times by default or from many experiments, starting out with 0.001 is going to be a good
#idea but note that we could always change this.

# So we could feel free to change this and make it a larger value.

# Recall that if we take a larger value, your training is going to be faster but risk divertgent.
# That is risk not leaving us towards that minimal loss whereas if we make it too small, then
#we would converge, but our training is going to take too much time. 
# So this value is kind of a great one that is very commonly used now particularly for this item optimizer.
# We have parameters beta_1 and betta_2 
# Now our beta_1 max and beta_2 max are all 1.
# So taking a value like 0.9 and 0.99 means we're taking higher values of beta_1 and beta_2.
# If We want to sped up training, we want to increase the values for beta_1 and beta_2.
# By default this value is set to 0.9 and 0.99.

# Now when doing or carrying out computations respect to the item optimizer, we're trying to avoid
#dividing by zeros.
# So we have this epsilon parameter here which is by default 1 times 10 to the negative 7.

# Now if one of the AMS grad variants of the item optimizer, we could just set this to true.

# Next we are going to import Adam Optimizer with the rest of our imports 

# Now we are going to make use of that.

# So actually in this compile, we have optimizer equals Adam and that's it.

# So we'll define our optimizer and we'll define the loss

# So let's run this again.

# Note: We will be running this compile function in our data preparation file.

#model.compile(optimizer=Adam(),
#               loss= MeanAbsoluteError()) # Output

# Epoch 1/100
#32/32 [==============================] - 1s 4ms/step - loss: 308520.1875
#Epoch 2/100
#32/32 [==============================] - 0s 3ms/step - loss: 308520.2500
#Epoch 3/100
#32/32 [==============================] - 0s 5ms/step - loss: 308520.1562
#Epoch 4/100
#32/32 [==============================] - 0s 3ms/step - loss: 308520.1250
#Epoch 5/100
#32/32 [==============================] - 0s 7ms/step - loss: 308520.1250
#Epoch 6/100
#32/32 [==============================] - 0s 5ms/step - loss: 308520.0625
#Epoch 7/100
#32/32 [==============================] - 0s 4ms/step - loss: 308520.0312
#Epoch 8/100
#32/32 [==============================] - 0s 6ms/step - loss: 308520.0000
#Epoch 9/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.9688
#Epoch 10/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.9375
#Epoch 11/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.9375
#Epoch 12/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.9062
#Epoch 13/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.8750
#Epoch 14/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.8125
#Epoch 15/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.7812
#Epoch 16/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.7188
#Epoch 17/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.7500
#Epoch 18/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.6875
#Epoch 19/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.6562
#Epoch 20/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.5938
#Epoch 21/100
#32/32 [==============================] - 0s 6ms/step - loss: 308519.5938
#Epoch 22/100
#32/32 [==============================] - 0s 5ms/step - loss: 308519.5625
#Epoch 23/100
#32/32 [==============================] - 0s 4ms/step - loss: 308519.5625
#Epoch 24/100
#32/32 [==============================] - 0s 4ms/step - loss: 308519.5312
#Epoch 25/100
#32/32 [==============================] - 0s 5ms/step - loss: 308519.4688
#Epoch 26/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.4688
#Epoch 27/100
#32/32 [==============================] - 0s 4ms/step - loss: 308519.4062
#Epoch 28/100
#32/32 [==============================] - 0s 5ms/step - loss: 308519.3750
#Epoch 29/100
#32/32 [==============================] - 0s 6ms/step - loss: 308519.3125
#Epoch 30/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.2812
#Epoch 31/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.2188
#Epoch 32/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.2812
#Epoch 33/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.1875
#Epoch 34/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.1562
#Epoch 35/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.0938
#Epoch 36/100
#32/32 [==============================] - 0s 4ms/step - loss: 308519.0625
#Epoch 37/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.0312
#Epoch 38/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.0312
#Epoch 39/100
#32/32 [==============================] - 0s 3ms/step - loss: 308519.0000
#Epoch 40/100
#32/32 [==============================] - 0s 2ms/step - loss: 308518.9688
#Epoch 41/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.9688
#Epoch 42/100
#32/32 [==============================] - 0s 2ms/step - loss: 308518.9062
#Epoch 43/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.8750
#Epoch 44/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.8750
#Epoch 45/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.8125
#Epoch 46/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.7812
#Epoch 47/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.7500
#Epoch 48/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.6875
#Epoch 49/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.7188
#Epoch 50/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.6875
#Epoch 51/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.6562
#Epoch 52/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.5938
#Epoch 53/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.5312
#Epoch 54/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.5312
#Epoch 55/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.5000
#Epoch 56/100
#32/32 [==============================] - 0s 4ms/step - loss: 308518.4688
#Epoch 57/100
#32/32 [==============================] - 0s 5ms/step - loss: 308518.4062
#Epoch 58/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.4062
#Epoch 59/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.3750
#Epoch 60/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.3438
#Epoch 61/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.3438
#Epoch 62/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.3125
#Epoch 63/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.2188
#Epoch 64/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.2500
#Epoch 65/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.1875
#Epoch 66/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.0938
#Epoch 67/100
#32/32 [==============================] - 0s 5ms/step - loss: 308518.1250
#Epoch 68/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.0938
#Epoch 69/100
#32/32 [==============================] - 0s 3ms/step - loss: 308518.0625
#Epoch 70/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.9688
#Epoch 71/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.9688
#Epoch 72/100
#32/32 [==============================] - 0s 4ms/step - loss: 308517.9375
#Epoch 73/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.9375
#Epoch 74/100
#32/32 [==============================] - 0s 4ms/step - loss: 308517.8438
#Epoch 75/100
#32/32 [==============================] - 0s 4ms/step - loss: 308517.8438
#Epoch 76/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.8125
#Epoch 77/100
#32/32 [==============================] - 0s 4ms/step - loss: 308517.7500
#Epoch 78/100
#32/32 [==============================] - 0s 4ms/step - loss: 308517.7812
#Epoch 79/100
#32/32 [==============================] - 0s 4ms/step - loss: 308517.7188
#Epoch 80/100
#32/32 [==============================] - 0s 8ms/step - loss: 308517.6875
#Epoch 81/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.6562
#Epoch 82/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.5938
#Epoch 83/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.5938
#Epoch 84/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.5938
#Epoch 85/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.5625
#Epoch 86/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.4688
#Epoch 87/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.4375
#Epoch 88/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.4688
#Epoch 89/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.4062
#Epoch 90/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.3438
#Epoch 91/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.3438
#Epoch 92/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.3125
#Epoch 93/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.3125
#Epoch 94/100
#32/32 [==============================] - 0s 2ms/step - loss: 308517.2812
#Epoch 95/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.2812
#Epoch 96/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.1562
#Epoch 97/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.1562
#Epoch 98/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.1250
#Epoch 99/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.1250
#Epoch 100/100
#32/32 [==============================] - 0s 3ms/step - loss: 308517.0625
#Epoch 1/100
#32/32 [==============================] - 0s 2ms/step - loss: 308517.0312
#Epoch 2/100
#32/32 [==============================] - 0s 2ms/step - loss: 308517.0312
#Epoch 3/100
#32/32 [==============================] - 0s 2ms/step - loss: 308517.0312
#Epoch 4/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.9688
#Epoch 5/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.8750
#Epoch 6/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.9062
#Epoch 7/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.8125
#Epoch 8/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.8125
#Epoch 9/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.7188
#Epoch 10/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.7188
#Epoch 11/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.7188
#Epoch 12/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.6875
#Epoch 13/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.6250
#Epoch 14/100
#32/32 [==============================] - 0s 3ms/step - loss: 308516.5938
#Epoch 15/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.5625
#Epoch 16/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.5312
#Epoch 17/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.5000
#Epoch 18/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.4688
#Epoch 19/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.4688
#Epoch 20/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.4062
#Epoch 21/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.3750
#Epoch 22/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.4062
#Epoch 23/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.3438
#Epoch 24/100
#32/32 [==============================] - 0s 3ms/step - loss: 308516.2500
#Epoch 25/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.2500
#Epoch 26/100
#32/32 [==============================] - 0s 3ms/step - loss: 308516.2188
#Epoch 27/100
#32/32 [==============================] - 0s 3ms/step - loss: 308516.1562
#Epoch 28/100
#32/32 [==============================] - 0s 3ms/step - loss: 308516.1562
#Epoch 29/100
#32/32 [==============================] - 0s 3ms/step - loss: 308516.1562
#Epoch 30/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.0938
#Epoch 31/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.0625
#Epoch 32/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.0000
#Epoch 33/100
#32/32 [==============================] - 0s 2ms/step - loss: 308516.0312
#Epoch 34/100
#32/32 [==============================] - 0s 2ms/step - loss: 308515.9375
#Epoch 35/100
#32/32 [==============================] - 0s 2ms/step - loss: 308515.9375
#Epoch 36/100
#32/32 [==============================] - 0s 2ms/step - loss: 308515.9062
#Epoch 37/100
#32/32 [==============================] - 0s 2ms/step - loss: 308515.9062
#Epoch 38/100
#32/32 [==============================] - 0s 2ms/step - loss: 308515.8438
#Epoch 39/100
#32/32 [==============================] - 0s 2ms/step - loss: 308515.7812
#Epoch 40/100
#32/32 [==============================] - 0s 2ms/step - loss: 308515.7812
#Epoch 41/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.7188
#Epoch 42/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.7188
#Epoch 43/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.6875
#Epoch 44/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.6562
#Epoch 45/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.6562
#Epoch 46/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.5625
#Epoch 47/100
#32/32 [==============================] - 0s 7ms/step - loss: 308515.5625
#Epoch 48/100
#32/32 [==============================] - 0s 4ms/step - loss: 308515.5312
#Epoch 49/100
#32/32 [==============================] - 0s 4ms/step - loss: 308515.4688
#Epoch 50/100
#32/32 [==============================] - 0s 4ms/step - loss: 308515.5312
#Epoch 51/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.4375
#Epoch 52/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.4375
#Epoch 53/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.3750
#Epoch 54/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.3750
#Epoch 55/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.3125
#Epoch 56/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.2812
#Epoch 57/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.2188
#Epoch 58/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.2188
#Epoch 59/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.1250
#Epoch 60/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.1562
#Epoch 61/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.1250
#Epoch 62/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.0625
#Epoch 63/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.0000
#Epoch 64/100
#32/32 [==============================] - 0s 3ms/step - loss: 308515.0000
#Epoch 65/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.9688
#Epoch 66/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.9688
#Epoch 67/100
#32/32 [==============================] - 0s 4ms/step - loss: 308514.9062
#Epoch 68/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.9062
#Epoch 69/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.9062
#Epoch 70/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.8125
#Epoch 71/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.8125
#Epoch 72/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.7500
#Epoch 73/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.7188
#Epoch 74/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.6562
#Epoch 75/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.7188
#Epoch 76/100
#32/32 [==============================] - 0s 4ms/step - loss: 308514.6562
#Epoch 77/100
#32/32 [==============================] - 0s 4ms/step - loss: 308514.6250
#Epoch 78/100
#32/32 [==============================] - 0s 2ms/step - loss: 308514.5312
#Epoch 79/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.5312
#Epoch 80/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.5000
#Epoch 81/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.4375
#Epoch 82/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.4375
#Epoch 83/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.4375
#Epoch 84/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.3750
#Epoch 85/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.3438
#Epoch 86/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.3125
#Epoch 87/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.2500
#Epoch 88/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.2812
#Epoch 89/100
#32/32 [==============================] - 0s 4ms/step - loss: 308514.2500
#Epoch 90/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.2188
#Epoch 91/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.1875
#Epoch 92/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.0938
#Epoch 93/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.0938
#Epoch 94/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.0625
#Epoch 95/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.0625
#Epoch 96/100
#32/32 [==============================] - 0s 3ms/step - loss: 308514.0312
#Epoch 97/100
#32/32 [==============================] - 0s 3ms/step - loss: 308513.9062
#Epoch 98/100
#32/32 [==============================] - 0s 3ms/step - loss: 308513.9375
#Epoch 99/100
#32/32 [==============================] - 0s 3ms/step - loss: 308513.9062
#Epoch 100/100
#32/32 [==============================] - 0s 3ms/step - loss: 308513.8438
#<keras.callbacks.History object at 0x0000018AB70EAA10>

# Notice that we are getting these values for loss.
# What if we start this in a varable 
# So we could have a history (see training_optimization2)