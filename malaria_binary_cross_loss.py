# Here we will be going over Binary Crossentropy Loss

import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import keras.layers 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization
import keras.losses 
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import Accuracy, RootMeanSquaredError
from keras.optimizers import Adam



IM_SIZE = 224


model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    tf.keras.layers.Conv2D(filters = 6, kernel_size = 5, strides= 1, padding = "valid", activation = "sigmoid"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, strides= 1, padding = "valid", activation = "sigmoid"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    Flatten(),

    Dense(1000, activation = "sigmoid"),
    Dense(100, activation = "sigmoid"),
    Dense(2, activation = "sigmoid"),
])

model.summary() 

# We're now building the model, so let's move on to the error sanctioning section.

# For error sanctioning binary classification problems we generally use the binary cross entropy loss.

# Let's look at this formula of the binary cross entropy loss from a ML cheat sheet website.


#   Math

# In binary classification where the number of classes M equals 2, crossentropy can be calculated as:

#       -(ylog(p) + (1 - y) log (1 - p))

# If M > 2 (i.e. multiclass classification), we calculate a separate loss for each class label per observation
#and sum and result.

#       M
#   - Omega Yo,c log(Po,c)
#     c = 1


# Just as most error sanctioning functions with a binary cross entropy, we're trying to penalize the model, when
#the actual prediction (Actual (y)) is different from the predicted value (Predicted (p)).

# And so in this case of a binary classification problem, if our actual prediction is meant to be a 1, and our model 
#predicts a 0, putting that into the equation would get us a one log zero.

#  1 Actual (y)
#  0 Predicted (p)

# So here we would have 1 log 0, And then we have 1 minus 1, which is 0, with a log of 1, since p is 0 and 1 minus 0
#is 1.

# -(1 log(0) + (1 - 1) log (1))

# If we plot out the curve for the log, we would have something like this.

# And we notice that as we approach zero, that's as X is tending to zero, we would have this log which is going towards
#negative infinity.

# And so the log of zero is a very large number.


#                      y (Y equals log of X)
#                      | 
#                      | 
#                      |       ______________
#                      |      /
#   -------------------|-----|-------------> x
#                      |    /1 (1 is here where the line intersect)
#                      |   |
#                      |  /
#                      | |
#                      |/ 


# Now with this case, we have zero so we can take this off. (1 - 1) (see line 69 to see original)

# -(1 log(0) + log (1))

# And so this means when the actual prediction is 1, and the predicted value is 0, our final output becomes a 
#large number.

# Now let's modify this.

# Let's say we have a zero for the actual, and a one for the predicted.

#       -(ylog(p) + (1 - y) log (1 - p))

# In this case we would have a similar outcome scenario.

# This is because we'll have y equals 0, and 0 times whatever the p is equals 0.

#       -(0log(1)) = 0

# Next we would have 1 minus the actual, which is 0, so 1 minus 0 equals 1.

#       (1 - 0) = 1

# Lastly, we would have 1 minus the predicted, which is 1, so 1 minus 1 equals 0.

#       (1 - 1) = 0

# Again, this would leave us with a large value because  of zero will take us towards a negative infinty
#on the plot chart.

# And then we have one times that big number, giving us a big number.

# And so our model is sanctioned because it hasn't correctly predicted the expected output.

# Now let's suppose that the acutal is indeed 1, and the predictioon is also 1.

#       -(ylog(p) + (1 - y) log (1 - p))

# In that case we would have 1 log of 1

#       -(1 log(1))

# And then 1 minus 1.

#       (1 - 1) = 0

# And lastly a log of (0)

#       ( 1 - 1) = 0

# But in our second step, because we have 1 minus 1 equals 0, this 0 and the 0 from step 3 will cancel each other out,
#and what will be left with is step 1.

#       -(1 log(1))

# And according to our plot chart, when X equals 1, log of X equals 0.

# So log 1 is 0, 

# And we have our final answer of 0.

# So our final output would be 0, telling us that the model has done its job correctly.

# If we do the same for when actually 0, and predicted 0, we'll always get 0.

# Our model now makes use of the binari crossentropy loss to update its weights.


# If we have a 0 for actual, and let's say a 0.8 predicted.

# We would compute this bce as a binary crossentropy loss, 


y_true = [0,]
y_pred = [0.8, ]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) # tf.Tensor(1.6094375, shape=(), dtype=float32)

# Notice we get a value of 1.6.

# Now let'smodify this.

# So let's say we have a y_pred of 0.02

y_true = [0,]
y_pred = [0.02, ]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) # Output tf.Tensor(0.020202566, shape=(), dtype=float32)

# Notice that we now have a return of 0.02.

# Now if we take the 0.02  and change it to 0.2 we'll have this.


y_true = [0,]
y_pred = [0.2, ]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) # Output tf.Tensor(0.22314338, shape=(), dtype=float32)

# Notice that our return is now 0.2

# Now we could always stack up outputs as we did in the beginning.

# So let's bring back these outputs 


y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 0]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) # Output tf.Tensor(1.1007609, shape=(), dtype=float32)

# Notice the value we have as a return for our loss.

# Also, we can change the zero to a one in our y_pred column


y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 1]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) # Output tf.Tensor(4.9340706, shape=(), dtype=float32

# Notice that this will increase the value of our loss

# Another argument we can pass in is a "from logits" argument.

# For the from_logits argument, by default, it's actually false.

# But then the default value of false is supposing that the output of our model, the y_pred, will always produce
#values in the range of zero one.

# Now, the way our model has been constructed ensures that all our values will be produced in that range because
#we have the sigmoid activated.

# And if we recall, with the sigmoid, as we increase the value of X, the output was going towards one.

# So as we increase the value of X, output goes towards one.

# As X becomes very small or take a very large negative number, X goes towards zero.

# So in fact, it's always going to ensure that the output lies between zero and one.

# That's why it's very important to have the sigmoid activated.

# Now, in the case where we don't have the sigmoid activated, our output doesn't necessarily lie between zero and
#one.

# What we're going to do is, use from_logits = True in our algorithm. 

# Specifying this simply means that we are implying that we are not sure about the output of our model always
#falling into the range of zero one.

# Now let's run our code below to see what we are returned.


y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 1]
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bce(y_true, y_pred)
print(bce(y_true, y_pred)) # Output tf.Tensor(1.0227046, shape=(), dtype=float32)

# Notice that we are returned a totally different response.

# So we have to be very careful when working with this.

# If the values range from zero to one, make sure we use a default from_logits equal false.

# From here let's go ahead and compile our model.


y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 1]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred))

# We have our optimizer, we have the loss, our losses, the binary cross entropy loss.

# For now we will comment out the metrics because we are not taking that into consideration right now.

# So That's it, we've compiled our model. 

model.compile(optimizer = Adam(learning_rate=0.1),
              loss = BinaryCrossentropy())
              #metrics= RootMeanSquaredError())