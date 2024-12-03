import tensorflow as tf # For our Models
import pandas as pd # For reading and processing Data
import seaborn as sns # For Visualization
import keras.layers 
from keras.layers import Normalization, Dense, InputLayer
import matplotlib.pyplot as plt


# Here we will be going over Linear Regression Models

# Now we will start the process of traing our ML Model.

# In our case, our model will be this straight line of equation Y = mX + c.
# As we can see, we have an input X, output Y, and we have our weights which are the constants m and c.

# With that said, next, we have X, which gets multiplied by m, and then we obtain mX. Then it gets added to c.
# That gives us Y, which actually equals mX + c. 
# Then we put everything into a box, which we will call our model

# Now notice we have our model here, we have X that leads into our model, and then we have our output that comes
#out of our model.

# What we are trying to do is get the most appicable values for m and c, such that we have an output that best
#represents our data set.

# So when we are talking about a model, essentially, what we have is a function which tries to be representative 
#of our data set.

# We should note that this model has to be chosen, such that it represents the best data set.



#                  Y = mX + c
#                 /     |---------------------|
#                /      |                     |
#               /   X -----> m -----> mX      |
#              /        |      Mult    |      |
#             /         |              |      |
#            /          |              | Add  |
#           /           |              |      |  
#          /            |              |      |
#         /             |              c -----> Y = mX + c
#        /              |---------------------
#       /
#      /
#     /
#    /
#   /
#  /
# /
#/
#
#


# We are going to look at this in the manner of managing training and optimaztion training sets.

# And before we get there, that is before we see how to get optimal values for m and c, we are going to 
#create this model using tensorflow. 

# The good news is that tensorflow makes it very easy for us to create deep learning models.

# Here we will define our Model

normalizer = Normalization()

model = tf.keras.Sequential([
    normalizer,
    Dense(1),
])
model.summary()
print(model.summary()) # Output.  Model: "sequential" 
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# normalization_5 (Normalizat  (None, 8)                17
# ion)
#
# dense (Dense)               (None, 1)                 9
#
#=================================================================
#Total params: 26
#Trainable params: 9
#Non-trainable params: 17
#_________________________________________________________________
#Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# normalization_5 (Normalizat  (None, 8)                17
# ion)
#
# dense (Dense)               (None, 1)                 9
#
#=================================================================
#Total params: 26
#Trainable params: 9
#Non-trainable params: 17
#_________________________________________________________________
#None
# Note: This model was ran our data preparation file, because that is where our data was. we just moved it over
#to this file because this is where we will discussing linear regressions.

# Here we have just created our first Model with Tensorflow
# Let's break all of this down.

# First we have our Sequential API which we used here in creating this model. 
#model = tf.keras.Sequential([
#    normalizer,
#    Dense(1),
#])
# This is one out 3 ways that we can use in creating these tensorflow models. 
# These are the 3

# Sequential API
# Functional API
# Subclassing Method

# But for now we will be working with the Sequential API

# Let's look at the documentation for the Sequential API
# tf.keras.Sequential - Sequential groups a linear stack of layers into a tf.keras.Model
# It takes in layers. That means instead of having this syntax -
#                                                              - model = tf.keras.Sequential([
#                                                                                             normalizer,
#                                                                                             Dense(1),
#                                                                ])
#                                                                model.summary()
#we could instead have this - 
#                           - model = tf.keras.Sequential()
#                             model.add(normalizer)
#                             model.add(Dense(1))
#                             model.summary()
#                             print(model.summary()) Output 
#Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# normalization_5 (Normalizat  (None, 8)                17
# ion)
#
# dense (Dense)               (None, 1)                 9
#
#=================================================================
#Total params: 26
#Trainable params: 9
#Non-trainable params: 17
#_________________________________________________________________
#Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# normalization_5 (Normalizat  (None, 8)                17
# ion)
#
# dense (Dense)               (None, 1)                 9
#
#=================================================================
#Total params: 26
#Trainable params: 9
#Non-trainable params: 17
#_________________________________________________________________
#None
# Notice that we get the same exact return as the original formula, showing that we can run either and get the
#correct result.
# Note: We ran this verison in our data preparation file to test it on our data.

# So let's get back to explaing our Sequential API.

# The way we build models when layers all form a sequence, that is if we're building deep learning models,
#where the way they are constructed in such that we have the input, we have the model, we have the output,
#and then all the layers which make up this model simply stacked up one layer after another.

#Example

#                         ---------------------------------------
#                         |                                     |
#                         |  |-----|       |-----|      |-----| |
#                         |  |     |       |     |      |     | |
#       --------------->  |  |  1  |  --   |  2  |  --  |  N  | |  -------------->
#                         |  |     |       |     |      |     | |
#                         |  |-----|       |-----|      |-----| |
#                         |                                     |
#                         ---------------------------------------

# Let's suppose we have this type of model made of different layers. layer 1, layer 2, all the way up until
#layer N, where we just simply stack up the layers in this way. 
# Then working with sequential api is a good option.


# Now let's make use of the exact model we are actually dealing with.

# Here we have 2 layers

# First is our normalization layer (N), which we have seen already.
# We understand that our inputs need to be normalized before being passed into our dense layer (D).
# But this should be the first time we are seeing the dense layer, so we try to explain what goes on inside
#of our dense layer.
# But we should note that this model is simply made up of the normalization layer and the dense layer.
# So without the normalization layer, let's take a look at how the dense layer works.

# With a dense layer, let's suppose we have an input.
# So let's say we have the horse power input (from our csv data).
# Then in our dense layer we take the input, the horse power, which we call X, and multiply it by m,
#and then add value c.

# The m is what we call the weight, and the c is what we call the bias. 
# And then what we have is mX + c. And this is our output. Which is actually equaled to (y) predicted. So let's
#call this (y predicted)
# mX + c = y predicted
 


#                         ---------------------------
#                         |                         |
#                         |  |-----|       |-----|  |
#                         |  |     |       |     |  |
#       --------------->  |  |  N  |  --   |  D  |  | -------------->
#                         |  |     |       |     |  |   
#                         |  |-----|       |-----|  |
#                         |                         |
#                         ---------------------------


# So what do we do now when we have many variables. 
# Even though we have many variables, we still have our same dense layer, but there is a difference.
# Let's suppose we have 8 variables. 
# All 8 variables go into our dense layer. 
# So what goes on in our dense layer, for each input, we have m1, m2, m3, m4, m5, m6, m7, and m8.
# Then we have m1 times X (m1X) similar to our mX.
# Then we continue the pattern as such, m1X + m2X + .... + m8X.
# Then we add the bias c. This will give us our dense layer.
# m1X + m2X + m3X + m4X + m5X + m6X + m7X + m8X + c

# This is exactly what goes on in our dense layer. 
# And then we have our otput of y predicted.

# So that's it. We now know exactly how this dense layer works.

# Notice that we have 8 weights, plus the bias (c), giving us 9 parameters in our sequential model.
# They are labeled as "trainable params".

# Also notice that we have 17 "non-trainable params". This comes from the normalization layer.
# It is non trainable simply because we've already fitted this normalization layer to our data, so we don't
#need to modify the mean and the variance anymore.

# So that's it, we understand exactly what is going on.
# Why we have 9 trainable parameters here. 
# So from here we should note that the way we construct the dense layer is quite simply. 
# All we need to do is see how many outputs do we need to have.
# In our case we just want to output a current price.
# And since we want to output this current price, which is just one value, our number of outputs here equal
#one, so thats's why we pick this number of outputs to be equal one.

#model = tf.keras.Sequential([
#    normalizer,
#    Dense(1),
#])
#model.summary()
#current_price = 1

# Also noice that our output was specified here in our dense parameter, (Dense(1)).
# In the event that we wanted to predict more than one current price, or predict a current price for
#now or later, we just have to specify it in the dense parameter, (Dense(2)), or how ever many current
#prices we want to predict. 



#Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# normalization_5 (Normalizat  (None, 8)                17
# ion)
#
# dense (Dense)               (None, 1)                 9
#
#=================================================================
#Total params: 26
#Trainable params: 9
#Non-trainable params: 17
#_________________________________________________________________
#Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# normalization_5 (Normalizat  (None, 8)                17
# ion)
#
# dense (Dense)               (None, 1)                 9
#
#=================================================================
#Total params: 26
#Trainable params: 9
#Non-trainable params: 17
#_________________________________________________________________
#None


# Now we will look at how to plot the model above.

# This should be quite easy by using the tf.keras.utils.plot_model, then we pass in our model.
# We also specify to file, to choose which file we want.
# Next we want to plot this model out then generate a file so we just say ""model.png", show shapes=True"

tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
print(tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True))
# Note: This equation was supposed to generate some sort of graph, but we had an issue with missing pip
#installs. Tried several but was still unlucky. Will revist if totally neccessary


# Since we were unable to generate the graph using the equation, we will draw it the best we can manually

# This is how the entire equation would look

# normalization_2_input: InputLayer |input: |[(None, None)]
#                                   |output:|[(None, None)]

#    normalization_2: Normalization |input: |(None, None)
#                                   |output:|(None, 8)

#               dense_3: Dense |input: |(None,8)
#                              |output:|(None,1)

# From the drawing of our graph, we see clearly that we have an input layer, we have our normalization,
#and we have  our dense layer.

# Notice how the input and output has been specified in our dense layer.
#  The "None" in our dense layer is the batch dimension, and since we could treat data of any batch size, 
#we just have this as "None". 
# That explains why we have "None" in our dense layer, that represents our batch dimension.
# Note: Because our data sets can be large and lead to issues that we will go over later, we generally
#like to have batch sizes of 32 or fewer.

# We could also in addition to our normalizer or before the normalizer, we could define an InputLayer.
# So let's declare that here
# We must also add it the import with Normalization and Dense
# Notice that we have "None" in our InputLayer in our graph, so our inputs are not specified. Input 
#shape is not specified and our output shape is not specified.
# So we will specify them. See Below for example in our model.
# Notice first we have the InputLayer, then we pass in the input_shape, which we could specify.
# We'll specify the shape of the batch by 8. (b,8)
# So if the batch size is 32, that gives the input_shape = (32,8)
# Since we don't usually know thw batch size ahead of time, we usually leave it blank and just use 
#the batch shape (8,)
# That gives us our batch dimensions

# This is the model we would run.

model = tf.keras.Sequential([
    InputLayer(input_shape = (8,)),
    normalizer,
    Dense(1),
])
model.summary()

# After running the model with our InputLayer, we now get back more information in our InputLayer returns.
# We still attempted to use the tf.keras method to no avail, but we'll include it anyway, for future
#codes (in case we get it working and we need it)

tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
print(tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True))

# This will return the new graph below with the additional information. (Still manually drawn)

# This is how the entire equation would look

# normalization_2_input: InputLayer |input: |[(None, 8)]
#                                   |output:|[(None, 8)]

#    normalization_2: Normalization |input: |(None, None)
#                                   |output:|(None, 8)

#               dense_3: Dense |input: |(None,8)
#                              |output:|(None,1)

# Notice that the batch size is still returning as "None", but we now have an 8 for our data shape in 
#in our input and output field inside our InputLayer.