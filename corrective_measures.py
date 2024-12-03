import tensorflow as tf # For our Models
import pandas as pd # For reading and processing Data
import seaborn as sns # For Visualization
import keras.layers 
from keras.layers import Normalization, Dense, InputLayer
import matplotlib.pyplot as plt
import keras.losses
from keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import numpy as np


# Here we will go over Corrective Measures for our predictions

# Our poor model performance actual has a name.
# It's called UnderFitting.

# Normally our validation loss should be above our training loss, but in some cases it is below the training loss.
 
# When this happens, we want to modify our models so that we have better loss values.

# We want to reduce our loss values as much as we can.

# In order to do this, we will need to make our models more complex.

# Right now the model we have looks like this, where we have a simple regression model, where we have 
#our inputs, our weights, and then our bias. 

# When we add that up we get our output.


# These are our inputs represented by o

# o-------\       (m1, m2, m3, m4, m5, m6, m7, m8) These are our weights
# o------- \                      |
# o---------\ ---------------- mx + c = Output
# o----------\|                   
#  Our Bias   C 
# o----------/ 
# o---------/
# o--------/
# o-------/


# So instead of just having one, we're going to stack up more neurons, by removing the bias and putiing new neurons,
#but they are doing the same thing, basically repeating the same process, then we will link the original 
#inputs to the new neurons, the same way they were previously linked to the bias.

# This will give us a new Dense-layers.

# This is what we call hidden layer

# o------- \      
# o\------  \                      |
# o\\------- \ 
# o\\\------- \                   
#  \\\\         o 
# o-\\\\\----- / 
# o--|\\\\\-- /-o 
# o//\\\\\\\ /
# o/|\\\\\\\/---o
#              D-L

# From here, we could add more Dense layers..


# o------- \     |--__--o
# o\------  \    | /|                 
# o\\------- \   || |---o
# o\\\------- \  ||/| |            
#  \\\\         o/|| |/
# o-\\\\\----- / \||/ /
# o--|\\\\\-- /-o-|||/--o
# o//\\\\\\\ /   \/\  /
# o/|\\\\\\\/---o|\-\/--o
#              D-L

# Notice how they are all linked up, and it's actually the same operation, mx + c.

# So basically we are taking this weight times the inputs plus the bias.

# Then we get the dense layer, times its weight plus the bias and basically continue the same operation for each
#dense layer added.


# o------- \     |--__--o
# o\------  \    | /|                 
# o\\------- \   || |---o
# o\\\------- \  ||/| |            
#  \\\\         o/|| |/
# o-\\\\\----- / \||/ /
# o--|\\\\\-- /-o-|||/--o
# o//\\\\\\\ /   \/\  /
# o/|\\\\\\\/---o|\-\/--o
#              D-L1    D-L2
#                |_______|
#               Hidden Layer

# We have this stacked again, we have our first hidden layer in layer one and layer two, and from here we can even
#add more layers. 
# So let's do that below.


# o------- \     |--__--o
# o\------  \    | /|   | \             
# o\\------- \   || |---o\ \____|
# o\\\------- \  ||/| |  \\-----o     
#  \\\\         o/|| |/    \/___|
# o-\\\\\----- / \||/ /   / ----o
# o--|\\\\\-- /-o-|||/--o-/-----|
# o//\\\\\\\ /   \/\  /  /      |
# o/|\\\\\\\/---o|\-\/--o-------|
#              D-L1    D-L2
#                |_______|
#               Hidden Layer

# Now we've created our third dense layer.

# So now we have our inputs, and our 3 dense layers.


# o------- \     |--__--o
# o\------  \    | /|   | \             
# o\\------- \   || |---o\ \____|
# o\\\------- \  ||/| |  \\-----o     
#  \\\\         o/|| |/    \/___|
# o-\\\\\----- / \||/ /   / ----o
# o--|\\\\\-- /-o-|||/--o-/-----|
# o//\\\\\\\ /   \/\  /  /      |
# o/|\\\\\\\/---o|\-\/--o-------|
# Inputs       D-L1    D-L2    D-L3
#                |_______|_______|
#                  Hidden Layer

# And obiously since we just producing one single price, this will lead to our output layer.

# So now we have input, hidden layers (that consist of 3 dense layers), and output.


# o------- \     |--__--o
# o\------  \    | /|   | \             
# o\\------- \   || |---o\ \____|
# o\\\------- \  ||/| |  \\-----o-----\     
#  \\\\         o/|| |/    \/___|      | o
# o-\\\\\----- / \||/ /   / ----o-----/  |
# o--|\\\\\-- /-o-|||/--o-/-----|        |
# o//\\\\\\\ /   \/\  /  /      |        |
# o/|\\\\\\\/---o|\-\/--o-------|        |
# Inputs       D-L1    D-L2    D-L3     Output
#                |_______|_______|
#                  Hidden Layer


# Next we will try to write out some Tensorflow code to represent this.

# This is how we would write out our model.

# First we have Dense, which represents our first dense layer. This first dense layer has 3 outputs. Then we saparate 
#with a comma. Then we move on to the next.

# Next we have our second dense layer and its outputs, which are 4.

# Next there is the third dense layer with two outputs.

# And lastly, we have the last dense layer, which is actually our output layer with one output.

# Note the outputs of our dense layers are their neurons.

# The dense layers get stacked together. This shows us how easy it is to carry out these operations with 
#TensorFlow.

# Also note that it is very important to ensure that we have the last dense layer or output layer, so that we are
#matching up with our data.

# ([Dense(3), Dense(4), Dense(2), Dense(1)])

# So with this, we have seen how we could do the real Tensorflow.

# Now we have made our model more complicated.

# We've added more hidden layers so that we could learn more complex information stored in our data set.

# There is one point we need to mention.
# And that's the activation functions.

# Activation Functions, simply put, are non-linear functions which add even more complexity to the model.

# If we added activation functions to our existing model, it will add more complexity for us, and possibly give
#us access to more complex information in our data set.

# Common Activation Functions are 

# - The Sigmoid Activation Function - 1 divided by 1 plus e to the negative x
# - The Tensh Activation Function - e to the x minus e to the negative x divided by e to the x plus e to the negative x
# - The Relu (Rectified Linear Unit) Activation Function - An example of this would be if when x is greater that zero, 
#we maintain x, that is x remains the same or the output equals the input.
# But when x is less than zero, the output becomes zero.
# - The Leaky Relu Activation Function - Now for this one, when x is greater than zero, the output is the same
#When x is less than zero, the output is negative of certain alpha times x, hence the term leaky.
# As the alpha is generally a very small number, so if we have 0.1, we could have a negative 0.1x, which is kind of
#like giving us very small outputs which are kind of close to zero. 

# For now we are going to use the relu activation function and we will see subsequently that all these activation
#functions could be gotten from tensorflow keras activations.

# We should also note that if we are to apply, for example, the sigmoid activation function, what we will do
#is just after multiplying our weights by the inputs and getting the outputs, we are going to the sigmoid, and
#we actually end up doing the sigmoid for each neuron.
# That will give us the sigmoid of our computations.

# We will apply that same logic to how we use the Relu Activation Function

# We now have everything we need to make our model perform better, and stop underfitting.

# We will add a dense layer to our model.keras Sequential function. 
# And let's say we have (32) neurons in that dense layer. 
# So our output woud be 32.

# Then we add another dense layer with (32)

# We will add a final dense layer of (32) next.

# And lastly since we want one output, we will have a finally dense layer, which will represent outpuy layer,
#which has only (1) output

# Next we specify the activation 
# So we'll have activation = relu
# We will do the same for each of our dense layers, but not the output layer.
# That is because we don't want to interfere how our model comes up with its outputs.

# Then we run our model.

# Example below.

#model = tf.keras.Sequential([
#                             InputLayer(input_shape = (8,)),
#                             normalizer,
#                             Dense(32, activation = "relu"),
#                             Dense(32, activation = "relu"),
#                             Dense(32, activation = "relu"),
#                             Dense(1), 
#])

# Note: This is an alternate way we can run this algorithm

#model = tf.keras.Sequential()
#model.add(InputLayer(input_shape = (8,)))
#model.add(normalizer)
#model.add(Dense(32, activation = "relu"))
#model.add(Dense(32, activation = "relu"))
#model.add(Dense(32, activation = "relu"))
#model.add(Dense(1)) 


# And to make this model even more complex, we can increase the number to 128

# To keep things simple for now, we will just use this format and increase our dense layer neurons to 128.

#model = tf.keras.Sequential([
#                             InputLayer(input_shape = (8,)),
#                             normalizer,
#                             Dense(128, activation = "relu"),
#                             Dense(128, activation = "relu"),
#                             Dense(128, activation = "relu"),
#                             Dense(1), 
#])

# Nottice that now we have many more parameters to train on, but we still only have 17 non-trainable

# And again, we have the 17 non trainable parameters because our original model we had 17 non trainable and 
#9 trainable parameters. 

# Now that we have increased the dense layer neurons, we have many more parameters to train on, but the 17
#remains the same because our normalizer is still the same.

# Next we will plot our new model based on the new information we have.

# tf.keras.utils.plot_model(model, to_file = "model.png", show_shapes = True)
# Note: This is the function we tried to use last but still isn't functioning prorperly.
# We will circle back around to this issue.

# So we will draw the plot manually again.

#                     | input:  | [(None, 8)]
# input_5: InputLayer | output: | [(None, 8)]
#                   |
#                   |
#                   |            | input:  | [(None, 8)]
# normalization_2: Normalization | output: | [(None, 8)]
#                   |
#                   |
#                   |    | input:  | [(None, 8)]
#       dense_ 10: Dense | output: | [(None, 128)]
#                   |
#                   |
#                   |    | input:  | [(None, 128)]
#       dense_ 11: Dense | output: | [(None, 128)]
#                   |
#                   |
#                   |    | input:  | [(None, 128)]
#       dense_ 12: Dense | output: | [(None, 128)]
#                   |
#                   |
#                   |    | input:  | [(None, 128)]
#       dense_ 13: Dense | output: | [(None, 1)]


# Notice that this plot represents the new comlex mmodel we have. 

# You can see that we have Our original 8 inputs.

# Next we have our 3 dense layers with their 128 neurons

# And lastly we have our output layer with its 1 neuron.

# From here, we will compile our model again.

# We will use this alogrithm to run our model.

#model.compile(optimizer=Adam(learning_rate = 0.1),
#               loss= MeanAbsoluteError(),
#               metrics= RootMeanSquaredError())

# Notice that we changed our learning rate to 0.1.

# Now we can run our model.fit on our new data.

# Notice that our new loss is a smaller magnitude compared to what we had previously.

# Also how much we are able to reduce the underfitting in our model.

# The starting loss is 153,005

# With the new complexity of model we are able to cut that down to 33,928.

# Next we will view our plots for this new complex model return.

# We will do that by running these functions again.

#plt.plot(history.history["loss"])
#plt.plot(history.history["val_loss"])
#plt.title("model loss")
#plt.ylabel("loss")
#plt.xlabel("epoch")
#plt.legend(["train", "val_loss"])
#plt.show()

# As you can see we have a totally different plot now thanks to the complexity of our model.
# We were able dramatically cut down on the underfitting.

# Our training data value dropped from 160,000 to around 40,000.

# Also, our validation data dropped from 80,000 to around 42,000.

# Notice that this time around we have a validation loss that is higher than that of the training.

# This is normal because obviously the model was trained on the training data so it would tend to perform better
#than data it wasn't trained on.

# Now if a model performs very well on the training data and it doesn't perform well on the validation data,
#then we have a problem which is known as overfitting.

# But for now we will go over the root squared mean error plot with our new complex data model, as we did for the
#model loss.

# These are the functions we will run to generate our plots.

#plt.plot(history.history["root_mean_squared_error"])
#plt.plot(history.history["val_root_mean_squared_error"])
#plt.title("model performance")
#plt.ylabel("rmse")
#plt.xlabel("epoch")
#plt.legend(["train", "val_rmse"])
#plt.show()

# Notice that we have something very similar to our model loss.

# We were able to cut down the size of the underfitting dramactically.

# Our starting value for the model performance training data was about 200,000, and we were able to cut that down
#to about 50,000.

# Also, our validation rmse data value was about 100,000, and we were able to cut that down to about 55,000.

# Next we will run the model for our test data again on our new complex model.

# This is the function we will run.

# model.evaluate(x_test,y_test)
# Output 
# 4/4 [==============================] - 1s 39ms/step - loss: 39688.8086 - root_mean_squared_error: 48269.4531

# This is the loss value of our test data 39688.
# This is the root mean squared error value 48269.

# From here we will obtain and print the y predicted (y_p).
 # This is how we will run the function 


# y_p = list(model.predict(x_test)[:,0])
#print(y_p) Output

#4/4 [==============================] - 0s 11ms/step
#[118175.805, 423121.75, 243371.27, 166764.8, 239373.98, 348187.94, 231654.08, 370706.12, 400984.56, 443528.22, 
#444870.75, 229357.6, 240034.92, 146138.52, 415556.06, 304528.06, 258606.55, 453527.16, 302679.2, 300937.2, 263252.53, 
#416502.25, 145990.61, 160818.69, 525040.75, 384956.06, 328580.72, 252183.8, 338395.56, 255337.94, 318533.1, 137830.39, 
#351857.75, 191448.7, 127488.445, 437562.44, 216331.02, 389646.66, 515316.94, 330425.5, 487673.56, 119661.37, 155329.84, 
#355620.1, 573348.94, 260481.83, 468374.22, 434659.5, 294966.75, 296966.0, 330515.75, 259781.67, 538853.4, 486945.0, 
#256247.81, 355335.25, 115991.02, 259465.06, 298533.0, 223550.48, 350455.0, 522534.8, 361845.88, 485622.0, 179972.73, 
#245673.23, 280628.2, 287850.56, 520098.62, 340227.2, 186040.84, 243026.47, 149731.19, 469493.7, 170299.11, 298012.38, 
#96152.74, 471071.62, 456611.9, 229705.45, 146824.89, 358873.25, 526925.3, 352987.62, 217204.39, 336310.75, 192949.89, 
#392077.88, 460489.7, 448826.9, 444856.94, 130602.21, 172756.05, 310388.16, 48941.074, 202365.02, 425890.06, 395058.72, 
#150340.84, 433063.44]

# Notice that these are the returns for our y predicted formula.

# Next we will use a bar graph to plot out our Actual Car Prices vs our Predicted Car Prices.

# These are the functions we will use to do that

#y_a = list(y_test[:,0].numpy())

#y_p = list(model.predict(x_test)[:,0])

#ind = np.arange(100)
#plt.figure(figsize=(40,20))

#width = 0.1

#plt.bar(ind, y_p, width, label = "Predicted Car Price")
#plt.bar(ind + width, y_a, width, label = "Actual Car Price")

#plt.xlabel("Actual vs Predicted Price")
#plt.ylabel("Car Price Prices")

#plt.show() 

# Notice that with our bar graph we now obtain a model that performs way better than our previous model.

# Our Blue lines represent the actual current price, and the orange lines represent the predicted prices.

# Our model still isn't perferct, but it's functionality is doing quite well.


# At this point we've taken some creative measures and our model is performing better.

# Next we will look at how to load our model even faster and more efficiently.
# This can be done using the Tensorflow data API.

# For this we will be looking at the tf.data function.

# Inside the tf.data overview, we will be using the tf.data.Dataset
# The tf.data.Dataset function represents a potentially large set of elements.

# This class is made of many methods so we will be starting with the fromTensorflowSlices method.

# So from here we will adapt our code to use Tensorflow's data API in order for us to gain from all the 
#advantages that come with it.

# Now note, that when we're working with a data set of let's say, a thousand elements, we wouldn't clearly
#notice the advantage of using this  data API.

# But as our data sets get larger, it is very important to master how to use this API.

# With that said, we're going to redefine our XTrain.

# So here, we're going to say train data set, and we're going to have tf.data.Dataset dot from tensorslices

# Next we are going to create a tuple, then we will pass in our XTrain and YTrain

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Now that we got this, the next thing we're going to do is we're going to is shuffle our data set.

# Now we've already done shuffling, but in the case of where shuffling wasn't done previously, we could
#do the shuffling very easily now.

# So here we have train dataset dot shuffle and we specify a buffer size, let's say 8.

# And let's see exactly what this buffer size actually means.
# For instance, if our dataset contains 10,000 elements but buffer size is only 1,000, then shuffle will
#initially select a random element from only the first 1,000 elements in the buffer. Once an element is selected,
#it's space in the buffer is replaced by the next(i.e 1,001 st) element, maintaining the 1,000 element buffer.

# Also inside our shuffle function, we will add a reshuffle each iteration by setting that parameter to true.

# train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True)

# With that done, next we will have our train dataset batched by size 32

# train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32)

# And from here we can do prefetching.
# Prefetching creates a Dataset that prefetches elements from this dataset.
# Also, we are told that this allows later elements to be prepared while the current element is being processed.

# train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).
#prefetch(tf.data.AUTOTUNE)

# So we now have a training data set.

# Next we will add this, and run it to batch our data set.
#   for x,y in train_dataset:
#       print(x,y)

# With all of that, now we've batched our data and we're ready to train it.

# First we will resuse our functions that we used to batch our data to run our validation.

# This is how we will run the validation with our new algorithm

# val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

# Now that our data has been validated we will test it using the same method as the last two processes.

#test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#test_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)


# These are the steps we need to take to convert this data set so that we can work with our tensorflow Data API.

# We won't do any modifications on this model.

# All we need to do is specify train data set.
# The we add our val data set.

# We can do that like this.

# history = model.fit(train_dataset, validation_data=val_dataset, epochs = 100, verbose = 1)

# Next we will run a new plot for our trained data set model loss value.

# We will use this alorithm again.

#plt.plot(history.history["loss"])
#plt.plot(history.history["val_loss"])
#plt.title("model loss")
#plt.ylabel("loss")
#plt.xlabel("epoch")
#plt.legend(["train", "val_loss"])
#plt.show()

# Notice that our value losses continue to drop and improve as we improve and make our model complex.

# Next we will run a new plot for our trained data set root squared mean error value.

# We will use this algortihm again.

#plt.plot(history.history["root_mean_squared_error"])
#plt.plot(history.history["val_root_mean_squared_error"])
#plt.title("model performance")
#plt.ylabel("rmse")
#plt.xlabel("epoch")
#plt.legend(["train", "val_rmse"])
#plt.show()

# Notice that this new model is running much more efficiently with our improved complex model data.

# Finally we will evaluate all of this new data in our complex model.

# We will use this algorithm again.

# model.evaluate(x_test,y_test)
# Output 4/4 [==============================] - 0s 3ms/step - loss: 39776.9453 - root_mean_squared_error: 48923.9219

# Notice that this isn't actually very different from what we had previously, and we may feel like this is producing
#a different plot because of the drastic changes.

# But the model in its current state already had a loss of around 30,000 as opposed to the 150,000 from our
#previous plots.

# So if we have to recompile our model, it doesn't really change much, but it does speed up the training, cutting back
#the unnessecary processing.

# So we wouldn't expect to have better loss values with a tf.data, but instead we could attain a better performance
#faster.

# Now we can go ahead and view this on our bar chart.

# We will use this algorithm again

#y_a = list(y_test[:,0].numpy())

#y_p = list(model.predict(x_test)[:,0])
#print(y_p)


#ind = np.arange(100)
#plt.figure(figsize=(40,20))

#width = 0.1

#plt.bar(ind, y_p, width, label = "Predicted Car Price")
#plt.bar(ind + width, y_a, width, label = "Actual Car Price")

#plt.xlabel("Actual vs Predicted Price")
#plt.ylabel("Car Price Prices")

#plt.show()

# Notice that it is similar to what we had already.

# That is fine for now. We are just aiming at mastering the basics of working with a data API.

# Though in subsequent sections, we'll look at even more interesting ways of working with this API. 