# Here we will be going over the process of Building Convnets

import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import keras.layers 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization
import keras.losses
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.optimizers import Adam



IM_SIZE = 224


# In order to create our convolutional neural network, we're going to make use of the sequential API which we 
#built earlier.

# And then we will define our Inputlayer. Our input shape is 224, and we will define it as an IM_SIZE by IM_SIZE by 3.

# And from there we have the normalizer.

# Next we have our Conv2D layer. We will only be using the filters, kernel_size, strides, padding, and activation
#for now.

# We will put the tf.keras.layers.Conv2D right after the Inputlayer as our normalizer

# We want 6 filters with a kernel_size of 5

# Our stride will equal 1 to indicate a stride of 1 for both width and height.

# Next, the padding is valid, which indicates that there is no padding.

# And finally, we will set our activation equal to "sigmoid" to indicate that we are replicating the Lenet architecture.

# From the Conv2D, we have a max pooling layer.

# Looking at our documentation, this is the format for our max pooling,
#tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = None, padding = "valid", data_format = None, **Kwargs)

# We want a pool_size equal to 2 and a stride of 2.

# Next we can simply copy and paste the Conv2D and MaxPool2D to use for our next set of feature maps.
#Note: We can add a space between the copy and pasted additions.

# The number of filters in our new feature map is 16. But everything else remains the same.

# Also, the MaxPool2D remians unchanged.

# Next we will move on to the flatten layer.

# The Flatten layer is in charge of converting everything into a 1d.

# From the flatten we now have a dense layer, which we've seen already.

# Next we add 2 more Dense layers 

# Now we'll make sure that as we create the final dense layer, that it has an output of 2 neurons, since we are 
#dealing with a binary classification problem.


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

model.summary() # Output



#Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #   
#=================================================================
# conv2d (Conv2D)             (None, 220, 220, 6)       456       
#
# max_pooling2d (MaxPooling2D  (None, 110, 110, 6)      0
# )
#
# conv2d_1 (Conv2D)           (None, 106, 106, 16)      2416
#
# max_pooling2d_1 (MaxPooling  (None, 53, 53, 16)       0
# 2D)
#
# flatten (Flatten)           (None, 44944)             0
#
# dense (Dense)               (None, 1000)              44945000
#
# dense_1 (Dense)             (None, 100)               100100
#
# dense_2 (Dense)             (None, 2)                 202
#
#=================================================================
#Total params: 45,048,174
#Trainable params: 45,048,174
#Non-trainable params: 0
#_________________________________________________________________


# Notice how the dense layer is responsible for a hunge percentage of our parameters. (See line 100)

# If we wanted, we could reduce the parameters by reducing that dense layers value.

# Reducing the value in that dense layer will give us a smaller model.

# A point to note is these number of parameters (456) and (2,416), which we pre-calculated in the last module.

# You will notice that we get these exact numbers when we run our model. (See lines 88 and 93)

# For the dense layer it's obvious, we have a 1000 times 100, plus an additonal 100 for the biases. (See line 102)

# And then we have a 100 times 2, plus an additional 2 for the biases gives us 202. (See line 104)

# It should also be noted that there was a slight error, as we don't have a 5 by 5 by 3, as previously calculated 
#in the last module.

# Instead we have a 5 by 5 by 6, since we have an input number of six channels. (See line 644 in last module)

# And if we take 5 by 5 by 6, and multiply it 16 times, we should have 2,416. (See line 715 in last module)
# Note: All of the filters starting on line 729 in the last module should be updated to 5 by 5 by 6 in the last module.

# It should be noted that we are trying to replicate the Lenet architecture, and this is in no way the state of the
#art kinds of models we use today.

# So in the section on the corrective measures we are going to use even better models.

# For now, we will get an understanding of this.