# Here we will be going over Data Processing for our Malaria Data.

import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import keras.layers 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization
import keras.losses
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.optimizers import Adam


dataset, dataset_info = tfds.load("malaria", with_info=True, as_supervised=True, shuffle_files=True, 
                                  split=["train"])


for data in dataset[0].take(4):
    print(data)


def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    dataset_size = len(dataset)
    train_dataset = dataset.take(int(TRAIN_RATIO*dataset_size))

    val_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))

    val_test_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))
    val_dataset = val_test_dataset.take(int(VAL_RATIO*dataset_size))

    test_dataset = val_test_dataset.skip(int(VAL_RATIO*dataset_size))

    return train_dataset, val_dataset, test_dataset


TRAIN_RATIO = 0.8
VAL_RATIO= 0.1
TEST_RATIO = 0.1


train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)  

print(list(train_dataset.take(1).as_numpy_iterator()), list(val_dataset.take(1).as_numpy_iterator()), 
      list(test_dataset.take(1).as_numpy_iterator()))


for i, (image, label) in enumerate(train_dataset.take(16)):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(image)
    plt.title(dataset_info.features["label"].int2str(label))
    plt.show()
    print(plt.imshow(image))

# At this point we will be diving into data processing.

# Our data processing unit for now will be made of two parts.

# The first part will be the resizing part.

# So if we have an input image of let's say 102 by 102, as width and height, we would have this.


#         102
#   _____________________                    
#  |                     |                         
#  |                     |                            
#  |                     | 102      
#  |                     |                          
#  |                     |                            
#  |                     |                          
#  |_____________________|                           


# We're going to transform this into an image of a fixed width and fixed height. 


#         102                                                 
#   _____________________                              ______________________
#  |                     |                            |                      |
#  |                     |                            |                      |
#  |                     | 102      ----------->      |                      |
#  |                     |                            |                      |
#  |                     |                            |                      |
#  |                     |                            |                      |
#  |_____________________|                            |______________________|


# Now with that said, all our images, irrespective of their width and heights, will now just have one width
#and one height.

# In this case we'll consider an image size of 224.

# So our width and our height will be converted to 224.


#         102                                                224    
#   _____________________                              ______________________
#  |                     |                            |                      |
#  |                     |                            |                      |
#  |                     | 102      ----------->      |                      |  224
#  |                     |                            |                      |
#  |                     |                            |                      |
#  |                     |                            |                      |
#  |_____________________|                            |______________________|


# And subsequently, we'll see why we are picking this image size of 224.

# After this resizing, our next part will be on normalization.

# So we'll have an input image which will normalize, such that all the data falls in a given range.


# Here we have the standardization process, and the normalization process.

#    X = X - M                                          X - X min
#   -----------                                    X = ----------------
#       SD                                              X max - X min
#  Standardization Process                          Normalization Process


# In this case we are going to use the normalization process, and we're going to explain why.

# In the previous example of the car prediction, we actually worked with standardization.

# In standardization, each value is subtracted from the mean and then divided by the standard deviation.

# That's actually each value on each and every column of X here is being standardized based on its mean and 
#standard deviation.

#   X1   |   X2  |   X3  |   X4  |   Y
#--------------------------------------------
#   20   |  5600 |  2    |  50   |  750
#   30   |  7100 |  5    |  80   |  641
#   21   |  9800 |  19   |  70   |  900
#

# As we'll notice, these values, for example, are normally distributed.

# That is, we have a mean value on average value and we have a standard deviation, or range of values where most 
#of the time our values fall in.

# So it will look bell shaped.


#        |------------|
#       |      |       |
#      |       |        |
#     |        |         |
#    |         |          |
#   |          |           |
#  /           |            \
# /            |             \
#/             |              \
#            Mean
#|                             |
#|-----------------------------|
#       Standrad Deviation


# We have our mean and then we have a certain standard seviation

# That's why if we look at X2, we wouldn't have 5,600, 7,100, and then a value of let's say 12, for example.

# This isn't very typical since most of the time these values fall under a given range and there's a certain
#average value for which all of the values fall around.

# So that's why it's typical to have these kinds of values.

# Next, we'll notice how these values fall under a given kind of range (X3) and with this too (X4).

# Now, this is for when we deal with standardization, and that's why we previously used the Standardized
#Process.

# In the case of image data, the choice of whether to standardize or normalize will depend on the kind of data we're
#dealing with.

# So if we have images where most of its pixels revolve around a particular mean value, then we'll want to standardize
#and if this image is made of pixels where their values are mostly different from one another, we'll want to
#normalize.

# In our case, as we continue in this section, we are gonna go with the Normalization Process, which means we will
#have X - X min, which is 0, divided by X max, which is 255 - X min, which is 0.

#     X - X min = X
#-----------------------
#   255 - X min = 255

# So simplified we would do X divided by 255

#       x
#  ------------
#      255

# So we'll normalize the inputs before passing them into our model.

# Now, it should be noted that for some other data sets like, for example, ImageNet data set or the RDE20K Image 
#Segmentation dataset, they have their known mean and standardized deviation values. 

# Nonetheless, for whatever problem we have to deal with, we may have to work with standardization or normalization.

# We would most likely experiment to see which one works better.

# With that being said, since we're dealing with a Tensorflow data API, we are going to use this map method that helps
#us in this pre-processing.

# train_dataset.map

# We'll start first with resizing.

# Then we'll call the resizing method, which we shall define.

# This resizing method takes in as parameters the image and the label.

# But note that we're only doing processing on the image.

# So we're going to pass this in, and then we're going to resize it using the resizing method that comes with tensorflow,
#tf.image.resize.

# So let's return our tf.image.resize, and pass in its parameters of image, then the image size by image size

# So we'll have (IM_SIZE, IM_SIZE)

# Note that we must define IM_SIZE before we can use it.

# So we'll set our image size to the size we specififed earlier at 224.

# Now that we have resized this, we just add a label

# So here again, we're just basically taking the image, resizing it, and the label remains the same.

# Now we can run it, and then we have our trained dataset, which now has been resized.


IM_SIZE = 224

def resizing(image, ladel):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)), label


train_dataset = train_dataset.map(resizing)
print(train_dataset)

print(resizing(image, label))

# Next we will write a for loop for our image and label to print out our values.

# For example we will specify that we want to take out just one element by using a .take

for image, label in train_dataset.take(1):
    print(image, label) # Output

# Note: We achieved the same reult by just printing out the resizing function with its parameters.

#  We'll notice that the shape of this image is in the shape of 224, 224, 3, irrespective of whichever image 
#we choose.

# Also notice that this is a float 32, Unlike previously where we had an unsigned int.

# Now we could always do casting right here to modify that, depending on what we are working on.

# With that being said, we're done with the resizing.

# Next we will look at rescaling. 

# We will start the process of doing this by updating our resizing function to def resizing_rescaling.

# To make things easier we'll just use the names resize and rescale for our updated function

def resize_rescale(image, ladel):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0, label

# So our resizing and rescaling function is such that after resizing, we rescale by dividing by 255.

# So we divide all of our values by 255, and our labels remain the same. (see line 273)

# Next we will update our train dataset to reflect our new function.

train_dataset = train_dataset.map(resize_rescale)

# Now we can run our new function.

# Just as we did before in the previous section, we're going to shuffle our data, put it in batches, and then we'll
#do prefetching.

train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)