# Here we will be going over How and Why Convnets Work

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


IM_SIZE = 224

def resizing(image, ladel):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)), label


train_dataset = train_dataset.map(resizing)
print(train_dataset)

print(resizing(image, label))


for image, label in train_dataset.take(1):
    print(image, label) 


def resize_rescale(image, ladel):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0, label

train_dataset = train_dataset.map(resize_rescale)

train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)


# We are now ready to build our model, and up to now nueral networks have been performing quite well.

# We had seen previously that if we have three neurons / inputs, and in the output we have three neurons as well,
#that would give us nine different connections, and hence nine different weights, plus the bias.

# But for this example, let's consider only the weights. 

# So we have nine different parameters, for three inputs and three outputs.

# And if we have five in our next layer, considering only the weights, that would be 3 (from our previous layer), times 
#5, which will give us fifthteen different parameters.

# We're dealing with an image like the ones we have, which are 224 by 224 by 3, (3 for the 3 different
#channels of red, blue and green).

# So we have this input image now.

# So unlike previuous examples, where we would have features, or specifically in this case input features,
#which do not contain as many elements as this.

# We now have this case where if we want to count the number of features or the number of pixels, we would
#use 224 x 224 x 3, which would give us 150,528 different input features to take into consideration.

# Hence, instead of having a total number of three for our initial input layer, we are going to have 150,528
#different values.

# And if we want to do the same computation which got us the nine parameters, we would have 150,000 times 3,
#that's approximately 450,000 different parameters.

# Now what if we modify this number of neurons in our first dense layer, which was initially 3, and replace it
#1,000 neurons in the output?

# We would see that we no longer have 450,000, but we would now have 150 million different parameters, where each and
#every parameter has to be trained and opitimized.

# That is beacuse our input layer which we changed to 150,000 would be timesed by 1,000, which is what we changed our
#first example dense layer to.

# This becomes clear that deep neural networks like this are better.#

# Dense layers of fully connected layers aren't scalable, since when we increase the number of features, the total
#number of parameters also increases considerably.

# Hence, we need to build a type of layer unlike this one where each neuron isn't connected to all the previous neurons.

# And this layer happens to be the convolutional layer.

# In order to better visualize this, and get a better understanding of the convolutional layer, we will look at the
#CNN explainer. (Convolutional Neural Network Explainer)

# Before we look at the CNN explainer, let's look at this example.

# Here we have a 4 by 4 image.

# So we have 16 different pixels.

# o o o o
# o o o o
# o o o o
# o o o o

# If we flatten out these pixels, we would get this shape.

# Now we have 16 inputs.

# And then in the output, we have four different neurons.

# This would give us 16 connections per neuron, whcih will leave us with 64 parameters, excludinggthe bias.

# o
# o
# o         
# o         
# o             o
# o         
# o             o
# o         
# o             o
# o
# o             o
# o         
# o
# o
# o
# o         


# But with the convo layer, we could go from this 4 by 4 to just a 2 by 2, with just nine parameters.

# So we would have what we call a kernel here, or a filter, which is 3 by 3.

# Which actually corresponds to our weights, which we've seen already.


# o o o o
# o o o o    ------------------>    o    o
# o o o o                           o    o
# o o o o       o   o   o
#               o   o   o
#               o   o   o
#                Filter

# The kernel/filter here, kernel size three, will produce the output of 2 by 2, which when we flatten out will give
#us the four neurons that we used to make the connections to our 16 inputs.

# And so instead of working with 64 parameters, we're working with nine parameters.

# If we want to replicate that same example in the CNN explainer, here's what we get.

# We have this input right here, which produces this output.


#       Input                                             Output
#________________________________                  __________________________________
#|______|_______|_______|_______|                  |               |                |
# ______________________________                   |               |                |
#|______|_______|_______|_______|                  |               |                |
# ______________________________                   |_______________|________________|          
#|______|_______|_______|_______|                  |               |                |
# _______________________________                  |               |                |
#|______|_______|_______|_______|                  |               |                |
#                                                  |_______________|________________|


# Now let's say that we have a kernel/filter size equal to 3 by 3.

# And because we have a kernel/filter size of 3 by 3, we are able to get this output.

# But how is this output gotten?

# We would see that at the top left hand corner, we are going to fit our kernel/filter, which is of size 3 by 3.

# So we'll put in our kernel/filter there, and for each and every value of our kernel/filter and multiple it with
#a corresponding value in the input.

# To obtain the first value, you see how we pass this kernel/filter on the input.

# So at this top left position, we pass this kernel/filter.

# We will notice how we have a 3 by 3 kernel/filter, which is passed on this input.
# This will be represented by the first three columns on rows (194, 196, 198)

# And we have the output, which is the first square located in the top left corner of output grid.

# Then to get this next position, to get the next input location, we would slide our kernel/filter to the right of 
#the input.

# That would get us our next output.

# Next we would slide our kernel/filter down to the bottom right portion of our input grid.

# This will get us our third output location.

# And lastly we will slide our kernel/filter to the bottom left of our input grid.

# This will give us our fourth and finale output location.

# So that's how we get all of the outputs from the inputs and the kernel/filter.

# The kernel/filter produces the output by the amount of times it can slide from one full end of the input to the
#other end.

# And the same for top to bottom.

# Note: The count is one starting at the kernel/filter's starting location.

# The next thing to notice is that by reducing the kernel size perments us to extract more features from the inputs.

# Let's say we had an input grid of 7 by 7, and an output grid of 5 by 5, with a kernel/filter size of 3by 3,
#we would be able to extract far more features from the input than our original 4 by 4 input, 2 by 2 output grids.

# That is because in comparison to the original grid, the new grid will have a reduced kernel/filter size because
#the 3 by 3 kernel/filter has a lot more grid to slide across, giving it access to more input features.

# Now, we are using a reduced kernel/filter size of 3 by 3 inside of our larger 7 by 7 input grid which permits us 
#to extract much more complex information, or complex features from the input.

# Working with larger kernels/filters will permit us to extract larger input features.

# One logical question which may come to our mind is, how are we getting the output feature map size.

# We can get that using this formula where the output width is equal to the input width minus the kernel/filter size
#plus one.

# W out = W in - F + 1 (From left to right)

# H out = H in - F + 1 (From top to bottom)

# Note: If we add padding to our input, our input will increase by twice the number of our specified padding size.
# For example, if we have an input size of 7 by 7 and we add padding of 1, we would now have an input with the size
#of 9 by 9.
# This is because 1 is added all the way around. 1 to the left and 1 to the right. 1 to the top and one to the bottom.

# Note: This will also increase our output size as well.
# This is because we have more input space to travel through with our kernel/filter.

# That is a basic understanding of the padding works.

# We generally use the zero padding, though there are other padding methods, but the zero padding is one of the most
#common since it's easy to use and it's computationally less expensive to work with.

# And we also have another advantage of working with a padding, which is that of ensuring that the corner pixels
#have an influence on our output features which are generated.

# Let's image that we stay with zero padding.

# If we're dealing with an image in which most of the information, or most of tge relevent information is centered,
#then there is actually no issue since the kernel/filter will go through each and every pixel we have of our image
#of a person.

# Now, if we modify the image such that we have a person's face at the corner of the image.

# We'll see that unlike with the image where the person was centered, and that our kernel/filter was able to pass
#through each and every part of the image, with the image of the face in the corner, we have a different scenario.

# Now, what we could do is, if we monitor the number of times the kernel/filter goes through the center persons head,
#we would see that we can actually count the number of times the kernel/filter passes through easily.

# But in the case of the image of the face in the corner, we are on the borders of the input, and since we are on 
#the borders, we cannot successfully pass our kernel/filter through the entire image of the face, beacuse the
#kernel/filter cannot go outside of the inputs grid to pass through.

# So we see that in this case where we are on the borders, this influences the outputs in a smaller way, or exerts
#less influence on the values we get in this feature map which is generated.

# So in an example where we don't have the image of the person in the caenter and we only have the image of the face
#in the corner, we would find that it would have been better to at least pass through the head region just as we did
#with the image in the center where we were able to count the exact number of times we passed through the head region,
#and we were able to extract very useful information from that image because our kernel/filter can pass through
#entirely.

# Now, to remedy the situation of the image of the face in the corner, we have padding.

# We see that when we increase the padding in the input, we are adding more space to our input and therefore adding
#more space for our kernel/filter to move through.

# Thw image stays in its location but the space around it expands making it easier for the kernel/filter to access
#for information about the image.

# Hence, this useful information has more influence on the output features which are being generated, and which is 
#very important because we are trying to extract information from the image in the corner of our input, and pass it
#to the output.

# Another hyper parameter we could look at is the stride.

# Note: We've been looking at all the other hyper parameters so far (Input size, Padding, Kernel size)

# Basically, the stride determines how many pixels we move through at once inside our input with our kernel/filter.

# The default stride is one, which means that our kernel/filter will one pixel left or right, up or down at a time
#inside of our input.

# Note: The stride also affects the movement of our kernel/filter up and down strides also.

# Increasing the size of our stride actually reduces the size of our output and hence reduces the amount of information
#we extract from the inputs.

# And so in general, we get better results by working with smaller kernels/filters and smaller stride values.

# This is because we are able to extract more information from our inputs this way.

# Generally the kernel/filter size of 3 is used in practice with a stride of 1.

# Using padding is always optional.

# And the new formula for when we do decide to use padding is as follows.

#          W in - F + 2P 
# W out = ------------------------ + 1
#                  S

# So this can be read as Width output = Width input - filter plus 2 times the padding divided by stride plus 1.

# For example, let's say we have an input size of 6, that will give us 6 minus filter size of 3 (default filter size),
#plus 2 times the padding of 1 (default padding size),  divide that by the stride of 1 (default stride size),
#and add 1.

# Our anwser should be 6.
# That's how we would obtain our W out size when using the new formula.

# Also note that one good thing when working with a library like Tenslorflow, is when we don't know the exact padding
#to use such that we have a particular output size, we could specify the padding to be valid.

# Once we specify the padding to be valid, Tensorflow automatically calculates the padding for us such that output
#we want matches up.


# Up to this point, we've been supposing that our input image is two dimensional, with a height and a width.

# Now what if we use the kinds of images we have in real life, that are 3D images where we have the red channel,
#the green channel, and the blue channel.

# So if we have an RGB image, we'll see how we can get that output.

# The way this is done is quite starighforward.

# So what we do in this case is include the padding.

# What we will do is multiply each element of our kernel/filter by its corresponding element in the input starting in the
#top left hand corner (including padding), and then add all of the resulting elements up. Then we will slide our 
#kernel/filter to the right and repeat the process. 

# We will do this until we have done the entire width of the input grid.

# Once the width is complete we will do the same process for the height.

# We will do this process for each RGB channel grid in our 3D image.

# This will give us our output for the 3D image.


# From here we will be moving into an Explained Visually project by Setoza.

# Filtering is a part of image processing, and since we are dealing with image data, it's important for us to
#understand how this works.

# If we have an image that we're filtering over, we would notice that no matter what the image is, we would always
#get an output where the outlines are being highlighted.

# Let's imagine that we have an image and the output is bascically a black screen with the outline of an image
#drawn on it, with just the edges hightlighted.

# The explaination says that the filter highlights large differences in pixels values, which generally occur at the
#edges.

# So around the regions of the edges, we would see the large differences are being outlined as compared to the rest
#of the image which is just a black screen and there is no difference.

# And since there is no difference, we would just have the black region.

# The major difference between what we are going to do here and the convolutional neural network is that with this
#we know the kernels/filters values.
 
# So for example we would know that we would have the metrics -1, -1, -1, -1, 8, -1, -1, -1, -1 (kernel/filter elements
#inside our 3 by 3 kernel/filter), which is an edge detector.

# And because we know this, for example, we would be able to obtain our output.

# But with the CNN, or convolution layer, to be specific, what we would do initially is just initialize these values
#and we let the model do training to learn these values automatically.

# So these values are learned by the model during training automatically.

# One of the very first CNN's, or Convnets, was built by Yann LeCun in 1989.

# And this is the structure of the convnet known as the LeNet.

# This Lenet takes in an image.

# For example let's say we have a 28 by 28 by 1 image.

# We also have just one channel, which is black and white image.

# And then we'll pass this image to a convolutional layer.

# After passing through the convolutional layer we would have the sigmoid.

# Also note that for example, we would have a 5 by 5 kernel/filter, with a plus 2 padding.

# We would add zeros (The default character for our padding) around our input, and then add another layer of zeros
#around that first one.

# This would give us an output of 28 by 28 by 6. 
# We'll see how this output was obtained soon.

# So we've seen we have the sigmoid activation.

# From here we have the pooling layer.
# This is a sub-sampling layer and we'll understand how it works.

# So we have 2 by 2 average pooling kernel/filter, with a stride of 2.

# From this we would have another convolutional layer, with a 5 by 5 kernel/filter, and no padding, and we would
#have this output 10 by 10 by 16.

# Next we have an activation, pooling with a 2 by 2 average kernel/filter, with a stride of 2, and this output
#5 by 5 by 16 with flattening. 
# (with flattening all of the features have been modified).

# So we'll leave from the 3D tensor to a 1D tensor.

# Next we have a dense fully connected layer, followed by a sigmoid. Dense: 120 fully connected neurons, sigmoid

# Then another dense fully connected layer, followed by a sigmoid. Dense: 84 fully connected neurons, sigmoid

# Lastly, another dense fully connected layer, with an output: 1 out of 10 classes. Dense: 10 fully connected neurons, output:
#1 out of 10 classes.

# And the exact reason we have this output of 10 neurons since we have 10 classes is because we were predicting
#whether an input is a 1.

# So those inputs are images of handwritten digits.

# So we want to predict whether the handwritten digit is a 1, a 2, or 3, up to 9.

# So we have that, and we also have a 0.

# So 0 to 9 gives us 10 possibilities and that's why we have 10 different classes in our final dense layer.


# Now for the AlexNet, it was built to correctly classify whether an input image belongs to one of a thousnd classes
#in the imageNet data set.

# So we would have the AlexNet with a different architecture.

# And now we'll get into the understanding of how the outputs are obtained.

# And so we'll be rebuilding the Lenet architecture, but this time around using an input of 64 by 64 by 3.

# So we have the RGB channels.

# We have R, G, and B.


#               64 by 64 by 3 (Output Feature Map)
#   ______R__________
#   |   _____G_______|_           
#   |   |   ____B____|_|_        
#   |   |   |        | | |
#   |   |   |        | | |
#   |___|___|________| | |
#       |___|__________| |
#           |____________|
#

# If we pass this input through this convolutional layer, (continue on line 528)

#   Parameters
# _________________________
# |    F = 5   P = 0      |
# |                       |
# |                       |
# |    S = 1  Nf = 6      |
# |                       |
# |                       |
# |       Np = 456        |
# |_______________________|


#we are going to have this output


#           60 by 60 by 6 (Input Feature Map)
#   _________________
#   |   _____________|_<-------- First feature map.    
#   |   |   _________|_|_        
#   |   |   |  ______|_|_|__    
#   |   |   |  |  ___|_|_|__|__
#   |___|___|__|__|__|_|_|__|__|_
#       |___|__|__|__|_| |  | |  |
#           |__|__|__|___|  | |  |
#              |__|__|______| |  |
#                 |__|________|  |
#                    |___________|


# And how do we get this output?

# To get this output, we have to take into consideration the parameters that were given to us for a convolutional 
#layer.

# That said, the Filter size is equal to five.

# So if you check our parameters, we have a Filter size of 5, 0 Padding, Stride equal to 1, and then the number
#of filters equal 6.

# We'll calculate the number of parameters shortly.

# For now, those are the four most important parameters.

# Now we will go over our 5 by 5 filters. 

# We can see that we have the dot products which are completeted to get the outputs as usual.

# So we'll take our first 5 by 5 by 3 filter and connect it to every channel in our R G B 64 by 64 by 3.
# Note: Each channel in our 5 by 5 will be connected to its corresponding channel in our 64 by 64.


#               64 by 64 by 3 (Output Feature Map)
#   ______R__________
#   |   _____G_______|_           
#   |   |   ____B____|_|_        
#   |   |   |        | | |
#   |   |   |        | | |
#   |___|___|________| | |
#  |    |___|__________| |
#  |   /   _|____________|
#  |  |   /
#  |  |  /
# /  /  /
# 5 by 5 by 3        5 by 5 by 3         5 by 5 by 3  ------ Filters

# 5 by 5 by 3        5 by 5 by 3         5 by 5 by 3  ------ Filters


# We will do this to complete the dot products.

# Then we will add it all up to obtain each and every value for the first feature map in our 60 by 60.(see line 533)

# So to obtain this first feature map, we are basically using our first 5 by 5 filter with our 64 by 64 R G B.

# Now, this shows us clearly that when we specify that the number of filters equals 6, it doesn't mean we actually
#have just 6 of the filters stacked.

# What happens is we have three channels in our R G B, and each of the 6 filters has 3 channels.

# And so what we call a fiter is the 5 by 5 by 3 kernels we have.

# Now that we've done the computatuon for the first 5 by 5 by 3 filter, we will do the same the reamining 5.

# That's why we have 6 channels in our 60 by 60 instead of 3 like our filters and the R G B (see line 531).

# Then if we do 5 by 5 by 3, and all that times 6 (5 x 5 x 3 x 6), we should have a total of 450.

# And since for each of our filters we add a bias, that will give 456 parameters to be trained.

# So basically we have 450 weights plus 6 biases to total our number of pararmeters to be trained, as specified in
#our parameters.

# Now we understand why we have 6 channels in our 60 by 60 by 6.

# Now how did we obtain the 60 by 60?

# The way we obtained the 60 by 60 is by applying this formula, which we've seen already.

#          X in - F + 2P 
# X out = ------------------------ + 1
#                  S

# So we just have the output equal 64, filter size 5, 0 padding, Stride 1, which will give us this simplified
#equation 64 - 5 + 0 / 1 + 1 = 60

# That is how we get our 60 by 60 input.

# From here we move to the subsampling pooling layer.

# For the pooling layer we have these two parameters.


#   Parameters
# ______________________
#|_F = 2______S = 2_____| 

# This gives us the filter size and the number of strides.

# To obtain the dimension for the pooling, the formula is slightly different, obviously, because here we don't have 
#the padding.

# So we have X minus filter, divided by the stride, plus 1.

#           X in - F
# X out = --------------- + 1
#               S

# This will take us from a 60 by 60 by 6 to a 30 by 30 by 6 feature map.

# Notice how we still maintain the number of channels, but our input feature map has been subsampled.

# To obtain this, we have X.

# In this case equals 60, so this the equation we would have with this formula.

# 60 - 2 = 58 / 2 = 29 + 1 = 30.

# And that's how we obtained our 30 by 30 by 6.

# Note: We still maintain the number of channels because we still have the same number of filters.

# For the particular case of the Max Pooling, if we want to understand how this works, we'll take a look at this
#example

# Take notice that as we pick these values, we will be returned the max value of all the values in that pool,
#in this case which will be 0.

# So basically what we're doing is we're just simply sliding through the whole image with our kernel/filter and 
#returning the max value of all the values in the pool at that moment.

# In some cases, we will instead take the average of the pool, which is known as average pooling.

# But for the max pooling which is commonly used, we take the max of all the values in the pool at that moment. 


#   (Single Channel From Input Feature Map)
# ________________________________________
# | 0   0   0   0   0   0   0   0   0   0 | 
# | 0   0   2   2   1   2   0   1   0   0 |
# | 0   3   1   3   1   4   0   0   0   0 |
# | 0   0   0   1   1   0   1   0   2   0 |
# | 0   0   1   4   0   0   2   0   3   0 |
# | 0   1   3   0   0   1   3   0   1   0 |
# | 0   2   4   0   1   2   1   3   1   0 |
# | 0   1   1   0   2   1   3   2   1   0 |
# | 0   2   1   0   3   0   2   1   2   0 |
# | 0   0   0   0   0   0   0   0   0   0 |
# |_______________________________________|

# Let's look at the position in the left hand corner and imagine that our 2 by 2 kernel/filter is there.

# Next we will examine the values that fall within our kernel/filter (0, 0, 0, 0).

# So we have a kernel/filter size of 2 by 2, as specified by the parameters, and we're picking out just one of the 
#channels from the input feature map from our 30 by 30.

# Also remember that we have a stride of 2.

# This means that the next set of pool numbers in our kernel/filter (if we move to the right) will be (0, 0, 2, 2),
#which will give us a max pooling value of 2.

# We'll move again and get max value which again is going to be 2.

# We move again and get a max of 1.

# And finally we'll move to last location along the width and get the max value, which will be 0.

# We'll do this for the entire channel, width and height until we have gotten all the max values and formed a new
#feature map.

# So we would leave from a 10 by 10 by 6, and go to a 5 by 5 by 6 Input feature map.

# After the subsmpling layer we have the activation.

# We've alreday looked at the activation in the previous section.

# We've seen the sigmoids and the relu, and the leaky relu.

# So we we should have an understanding of those concepts by now.

# Now we will move to the next convolutional layer.


#   Parameters
# _________________________
# |    F = 5   P = 0      |
# |                       |
# |                       |
# |    S = 1  Nf = 16     |
# |                       |
# |                       |
# |       Np = 2416  relu |
# |_______________________|


# Our parameters here are filter size of 5, 0 padding, stride of 1, number of filters is 16, and the number of 
#parameters is 2,416.

# So we could take this as an exercise to be able to show the total number of parameters we have is 2,416.

# And we will have this output here.

# 5 by 5 by 6        5 by 5 by 6         5 by 5 by 6  ------ Filters

# 5 by 5 by 6        5 by 5 by 6         5 by 5 by 6

# 5 by 5 by 6        5 by 5 by 6         5 by 5 by 6  ------ Filters

# 5 by 5 by 6        5 by 5 by 6         5 by 5 by 6

# 5 by 5 by 6        5 by 5 by 6         5 by 5 by 6    # 5 by 5 by 6  ------ Filters

# Note that these 16 fiters represent our output.

# Notice that we also have the relu added to our parameters.

# This will all get us an output of 26 by 26 by 16.

# Recall that the number of filters dictates the number of channels inour output.
# Note: This is not to be confused with the filter size, which is 5, but we are referring to the Nf, which is 16.

# And the 26 by 26 is obtained by using the same formula that we've seen already.

#          X in - F + 2P 
# X out = ------------------------ + 1
#                  S

# From here we have another subsampling with these parameters.

#   Parameters
# ______________________
#|_F = 2______S = 2_____|

# The subsampling output will be 13 by 13 by 16.

# And then we flatten all of this out to get 2704.

# So after doing subsampling, we obtained our 13 by 13 by 16.

# When we multipy all this we should have 2,704.

# And that is what we'll call our flatten layer.

# So basically we pass our 13 by 13 output through a flatten layer to obtain the 2,704.

# The 2,704 takes each and every value we have in our 13 by 13 feature map and then just simply places it 
#in a one dimensional output, which gives us our flattened layer.

# Then from here, we have a dense layer.

# And then we have a 1,000 neurons in the output.

# And finally, we have 200 neurons.

# In our case we should have two, since we're actually predicting whether it's a parasite or not.

# Now that we understand all of this we should be ready to dive into the code.

# If we want to build convolutional layers with tensorflow, we could make use of the tensorflow keras mmodel.

# Here, we will be looking at the tf.keras.layers.Conv2D

# These are the arguments for the tf.keras.layers.Conv2D that we would pass in.

# (filters, kernel_size, strides=(1,1), padding = valid, data_format = None, dilation_rate = (1, 1), groups = 1, 
#activation = None, use bias = True, kernel_initializer = glorot_uniform, bias_initializer="zeros", 
#kernel_regularizer = None, bias_regularizer = None, activity_regularizer = None, kernel_constraint = None,
#bias_constraint = None, **Kwargs )

# Stride - Note that the stride comes as a tuple. So if we want a stride of 2, for example, the first number is the height
#and the second number is the width. So if we wanted to move along the width of the feature map with a stride of 2,
#we would specify it like this. (1,2)

# Also note that if we wanted the stride to be equal for height and width, we would just use a single number to like
#2, for example, to indicate that we want a stride of 2 horizontally and vertically.

# Padding - One of "valid" or "same" (case sensitive). "valid" means no padding. "same" results in padding with
#zeros evenly to the left/right, top/bottom of the input. When padding = "same" and stride = 1, the output has the
#same size as the input.

# Data_Format - A string. channels_last by default (60 by 60 by 6), or channels_first (6 by 60 by 60). 

# Dilation-rate - An integer or tuple/list of 2 integers, specifying the dilation rate to use in convolution.

# Groups - A posisitive integer specifying the number of groups in which the input is split along the channel axis.

# ACtivation - Activation function to be used. If we don't specify the activation, none will be used.

# Use_bias - Boolean, whether the layer uses a bias vector.

# Kernel_initializer - Intializer for the kernels weights matrix.

# Bias_initializer - Initializer for the bias vector.

#Kernel_regularizer - Regularizer function applied to the kernels weights matrix.

# Bias_regularizer - Regularizer function applied to the bias vector.

# Activity_regularizer - Regularizer function applied to the output of the layer.

# Kernel_constraint - Constraint function applied to the kernel matrix

# Bias_constraint - Constraint function applied to the bias vector