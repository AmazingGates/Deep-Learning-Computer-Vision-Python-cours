# In this section we will be going over the Alexnet.

# In this section we will look at other state of the art convolutional neural network based models.

# We are going to see what makes the alexnet models so powerful.

# First we will discuss the dataset of Alexnet, which is called ImageNet.

# ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories.

# The images were collected from the web and labeled by human labelers using Amazon's Mechanical Turk crowd sourcing
#tool.

# In all, there are about 1.2 million training images, 50,000 validation images, and 150,000 testing images.

# It is also important to note that the images are down-sampled to a resolution of 256x256.

# As for the overall architecture, we have the conv layers followed by max pulling layers, which end with the 
#dense layers, followed by the output layer.

# Another point to note here is that, given that at a time, many times the non-linearity used was the tangent
#or the sigmoid.

# Getting back to the general architecture we would see that the very first convnet has a kernel size of 11 by 11
#for example.

# And although these kinds of kernels permit the network to capture much larger spatial context, we'll see that 
#they are computationally more expensive compared to the kernels with smaller filter size.

# And as we'll see in subsequent sections, the conv nets developed after that didn't use these kinds of large kernel
#sizes, as they were able to make use of these kinds of smaller filters to still capture the large spatial context
#the 11 by 11 filters captured.

# Then to overcome overfitting, the others make use of data augmentation and the dropout technique.

# We can go back to our previous sections if we want to revisit those concepts.

# With that said, we would see that the training details and then one very interesting advantage of working with
#the 11 by 11 kernel size filters is the fact that we could have visualizations.

# So because those kernel sizes are large enough we could visualize them, and then clearly from here we could
#see how our conv layer can capture low level features.

# And so we would see that the first conv layer would permit us to capture low level features. 
 