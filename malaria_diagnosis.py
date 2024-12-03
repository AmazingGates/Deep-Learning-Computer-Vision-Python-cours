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


# In this section we will begin the study of Malaria Diagnosis.

# We will begin with task understanding.

# According to the World Health Organization, the estimated number of malaria deaths stood at 409,000
#in the year 2019.

# In this section, we are going to build a machine learning model based on convolutional neural networks
#to diagnose malaria in a person based on cell information gotten from a microscope.

# In this section, we'll start with loading our data.

# After loading our data, we will visualize this data, process this data, build a model suited for this data, 
#then train this model, and finally, evaluate and test our model.

# As usual , we'll start by defining the task, which in this case entails correctly classifying whether an 
#input cell contains the malaria parasite or not.

# Then we'll go ahead and prepare our data. 
# This data is gong to be made available to us from the tensorflow data sets.

# Then we'll build our model.
# The model, or the particular model we'll be working with in this section will be the convolutional neural
#network.

# Then from here, we'll define the error function.

# We'll go ahead and train our convolutional neural network.

# We'll check out some performance measurement metrics like accuracy, F1's core, precision, recall, and many
#others.

# Then we'll do validation and testing.

# And finally, we'll take as many corrective measures as we can.

# In essence, what we want to do is build a model like this, which takes as input a segmented cell from a thin blood
#smear, and say whether this segmented cell is parasitized or unparasitized.


#       Segmented Cell ----------> Model -----------> Parasitized or Unparasitzed / Infected or Uninfected


# We're supposing that we have no medical background, so we'll briefly look at how malaria diagnosis is related
#to those segmented blood cells.

# To start, we get infected by malaria once we are bitten by a mosquito.

# These mosquito bites usually lead to the passing of Plasmodium Pacifarum parasite into our blood system.

# And so to diagnose whether a particular person has got the malaria or not, it's important to get that persons blood.

# The medical practitioner has to select a finger to puncture, usually the third or fourth finger.

# Then to obtain this blood, they puncture the side of the ball of the finger.

# In the case the blood doesn't well up, they would have to generally squeeze the finger so thhey can obtain
#the blood.

# Then always grasp the slide by its edge.

# So to control the size of the blood drop on the slide, touch the finger to the slide from below.

# From that cdc.gov website, we'll get to the microbe notes website where we'll get more colored images.

# Now, we've obtained the persons blood and there are two possibilities.

# One is getting a thin smear, and the other is getting a thick smear.

# In our case, our data set is obtained from a thin smear.

# So we have our thin smear, which when passed under a microscope, produces images for us to examine.

# From this point, we now segment the cells.

# The cells are now segmented and that's how we obtain our segmented cell images, which now can be used by 
#our model to predict whether that patient has got the malaria parasite or not.

# It should also be noted that we're dealing with a classification problem since our output can only take two
#discrete values.

# This type of classification problem is known as binary classification.

# Since throughout the section, we'll be dealing with image data, it's important to understand how image data
#is represented.

# For example, let's say we have an image of a bird and we zoom in on it.
# The more we zoom in on the image, the more the image is broken down into pixels.
# We could localize the pixels.

# So basically, the image is made up of all these tiny pixels.

# If now we take that image from our dataset, we would notice, for example, that we had 86 by 82 PX, or 86 by 82
#pixel image, for example, meaning that we have 86.

# We would see that we go up to 86.

# So at that point, we have 86 and the width.

# So we would have 86 pixels.

# If we had to go from that point to another point along the width, we would go through the 86 pixels.

# And then if we had to go from that point to a point along the the height, we would have 82.
# That is because we initially had 86 by 82.

# We should see what would be displayed in that position.

# So as we move, we should be able to see what is displayed.

# At that posiition, for example, we had 85, 34.

# So we would have gone 85 steps along the width, and then 34 steps along the height.

# We're considering our origin to be the top left corner of the image.
# So we would have 85 steps along the width to the right side, and 34 steps along the height to the bottom.

# We should be able to localize the pixels at that location.

# Then we would notice when all of the pixels are combined, we will be able to form our image.

# When we zoom out we notice that the pixels become less evident to notice those pixels.

# It should also be noted that each of those pixels contain values ranging from 0 - 255.

# And for each and every pixel, we have three different components.
# The Red, The green, and the blue.

# So if we break the image into the three different components, we would have a break down of the pixels
#value classification according to the scale of colors.

# Note that the values would normalized.

# So basically what we would have done is, taken all the values, and then divide them by 255.

# So if we want to get the un-normalized values, we should be able to take the normalized value and multiply it 
#by 255 to obtain the original values.

# Also, we can represent image data in terms of the height, the width, and the number of channels, which in this case
#would equal to three.
# H,W,(C = 3)

# So we have height, that is the shape of our image tensor, by Width, by 3. 
#H,W,3

# And another common format is the grayscale format.

# With the grayscale format, the number of channels equals one.
#We have just one channel, and it could be represented as a 2D tensor.

# Another interesting point to notice, though we've said for each position, or each pixel, we have a given 
#value per each component.

# We havee to note that all these values fall between black and white.

# That is for Black, if we want to get a black value, the pixel value will be zero.

# And then for white, the pixel value will be 255.

# Obviously when we normalize these values, we'd be going from 0 up 1.

