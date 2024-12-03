import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer, Dropout, Rescaling, Resizing
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L2, L1
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns

# In this new section we will be going over the process and steps of evaluating and detecting Human Emotions.

# We will be starting with data preparattion.

# In this session we will be building a system that allows us to automatically detetct whether an input, which in 
#this case is an image, an image of angry, happy, or sad.

# So in fact we want to be able to have this kind of input, pass it into our system which we are going to build, 
#and then automatically infer what the emotion of the person in the image is.

# And so to train this system we'll be making use of the datasets that have been made availble to us from kaggle.

# We are now going to take a closer look at the dataset we will be using.

# Notice that inside our dataset we have test and train directories.

# Inside each directory we will see three different folders with the labels, Happy, Sad, Angry.

# So what we do is generally when trying to create this kind of dataset for classification problems, what we want 
#to do is make sure that we put each of the three different images in separate folders.

# This makes it easier to build our model.

# From this point we will go on to create a tf.dataset based on the images we downloaded from kaggle.

# To do this we will make use of the tf.keras.utils.image_dataset_from_directory

# This method has a parameter of directory which will take one of our directories of test or train.

# That will look like this 

# tf.keras.utils.image_dataset_from_directory(
#                        directory,
#)

# We will be using the train directory first so we can create that using train_directory and assisning that to path
#of our train directory from our dataset.

train_directory = "Human Emotion Images/train"

# We will do the same thing for the validation but we will assign it to the test directory.

val_directory = "Human Emotion Images/test"

# These two will get initialized above this code.

CLASS_NAMES = ["angry", "happy", "sad"]

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256
}



train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',
#    label_mode='int',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=99,
)

# We can get a deeper understanding of the parameters by reading over the documentation.

# Notice that we have a parameter of class_names that is assigned to CLASS_NAMES. In order to use this parameter as is
#we have to initialize CLASS_NAMES above our code and assign it a list of values. (see line 62)

# Now we will configure our validation data. 

# We will do the same thing we did with the training dataset.

# The difference is going to be the batch size parameter which we will use like this, 
#batch_size=CONFIGURATION["BATCH_SIZE"]

# And we will do the same for the imgaine size which we will use like this, 
#image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    labels='inferred',
#    label_mode='int',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
)

# Now we have to initialize CONFIGURATION above with the other variables and specify the batch size, which is going
#to be 32, and the image size, which is going to be 256.

# From here we will look at our dataset.

# This is how we will do that and run it to see our data.

# Note that we used the .take function to specify that we only want to get back one batch.

for i in val_dataset.take(1):
    print(i)

# Notice that we get this back before we get back our actual data.

# Found 6799 files belonging to 3 classes.
# Found 2278 files belonging to 3 classes.

# Note the first number (6799) is the number of images in our training dataset. And the second number (2278) is
#the number of images in our validation dataset.

# We are only interested in looking at the labels at this point which appear at the end of our dataset.

# Notice that the labels are between 0 and 2, that is because we only have 3 classes.

# Now, instead of having our label mode set to int, we will change it to categorical, and then run it to see
#the difference.

# Now we will move on to visualize our data.

# To do this we will use this plt code.

# This will take a figure method and get a parameter which specifies the figure size.

# Our figsize will be 12 by 12

plt.figure(figsize=(12,12))

# Next we will use a for loop to access the images in our train dataset, but we can use the same process for the
#val dataset. We will still use the .take to specify we only want 1 batch.

# Then we will use a in range method to specify that we only want the first 16 images from this train dataset.

# And inside our for loop we will have a sup plot, which will be a 4 by 4 and i+1

# Now that we have that we will do an image show method which will take as a parameter images.

# And then we use the indexer inside that parameter to select a specific image.

# Lastly inside that parameter, we will divide all of the pixels by 255

# The next step will be to plot out a title using a plt.title method.

# This method will take as a parameter the labels, which we will also index into to specify the labels we want.

# We will also be using the tf.argmax with our labels parameter.

# In this labels parameter we will also specify an axis of 0.

# The last piece of this parameter, which will go between the last two parentheses, is a numpy method.

# Just to be on the safe side we will also use the plt.axis method to specify we want the axis off

# Now we will run this to see our images.

for images, labels in train_dataset.take(1):
    for i in range(16):
        ax = plt.subplot(4,4, i+1)
        plt.imshow(images[i]/255.)
        #plt.title(tf.argmax(labels[i], axis=0).numpy())
        plt.title(CLASS_NAMES[tf.argmax(labels[i], axis=0).numpy()])
        plt.axis("off")
        plt.show()


# Notice that running the data this way we get use back numerical classifications as labels.

# We will modify the code to return the actual word classifications instead.

# To do this we will midify our plt.title by making use of our CLASS_NAMES variable we created.

# See line 181 for modification.

# Notice that when we run our train dataset now we will get back the actual word classifications for our labels.

# At this point our dataset is now ready for training.

# First, we will add the prefetch parameter to our train dataset method for more efficient usage.

# This is how we will do that.

train_dataset = (train_dataset.prefetch(tf.data.AUTOTUNE))

# For now we are not going to include the batching because we've alrady included the batching in our code with the
#the batch sizes that we added.

# Now we do the same thing for he validation dataset.

# That will look like this.

validation_dataset = (val_dataset.prefetch(tf.data.AUTOTUNE))

# Now we are ready to build our model.

# To start we will use a piece off code from a previous model that will do the same thing for us here in this model.

# This is the code that we will reuse.

resize_rescale_layers = tf.keras.Sequential([
    Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    Rescaling(1./255)
])

# Note that we could resize and rescale our data before we train our data, this way the model will
#train the resized and rescaled data when it does train.

# Another way we can do it is, instead of passing in the resized and rescaled data into our model first before
#training, what we could do is pass the data into the model directly, and then carry out the resizing and rescaling
#in the model as a layer in the model.

# Doing this is great for deployment because when we have to deploy this kind of system, we no longer want to
#resize it again. So all we do is just pass in the image and then the model will take care of the resizing and 
#rescaling on its own.

# 