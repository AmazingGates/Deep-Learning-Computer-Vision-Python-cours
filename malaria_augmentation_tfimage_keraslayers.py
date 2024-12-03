import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer, Dropout
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L2, L1
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns


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
dropout_rate = 0.2
regularization_rate = 0.01

def resizing(image, ladel):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)), label


train_dataset = train_dataset.map(resizing)
print(train_dataset)

print(resizing(image, label))


for image, label in train_dataset.take(1):
    print(image, label)
    


IM_SIZE = 224
def resize_rescale(image, ladel):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0, label

def augment(image, label):

    image, label = resize_rescale(image, label)

    image = tf.image.rot90(image)
    image = tf.image.adjust_saturation(image, saturation_factor = 0.3)
    image = tf.image.flip_left_right(image)
    return image, label


test_dataset = test_dataset.map(resize_rescale)
#train_dataset

BATCH_SIZE = 32

train_dataset = (
    train_dataset
    .shuffle(buffer_size=8, reshuffle_each_iteration=True)
    .map(augment)
#    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .shuffle(buffer_size=8, reshuffle_each_iteration=True)
    .map(resize_rescale)
#    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

class NeuralearnDense(Layer):
    def __init__(self, output_units, activation):
        super(NeuralearnDense, self).__init__()
        self.output_units = output_units
        self.activation = activation

    def build(self, input_features_shape):
        self.w = self.add_weight(shape = (input_features_shape[-1], self.output_units), initializer="random_normal",
                                       trainable = True)
        self.b = self.add_weight(shape = (self.output_units,), initializer="random_normal", trainable=True)

    def call(self, input_features):
        pre_output = tf.matmul(input_features, self.w) + self.b
        if(self.activation == "relu"):
            return tf.nn.relu(pre_output)
        
        elif(self.activation == "sigmoid"):
            return tf.math.sigmoid(pre_output)
        
        else:
            return pre_output
    
lenet_custom_model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu", kernel_regularizer = L2(0.01)),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),
    Dropout(rate = dropout_rate),

    Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu", kernel_regularizer = L2(0.01)),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    Flatten(),

    NeuralearnDense(1000, activation = "relu"),
    BatchNormalization(),
    Dropout(rate = dropout_rate),

    NeuralearnDense(100, activation = "relu"),
    BatchNormalization(),

    NeuralearnDense(1, activation = "sigmoid"),
])

lenet_custom_model.summary() 

metrics = metrics = [TruePositives(name="tp"), FalsePositives(name="fp"), TrueNegatives(name="tn"), FalseNegatives(name="fn"),
                     BinaryAccuracy(name="ba"), Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]

lenet_custom_model.compile(optimizer = Adam(learning_rate=0.01),
              loss = BinaryCrossentropy(),
              metrics = metrics)

#history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)

test_dataset = test_dataset.batch(1)

test_dataset

lenet_custom_model.evaluate(test_dataset)

labels = []
input = []

for x,y in test_dataset.as_numpy_iterator():
    labels.append(y)
    input.append(x)

print(labels)

labels = np.array([i[0] for i in labels])

print(labels)

print(np.array(input).shape)
print(np.array(input)[:,0,...].shape)


predicted = lenet_custom_model.predict(np.array(input)[:,0,...])
 

print(predicted) 
print(predicted.shape)
print(predicted[:,0].shape)
print(predicted[:,0])
  

threshold = 0.5
# threshold = 0.25
# threshold = 0.75

cm = confusion_matrix(labels, predicted > threshold)
print(cm)


plt.figure(figsize=(8,8))

sns.heatmap(cm, annot = True,)
plt.title("Confusion matrix - {}".format(threshold))
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

fp, tp, thresholds = roc_curve(labels, predicted)
print(len(fp), len(tp), len(thresholds))

plt.plot(fp, tp)
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.grid()

skip = 20

for i in range(0, len(thresholds), skip):
    plt.text(fp[i], tp[i], thresholds[i])

plt.show()

class LossCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        print("\n For Epoch Number {} the model has a loss of {} ".format(epoch + 1, logs["loss"]))

    def on_batch_end(self, batch, logs):
        print("\n For Batch Number {} the model has a loss of {} ".format(batch + 1, logs))

csv_callback = CSVLogger(
    "logs.csv", separator=",", append=False
)

csv_callback = CSVLogger(
    "logs.csv", separator=",", append=True
    )

es_callback = EarlyStopping(
    monitor = "val_loss", min_delta=0, patience=2, verbose=1,
    mode = "auto", baseline=None, restore_best_weights=False
)


def scheduler(epoch, lr):
  if epoch < 3:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

scheduler_callback = LearningRateScheduler(scheduler, verbose = 1)


checkpoint_callback = ModelCheckpoint(
    "checkpoints/", monitor="val_loss", verbose=0, save_best_only=True,
    save_weights_only=False, mode="auto", save_freq=3,   
)


plateau_callback = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.1, patience=2, verbose=1
)

#history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1, 
#                                 callbacks = [csv_callback])

history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)                                 


# In this section we will be going over Augmentation with TF.Image and Keras Layers

# The first method we will be looking at is the TensorFlow image model (tf.image), which is made up of these different
#functions.

# tf.image.adjust_brightness
# tf.image.adjust_contrast
# tf.image_adjust_gamma
# tf.image_adjust_saturation
# tf.image_flip_left_right
# tf.image.flip_up_down
# tf.image.rot90

# These are just a few of the functions, but we could check out the rest in the documentation.

# Using the tf.image is known to be a more flexible way of implementing data augmentation, as we could alter our
#input image with all these different functions given to us.

# The next method we will use is the keras layers, and although we are limited by the number of augmentation layers
#made available to us, this method permits us to carry out data augmentation more efficiently and hence speed up
#the training process.

# tf.keras.layers.RandomContrast
# tf.keras.layers.RandomCrop
# tf.keras.layers.RandomFlip
# tf.keras.layers.RandomRotation 
# tf.keras.layers.RandomTranslation 
# tf.keras.layers.RandomZoom 

# These are a few of the functions that are available to us.

# Now to find the balance between these two methods, we could implement our own custom keras layers.

# And that's exactly what we're going to do in this section.

# Now we'll be looking at how to implement data augmentation with tensorflow.

# So we will dive into tensorflow images.

# We'll start by using tf.image.

# Recall that in order to do data augmentation we're basicalling modifying the images while keeping the labels 
#fixed.

# We can go over the Overview in tf.image to get a better understanding of the tools we will be using.

# The fisrt method we will use is the visualize.

# We will start by defining it, and passing two parameters, which are original, and augmented (which represent our
#image)

def visualize(original, augmented):
    #next we will have a subplot with 1 line and 2 columns which will occupy the first position in that two-column 
    #space
    plt.subplot(1,2,1)
    #Next we will have an image show, which takes the original.
    plt.imshow(original)
    #And then we repeat this for the second position and the augmented image.

    plt.subplot(1,2,2)
    plt.imshow(augmented)


# So that's our visualized method.

# Now what we'll do is get the origina image.

# So let's say that this is our original image, which is equal to an element we take from our dataset.

# And we'll just pick one element from our dataset.

# This should output a label so we'll include that here.

original_image, label = next(iter(train_dataset))

# So we have our original image here and now we'll work with our augmented image.

# And to obtain this augmented image we will modify our original image.

# To do this modification we are going to make use of the methods we have already seen.

# We'll start with the flip left right.

# With this method we can flip an image horizontally (left to right)

# All we need to do is pass the image.

# This is how we will implement the method.

#augmented_image = tf.image.flip_left_right(
#    original_image
#)

#augmented_image = tf.image.random_flip_up_down(
#    original_image
#)

#augmented_image = tf.image.rot90(
#    original_image
#)

#augmented_image = tf.image.adjust_brightness(
#    original_image, delta = 0.1
#)

#augmented_image = tf.image.random_saturation(
#    original_image, lower = 2, upper = 12
#)

#augmented_image = tf.image.central_crop(
#    original_image, 0.3
#)

augmented_image = tf.image.adjust_saturation(
    original_image, saturation_factor = 0.3)

# Now we have our augmented image.

# Now we can visualize this augmentation.

# Here we will visualize the original_image and the augmented_image

visualize(original_image, augmented_image)

# Now we can run this and see what we get.

# We should have two images. 

# The first one is our original and the second should be our augmented(modified original).

# This means that we got back datasets for both images.

# Now we can check on some other augmentation strategies.


# The next method we will look at is the tf.image.random_flip_up_down.

# This method will randomly flip an image vertically (upside down)

# Basically we can leave the code the same they we used for the tf.image.flip_left_right, but instead, we will
#will use the tf.image.random_flip_up_down.

# See line 344 for implementation.

# The next method we will look at is tf.image.rot90 degree rotation.

# This method will rotate an image 90 degress counter-clockwise.

# This is how we will implement that.

# See line 348 for implementation.

# The next method we will be looking at is the tf.image.adjust_brightness.

# This will adjust the brightness of our image.

# Note: With the adjust brightness, we have to pass a delta parameter to avoid error. This delta should be in the 
#range of negative 1 and 1. What the delta does is it basically adds up each and every pixel we have. For this 
#example we will use the value of 0.1.

# This method gives allows us to brighten our augmented version of our original image by adjusting the delta value.

# Here we will be looking at the tf.image.random_saturation

# This method allows us to randomly adjust the RGB saturation of our image. Also, this method has the two parameters 
#lower and upper. (lower = float. Lower bound for the random saturation factor),
#(upper = float. Upper bound for the random saturation factor.)

# Note: If the upper is higher than the lower we will get an error which is logical. So the lower value must 
#always be lower than the upper value in order to function properly.

# This is how we would implement that. (see line 356)

# The next method we will look at is the tf.image.central_crop

# This method crops the central region of the image(s).

# This function works on either a single image (image is a 3-D Tensor), or a batch of images (image is a 4-D Tensor)

# Note: The value that we pass as a parameter determines how much of the image gets cropped. (The lower the value 0.3,
#fro example, the more zoomed in we are of the center of the image, the higher the value 0.9, for example, zooms out
#from the image and may display so background, depending on the image and the size.)

# This is how we will implement this method. (see line 360)

# Now the way that we are going to integrate this augmentation in our data pipeline is going to be similar to the way
#we did the resize and rescale.

# So just like we used the map method, we are going to reuse this same method for augmentation.

# Let's supose we are going to add this code and then define our augment method.

# This is what we will have.

#def augment(image, label)

# And then what we'll do is we'll take this image, and for the image we're going to have it rotate 90 degrees.

#image = tf.image.rot90(image)

# Then next we are going to adjust the saturation by adjusting the saturation factor.

#image = tf.adjust_saturation(image, saturation_factor = 0.3)

# We will also flip the image using the tf.image.flip_left_right method.

# And lastly we will return the image and label.

# See line 81 - 85 for the process of the steps we took.

# Next we can can also include the resize, rescale.

# So we would have image and the label equal resize with rescale, which gets passed the image and the label.

# We're resizing and rescaling before doing the augmentation.

# Now we have the augment meta defined. (see line 83)

# Also, instead of passing resize_rescale into our train_dataset, we will pass in augment. (see line 90)

# The method we will run with the new augment data is the tf.adjust_saturation(original_image, saturation_factor = 0.3).
#(see line 374). This will help us visualize the augment function in action.

# Another thing we want to do before we do the mapping is perform a shuffle.

# So we'll modify the order in which we do things.

# See line 100 to see how we adding the map(augment) to be implemented after the .shuffle in our train dataset.

# We'll repeat the same process for the validation, but the only difference is that we are going to pass in the 
#resize_rescale for our map instead of the augment.

# See line 108 for implementation

# Notice that we are not augmenting the validation.

# Now that everything is set up we can now run our training. 

# We will do this without the callbacks for now.

history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)