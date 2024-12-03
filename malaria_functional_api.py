# In this section we will be going over and understanding Functional API's.


# Here we will be looking at different ways of creating models other than the sequential API which we've seen so far.

# In this section we'll also look at callable models.

# We'll look at building models via sub-classing.

# We'll also look at building our own custom layers.

# Previously in this course, we said that there are three ways in which models are built in tensorfow.

# Those are the Sequential API, The Functional API, and Model Sub-Classing.

# As of this point, we've been using the sequential API.

# We may ask ourselves why we may need to use a different method in creating tensorflow models when so far we've
#reached close to a 99% training accuracy and around a 95% test accuracy.

# As we may have noticed, so far all of the models we've been building have taken this kind of structure, where we
#have an input, we have the first layer, then the next layer, which has been stacked in a sequential manner, right up
#until the last layer, then we have the output.

#   ______        ______        ______        ______
#  |      |      |      |      |      |      |      |
#  |      |      |      |      |      |      |      |
#  |      |      |      |      |      |      |      |
#  |      | ---> |      | ---> |      | ---> |      |
#  |      |      |      |      |      |      |      |
#  |      |      |      |      |      |      |      |
#  |______|      |______|      |______|      |______|

# So the question we could ask ourselves is, what if we had a model which takes in let's say, two inputs, and has
#three outputs, for example.

# These kinds of models are very popular in deep learning and we will take a closer look at them.

# But before getting there, we could just imagine a problem where instead of classifying whether we have a non-parasitic
#cell, or a parasitic cell, we wanna know the exact position of that parasitic cell or in general that cell in the image.

# We would find that we would have one output which classifies whether it's parasitic or not.

# So our first output would be parasitic or uninfected.

# The second output would give us the exaact position of the cell in the image.

# So we can see how we would easily get two outputs from this.

# We can't really perform this action with a sequential API.

# So that's why working with a functional API is very important.

# The next point is we'll be able to create more complex model with the functional API.

# So there is a model known as the ResNet which is very popular in deep learning for computer vision.

# Now a ResNet structure will look something like this.


#                     _____________
#                    |             |
#   ______        ___|__        ___|__        ______        ______
#  |      |      |      |      |      |      |      |      |      |
#  |      |      |      |      |      |      |      |      |      |
#  |      |      |      |      |      |      |      |      |      |
#  |Input | ---> |Layer | ---> |Layer | ---> |Layer | ---> |Output|
#  |      |      |      |      |      |      |      |      |      |
#  |      |      |      |      |      |      |      |      |      |
#  |______|      |______|      |______|      |______|      |______|


# We have this model, where one layer gets passed into the next layer.

# The layers are then concatenated before being passed into the next layer.

# And so these kinds of structures, or models, could not be built with sequential API, and hence the need for,
#functional API's.

# And then the last reason why we are gonna be using the functional API is the fact that we coud use shared layers.

# With shared layers, we could have a layer, or particular layer in our model which already has a predefined way of
#encoding information.

# So when we pass information, let's say we have an input one for example, when we pass in this input one,
#our first layer, or, encoder, produces an output which is gonna be different from when we pass in another input, I-2.

# But the way it produces these outputs is in a very thoughtful manner.

# So we could have I-1, I-2, I-3, which all share that first layer, and then we have other layers of the model which
#follow on.

# With that said, we'll look at how to create functional API's.

# Before starting with the creation, we are going to import some classes.

# We'll start by importing the Input class, (See end of line 106)

# And then from tensorflow keras models, we're going to import Model. (See line 105)

import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import Accuracy, RootMeanSquaredError
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
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)


# We now have acess to the func_input, since we're using the functional API, this is how we call that.

# We have the func_input, and then we have Input, which we will call. 

# Input takes in the shape.

# We're going to copy the exact shape we have in the sequential API.

# At this point we could start stacking up all these different layers that we had stacked up when we were using
#the sequential API.

# We started with the Conv2D right up until the dense layer


func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image") 

x = Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")(func_input)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides= 2, padding = "valid")(x)

x = Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")(x)
x = BatchNormalization()(x)
output = MaxPool2D(pool_size=2, strides= 2, padding = "valid")(x)

#x = Flatten()(x)

#x = Dense(1000, activation = "relu")(x)
#x = BatchNormalization()(x)

#x = Dense(100, activation = "relu")(x)
#x = BatchNormalization()(x)

#x = Dense(1, activation = "sigmoid")(x)

#func_output = Dense(1, activation = "sigmoid")(x)

# Once we get to the end, we are going to create the lenet model from this.

# So we'll have lenet_model equal model, which will import it. And then we will pass in the func_input as a parameter.

#lenet_model = Model(func_input)

# Now let's say we have func_input, and we  have func_output (see line 214). (line 214 was nodified from line 212).

# We would modify our lenet_model to take as parameters the input and the output. We will also give it a name, which
#will be Lenet Model

feature_extractor_model = Model(func_input, output, name = "Feature_Extractor")

# Now we've created our lenet model.

# And then from here we could simply do lenet_model.summary

feature_extractor_model.summary()

# Now we'll notice that we should have exactly the same summary as we had with sequential API.

# Now let's run it and see what we get.

# We have the same model information that we have form our sequential API.

# So basically what we've done here is we've created this this model which we've previously done with the sequential
#API.

# Now that we've got this, we'll see that we have to change absolutely nothing from our code.

# So now we're going to compile our model without changing anything.

# We'll use the same lenet_model.

y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 1]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
#print(bce(y_true, y_pred))

#lenet_model.compile(optimizer = Adam(learning_rate=0.01),
#              loss = BinaryCrossentropy(),
#              metrics = "accuracy")

#history = lenet_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)

# Output

# 689/689 [==============================] - 2290s 3s/step - loss: 0.0197 - accuracy: 0.9950 
#- val_loss: 1.8456e-04 - val_accuracy: 1.0000


# Here is what we get as aresult.

# Now coming back to our model, we'll see that we have a feature extraction. (See lines 198 - 204)

# So these Conv layers are responsible for extracting useful features from the images.

# And the last layers are responsible for correctly classifying whether the image is parasitic or not.
#(See lines 208 216)

# With that said, we could build a model known as feature extractor.

# It will be constructed exactly the same way as we have our model constructed so far, the only differnce is we will
#leave out the flatten() layer, and the dense() layers.

# Note: Instead rewritting a whole new copy of the same code, we will just comment out lines 208 - 212 in the original
#code to indicate that we are only using the extractor.

# We will aslo comment out line 216, which was our old output, and modify line 202 to be our new output.

# Note: We must remember to modify the func_output to just output in our lenet_model = Model() (See line 229)

# Note: We will also cahnge the label of our model in that same Model() to "Feature Extractor". (See line 229)

# Lastly, we will change the names of our Model and the Summary to feature_extractor_model. (See lines 229 and 235)

# Now we will run it without the history.fit or the model.compile and see the results (See lines 258 - 262)

# Output

#Model: "Feature_Extractor"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# Input Image (InputLayer)    [(None, 224, 224, 3)]     0
#
# conv2d (Conv2D)             (None, 222, 222, 6)       168
#
# batch_normalization (BatchN  (None, 222, 222, 6)      24
# ormalization)
#
# max_pooling2d (MaxPooling2D  (None, 111, 111, 6)      0
# )
#
# conv2d_1 (Conv2D)           (None, 109, 109, 16)      880
#
# batch_normalization_1 (Batc  (None, 109, 109, 16)     64
# hNormalization)
#
# max_pooling2d_1 (MaxPooling  (None, 54, 54, 16)       0
# 2D)
#
#=================================================================
#Total params: 1,136
#Trainable params: 1,092
#Non-trainable params: 44
#_________________________________________________________________
#tf.Tensor(4.9340706, shape=(), dtype=float32)

# Notice our output (See line 316)

# At this point, instead of writting all of this we can simplify this code even further.

# We will call our feature_extractor_model() that we created, and pass into it the func_input as a parameter.

# Once we do this we will comment out lines (198 - 204) because all of this information is within our feature extractor
#model.

# Note that we will uncomment lines 206 -216 to run with our new feature extractor model.

# Notice that our model now looks like a function.

# That is because Tensorflow models are callable, just like the layers.

# And as we could see, the feature extractor model could be seen as a layer, just like the dense layers, or the 
#batchnormalization layers, and all the other layers.


#feature_extractor_seq_model = tf.keras.Sequential([
#    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
#    Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
#    BatchNormalization(),
#    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),
#
#    Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
#    BatchNormalization(),
#    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),
#
#])

#feature_extractor_seq_model.summary()


x = feature_extractor_model(func_input)
#x = feature_extractor_seq_model(func_input)

x = Flatten()(x)

x = Dense(1000, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(1, activation = "sigmoid")(x)

func_output = Dense(1, activation = "sigmoid")(x)

lenet_model_func = Model(func_input, func_output, name = "Lenet_Model")

lenet_model_func.summary()

# After all of this, we have sucessfully used functional API to build our model. 

# We condensed all of the Conv2d and MaxPooling2d Information, (the Extractor), into the x = feature_extractor_model() 
#and we were able run that with our classifying information, (the Dense Layers), and return our entire Lenet Model
#as a whole.

# In subsequent sections, we'll build even more complex models using these functional API models, where we're 
# #going to use shared layers.

# We're going to have mutiple inputs, multiple outputs and models where we're going to have even more complicated
#model configurations.

# It's important to note that we could mix up the functional API model creation style with that of the sequential
#API 

# So here, instead of having our feature extractor created like this, we are going to create it using the sequential
#API.

# This is how we can do this.

# Notice that we added our extractor to our sequential model and to our summary.

# Notice that we have the same exact whole model with the Sequential Model as we do with the Extractor Model.

# Note: We added the x = feature_extractor_seq_model(func_input) to our classifying data and commented out the original
#(see line 360) to show that they are the same.

# Also note that code below is commented out because we copied it and initialized above the x = feature_extractor_seq_model(func_input)
#code so that we could get access to the x = feature_extractor_seq_model(func_input) function. But it's the exact code
#we used.

#feature_extractor_seq_model = tf.keras.Sequential([
#    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
#    Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
#    BatchNormalization(),
#    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),
#
#    Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
#    BatchNormalization(),
#    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),
#
#])

#feature_extractor_seq_model.summary()

# This shows us that we could mix up the different types of creation models.


# From this point we'll look at the model SubClassing

# It's important to note that model subclassing permits us to create recursively composable layers and models.

# Now, what does that mean?

# This means that we could create a layer where its attributes are other layers, and this layer tracks the weights
#and biases of the sub layers.

# Before making an example, let's get this import.

# We're going to import layer from layers (See end of line 106)

# Now we can create our model using the Model subclassing