# Here we will be going over our Model Evaluation and Testing Process.

import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import keras.layers 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization
import keras.losses 
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

lenet_model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    tf.keras.layers.Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
    BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
    BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    Flatten(),

    Dense(1000, activation = "relu"),
    BatchNormalization(),
    Dense(100, activation = "relu"),
    BatchNormalization(),
    Dense(1, activation = "sigmoid"),
])

lenet_model.summary() 


y_true = [0,]
y_pred = [0.8, ]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) 

y_true = [0,]
y_pred = [0.02, ]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) 


y_true = [0,]
y_pred = [0.2, ]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) 


y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 0]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) 


y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 1]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred)) 


y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 1]
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bce(y_true, y_pred)
print(bce(y_true, y_pred)) 


y_true = [0, 1, 0, 0]
y_pred = [0.6, 0.51, 0.94, 1]
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred)
print(bce(y_true, y_pred))


lenet_model.compile(optimizer = Adam(learning_rate=0.01),
              loss = BinaryCrossentropy(),
              metrics = "accuracy")


history = lenet_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)
#history = lenet_model.fit(train_dataset, epochs = 3, verbose = 1)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss","val_loss"])
plt.show()


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["train_accuracy","val_accuracy"])
plt.show()

# Here we will eveluate our model using the lenet_model.

# Before we can evaluate our dataset we have to alter our test dataset shape so that everything is compatible.

# We can do that by performing the action below.

test_dataset = test_dataset.batch(1)

# Now we can evaluate our model

lenet_model.evaluate(test_dataset)

# Here's what we get once we evalutae our model.

# On data that this model has never seen, these are the results.

# Output 2757/2757 [==============================] - 171s 60ms/step - loss: 0.0549 - accuracy: 0.9993 

# Note that we could continue with this training.

# So we could train for more epochs as oppossed to what we have now.

# Many scientist have made the remark that they have forgotten to stop training, and then comback to notice that 
#that they've gotten an even better performing model, because they allowed the model to train for a longer time 
#frame.

# After evaluating our model, let's look at how to do model predictions.

# Now the whole idea of model predictions makes sense since we have trained our model on inputs and outputs.

#                     __________
#                    |          |
#     Input -------> |   Model  | <-------- Output
#                    |__________|


# And now we want to pass in an input and let our model automatically come up with the output.

#                     __________
#                    |          |
#     Input -------> |   Model  | --------> Output
#                    |__________|

# That is to say whether the image contains an infected cell, or uninfected cell.

# With that said, all we need to do is run this model predict code.

# So we have our predict method, and then we pass in our data.

# And then we specify that we want to take 1 value from the dataset using the .take()

# Now we can run our model.predict

print(lenet_model.predict(test_dataset.take(1))[0][0])

# Output 0.00010284152

# Now we'll define this method parasite or not, which is defined such that if we have an input X, then if that x is 
#less than 0.5, consider that we have parasitized cell, and if it's greater than or equal to 0.5, then it's an
#uninfected cell.

# Recall that the way that the data was created was such that parasitized was 0, and then uninfected was 1. 

# So we have a threshold value of 0.5

# This threshold value is defined now such that every value less than it is considered parasitized, and everything
#greater is considered uninfected.


def parasite_or_not(x):
    if(x<0.5):
        return str("P")
    else:
        return str("U")
    
p = 0
unin = 1

# If we now wanted, we could now alter our code to be even more percise and include our parasite or not function.

parasite_or_not(lenet_model.predict(test_dataset.take(1))[0][0])

# Output


# We are going to do the testing on 9 different elements.

# So we specify our 9 with a .take(), and then we do our subplots.

# First we do the imshow(), and we specify [0] because we don't want batch dimensions.

# And then we have our title. In the title we the actual output, and we have the models predicted output.

# Now we can run the code below.

for i, (image, label) in enumerate(test_dataset.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[0])
    plt.title(str(parasite_or_not(label.numpy()[0])) + ":" + str(parasite_or_not(lenet_model.predict(image)[0][0])))
    plt.axis('off')
    plt.show()


#1/1 [==============================] - 14s 14s/step
#1/1 [==============================] - 9s 9s/step
#1/1 [==============================] - 0s 365ms/step
#1/1 [==============================] - 0s 94ms/step
#1/1 [==============================] - 0s 94ms/step
#1/1 [==============================] - 0s 88ms/step
#1/1 [==============================] - 0s 86ms/step
#1/1 [==============================] - 0s 78ms/step
#1/1 [==============================] - 0s 78ms/step
#1/1 [==============================] - 0s 94ms/step
#1/1 [==============================] - 0s 94ms/step

# Notice that our images are label P:P, U:U, or U:P

# U:U means the actual is uninfected, and the predicted is uninfected.

# P:P means the actual is parasitized, and the predicted is parasitized.

# U:P means the actual is uninfected, but the predicted is parasitized.

# 