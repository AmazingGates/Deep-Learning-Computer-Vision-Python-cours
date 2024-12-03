# Here we will be going over the process of Training our Convolutional Networks

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

model = tf.keras.Sequential([
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

model.summary() 


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


model.compile(optimizer = Adam(learning_rate=0.01),
              loss = BinaryCrossentropy(),
              metrics = "accuracy")

# Now that we have compiled our model, we will go ahead and start the training process.

# We will train our model in a similar way that we have previuously done.

# We'll use our trained data and our validation data.

# Let's also reduce the learning rate from 0.1 to 0.01. (see line 147)

history = model.fit(train_dataset, validation_data = val_dataset, epochs = 3, verbose = 1)
#history = lenet_model.fit(train_dataset, epochs = 3, verbose = 1)

# When we run the model like this, it will throw an error because our train dataset and val dataset have different
#shapes.

# To fix this, we will make sure that everything we do for the train dataset we do for the val dataset.

# This will ensure that our train dataset and val dataset have the same pre-processing.

# We will also do the samething for the test dataset.

# See lines 76 - 81 for updated code.

# So now that everything is updated, we have been training for awhile and we notice that we are getting very poor
#results.


# Output 689/689 [==============================] - 5560s 8s/step - loss: 0.0015 - val_loss: 3.6432e-05  Epoch 1

# Notice we have a loss of 0.0012 and val_loss: 2.5828e-05 for Epoch 1 on our training model.

# Output 689/689 [==============================] - 2479s 4s/step - loss: 2.1689e-05 - val_loss: 1.2851e-05 Epoch 2

# Notice we have a loss of 2.1689e-05 and val_loss: 1.2851e-05 for Epoch 2 in our training model.

# Output 689/689 [==============================] - 6773s 10s/step - loss: 9.0638e-06 - val_loss: 6.3366e-06 Epoch 3

# Notice we have a loss of loss: 9.0638e-06 and val_loss: 6.3366e-06 for Epoch 3 in our training model.

# Note: These 3 Epochs were generated using standard models.

# Next we will Train for 3 Epochs using the lenet_model, and then compare the differences.

# Lenet_model Output 689/689 [==============================] - 7425s 11s/step - loss: 9.8780e-04 - val_loss: 2.0195e-05

# This is the first Epoch of our lenet_model training.

# Notice the value of the loss, loss: 9.8780e-04, and the validation, val_loss: 2.0195e-05

# Lenet_model Output 689/689 [==============================] - 7706s 11s/step - loss: 1.2559e-05 - val_loss: 7.7248e-06

# This is the second Epoch of our lenet_model training.

# Notice the value of the loss, loss: 1.2559e-05, and the validation, val_loss: 7.7248e-06

# Lenet_model Output 689/689 [==============================] - 2288s 3s/step - loss: 5.5146e-06 - val_loss: 3.8981e-06

# This is the third Epoch of our lenet_model training.

# Notice the value of the loss, loss: 5.5146e-06, and the validation, val_loss: 3.8981e-06

# Notice that by the third Epoch, our model is training quicker, and the difference between the value loss and value
#validation is much closer.

# Another thing we could do to make the debugging faster is take off the validation_data=val_dataset.

# We don't want to lose the original code so we'll create an alternative code without the validation_data = val_dataset
#(see line 165 for commented out original and line 166 for updated version without the validation_data = val_dataset)

# Also, we are going to change all of the "sigmoid" activations into "relu" activations, except the Dense 1 layer,
#that will remian a "sigmoid" activation. (See lines 85 - 94)

# We will also reduce the size of the kernel size from 5 to 3. (See lines 84 and 88)

# Lastly we will add batch normailzation to our model.

# In batch normalization, all values belonging to the same batch are standardized. (See lines 86, 90, 96, 98)

# Now we can re-compile our model with the updated data.

# These are the 3 Epochs we trained with the lenet_model, with the reduced kernel_size, with the "relu" activation,
#without the validation_data=val_dataset and the BatchNormalization.

# Output 689/689 [==============================] - 1961s 3s/step - loss: 0.0169

# Notice that our first Epoch has a training time of 3 seconds per step, with an overall training time of 1,961
#seconds, and a loss value of 0.0169

# Output 689/689 [==============================] - 1874s 3s/step - loss: 4.2378e-05

# Notice that our second Epoch has a training time of 3 seconds per step also, with an overall training time of
#1,874 seconds, and a loss value of 4.2378

# Output 689/689 [==============================] - 1891s 3s/step - loss: 1.4514e-05

# Notice that our third Epoch has a training time of 3 seconds per step also, with an overall training time of
#1,891 seconds, and a loss value of 1.4514.

# Note: We were supposd to use 100 Epochs to train our data on, but because of computational time constraints
#we had to reduce it down to 3. With more Epochs we most likey would have seen better training time and more
#accuracy.

# Also note that the model did train quicker without the validation_data=valdataset, with an average of 3 seconds per 
#training step for each Epoch.

# Now that we have compiled our new model, the next thing we are going to do is add a metrics = accuracy to our compile
#parameters. (See line 155)

# This will display the actual accuracy of our models performance.

# So when training we'll be able to see how the loss and accuracy will evole.

# A models accuracy is equal to the total number of times that model predicted an output correctly divided by the total
#number of predictions.

# This means that if we have a model X, for example, and then we have another model Y, and we allow these two models
#to carry out let's say, 1,000 predictions, 

# So that's 1,000 predictions for model X, and 1,000 predictions for model Y.

# Now if model X does 800 correct predictions, then its accuracy is 800 divided by 1,000. Which would give us a fraction
#of 0.8. So model X would be traiing at an 80% accuracy.

# Now, if we have model Y, which does 980 correct predictions, we would have 980 divided by 1,000. Which would give us
#a fraction of 9.8. So model Y would be traing at an 98% accuracy.

# And in this case we would see that model Y out performs model X.

# It should be noted that the accuracy as a performance metric isn't always the best choice of a performance metric
#when it comes to classification problems as others like the precision, the recall, the F1 score, and many others exist.

# For now we'll use the accuracy, and later on we will look at the other metrics which we could use when we're
#dealing with classification problems.

#Epoch 1/3
# 689/689 [==============================] - 1827s 3s/step - loss: 0.0178 - accuracy: 0.9957

# Notice that the accuracy for our first epoch is 99% with a 3 second per step training time.

#Epoch 2/3
#689/689 [==============================] - 1812s 3s/step - loss: 4.8146e-05 - accuracy: 1.0000

# Notice that the accuracy for our second epoch is 100% with a 3 second per step training time.

#Epoch 3/3
#689/689 [==============================] - 1858s 3s/step - loss: 1.6559e-05 - accuracy: 1.0000

# Notice that the accuracy for our second epoch is 100% with a 3 second per step training time.
# Note: Our model was able to reach 100% accuracy so quick mainly because of the low number of predictions it had 
#to make. Remember that we are only using 16 images to train so take makes it easier for our model to make 
#predictions as opposed to a model with many different predictions to make.

# Next we will plot his model out using the alogorithm below, Then we will plot out the Accuracy.

# This will give us two plots. One for our loss, and one for our accuracy


# Model Loss Plot

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss","val_loss"])
plt.show()

# We see that the training and validation losses both keep dropping.


# Model Accuracy

# Here we notice that the Accuracy tools keeps increasing.

# Though the training accuracy is slightly greater than that of the validation accuracy.

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["train_accuracy","val_accuracy"])
plt.show()

# Epoch 1/3
# 689/689 [==============================] - 2587s 4s/step - loss: 0.0154 - accuracy: 0.9960 - 
#val_loss: 8.6559e-05 - val_accuracy: 1.0000

# Epoch 2/3
#689/689 [==============================] - 2239s 3s/step - loss: 4.6386e-05 - accuracy: 1.0000 - 
#val_loss: 2.4309e-05 - val_accuracy: 1.0000

# Epoch 3/3
#689/689 [==============================] - 2411s 3s/step - loss: 1.6049e-05 - accuracy: 1.0000 - 
#val_loss: 1.0633e-05 - val_accuracy: 1.0000
