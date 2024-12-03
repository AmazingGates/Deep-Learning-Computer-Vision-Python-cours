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


# We can now test out the splitting on our own dataset.


# We will reload this.

dataset, dataset_info = tfds.load("malaria", with_info=True, as_supervised=True, shuffle_files=True, 
                                  split=["train"])

# Then we will run our for loop again.

for data in dataset[0].take(4):
    print(data)


# Next we will rerun splits function.    

def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    dataset_size = len(dataset)
    train_dataset = dataset.take(int(TRAIN_RATIO*dataset_size))

    val_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))

    val_test_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))
    val_dataset = val_test_dataset.take(int(VAL_RATIO*dataset_size))

    test_dataset = val_test_dataset.skip(int(VAL_RATIO*dataset_size))

    return train_dataset, val_dataset, test_dataset


# Now we can modify this. We will convert our Ratios back to their original proportions. 0.8, 0.1, 0.1.

TRAIN_RATIO = 0.8
VAL_RATIO= 0.1
TEST_RATIO = 0.1


train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)  

print(list(train_dataset.take(1).as_numpy_iterator()), list(val_dataset.take(1).as_numpy_iterator()), 
      list(test_dataset.take(1).as_numpy_iterator()))

# Note: If we ran it like this we would run into errors.

# That is because our dataset is actually made up of a list.

# This list is made up of the dataset and the types.

# So when we are running this, we're actually taking all the list data.

# So what we want to do is just pick out the dataset.

# And to do that, all we need to do is specify that we are picking out just the dataset by adjusting the assigned
#splits function like this splits(dataset[0]) (see line 50)

# Also, before running this let's make sure we take out just 1 element, because running the full dataset is going to be
#very time comsuming. We will do this by secifying take(1) for each dataset. (see line 52)

# Notice our results now a lot more precise.

# This is the image the lable has given us.

#[(array([[[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],
#
#        ...,

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]]], dtype=uint8), 1)] #[(#array([[[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],
#
#        ...,

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]]], dtype=uint8), 0)] #[(#array([[[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],
#
#        ...,

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]]], dtype=uint8), 1)]

# Notice that we are returned all of the validation test sets, which are known empty lists.

# 