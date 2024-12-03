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

# In this section we will go over the process of data preparation for our Malaria Diagnosis Model.

# In this data section we'll make use of a dataset contained in the tensorflow datasets module.

# From here we will choose the malaria dataset to work with.

# The Malaria dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells
#from the thin blood smear slide images of segmented cells.

# Now we will start the process of loading our data.

# This is how we will load our Malaria Data.

dataset, dataset_info = tfds.load("malaria", with_info=True)

# Now we will check our dataset information by using this formula.

print(dataset) # Output {'train': <_PrefetchDataset element_spec={'image': TensorSpec(shape=(None, None, 3), 
#dtype=tf.uint8, name=None), 'label': TensorSpec(shape=(), dtype=tf.int64, name=None)}>}

# Notice that clearly we see prefetched dataset.
 
# We have the image and we have the label.


# And we can also check the dataset info by using this formula

print(dataset_info) # Output

#tfds.core.DatasetInfo(
#    name='malaria',
#    full_name='malaria/1.0.0',
#    description="""
#    The Malaria dataset contains a total of 27,558 cell images with equal instances
#    of parasitized and uninfected cells from the thin blood smear slide images of
#    segmented cells.
#    """,
#    homepage='https://lhncbc.nlm.nih.gov/publication/pub9932',
#    data_dir='C:\\Users\\alpha\\tensorflow_datasets\\malaria\\1.0.0',
#    file_format=tfrecord,
#    download_size=337.08 MiB,
#    dataset_size=317.62 MiB,
#    features=FeaturesDict({
#        'image': Image(shape=(None, None, 3), dtype=uint8),
#        'label': ClassLabel(shape=(), dtype=int64, num_classes=2),
#    }),
#    supervised_keys=('image', 'label'),
#    disable_shuffling=False,
#    splits={
#        'train': <SplitInfo num_examples=27558, num_shards=4>,
#    },
#    citation="""@article{rajaraman2018pre,
#      title={Pre-trained convolutional neural networks as feature extractors toward
#      improved malaria parasite detection in thin blood smear images},
#      author={Rajaraman, Sivaramakrishnan and Antani, Sameer K and Poostchi, Mahdieh
#      and Silamut, Kamolrat and Hossain, Md A and Maude, Richard J and Jaeger,
#      Stefan and Thoma, George R},
#      journal={PeerJ},
#      volume={6},
#      pages={e4568},
#      year={2018},
#      publisher={PeerJ Inc.}
#    }""",
#)


# Next we will run a for loop to print out our data.

for data in dataset["train"].take(1):
    print(data) # Output

# {'image': <tf.Tensor: shape=(103, 103, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
## #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],
#
#       [[0, 0, 0],
##        [0, 0, 0],
#        [0, 0, 0],
## #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
##        [0, 0, 0],
#        [0, 0, 0],
## #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

##       ...,

#       [[0, 0, 0],
##        [0, 0, 0],
#        [0, 0, 0],
## #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
##        [0, 0, 0],
#        [0, 0, 0],
## #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
##        [0, 0, 0],
#        [0, 0, 0],
## #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]]], dtype=uint8)>, 'label': <tf.Tensor: shape=(), dtype=int64, numpy=1>}

# So now we have our data.

# Notice that we have 103 by 103 by three.
# That's our image

# And then if we scrool down we see the label, which is "1"

# Let's scrool back up and say we have four values, just as an example.

# To do this we will run the new code the same but change our .take value from 1, to 4.

for data in dataset["train"].take(4):
    print(data) # Output

# {'image': <tf.Tensor: shape=(103, 103, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
##        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
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

#       ...,

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]]], dtype=uint8)>, 'label': <tf.Tensor: shape=(), dtype=int64, numpy=1>}
#{'image': <tf.Tensor: shape=(106, 121, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       ...,

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]]], dtype=uint8)>, 'label': <tf.Tensor: shape=(), dtype=int64, numpy=1>}
#{'image': <tf.Tensor: shape=(139, 142, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       ...,

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]]], dtype=uint8)>, 'label': <tf.Tensor: shape=(), dtype=int64, numpy=0>}
#{'image': <tf.Tensor: shape=(130, 118, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       ...,

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]],

#       [[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
# #       ...,
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]]], dtype=uint8)>, 'label': <tf.Tensor: shape=(), dtype=int64, numpy=1>}

# So now we have four values, as oopposed to the one value we got back from .take(1)

# This shows us how we obtain images and the corresponding labels.

# We can now pass in more arguments like the shuffle files, as supervised, and the split.

# Here is a more elaborate explaination of how we can work with the split.

#   Split - Which split of the dataset to load(e.g. "train", "test", ["train", "test"], "train[80%]",...)
#If None, will return all splits in a Dict[Split, tf.data.Dataset]

# So we simply add as supervised equal true to our function, shuffle files equal true, and finally we specify
#the split.

# So we have our split that will be split equal ["train"]

# This is how we will run everything together.

dataset, dataset_info = tfds.load("malaria", with_info=True, as_supervised=True, shuffle_files=True, 
                                  split=["train"])

# Now that we have our training set, we'll separate them and run them separately.

# First we'll use the for loop to print data in the dataset like this

for data in dataset[0].take(4):
    print(data) # Output


# (<tf.Tensor: shape=(145, 148, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
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

#       ...,

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
#        [0, 0, 0]]], dtype=uint8)>, <tf.Tensor: shape=(), dtype=int64, numpy=1>)
#(<tf.Tensor: shape=(133, 127, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
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

#       ...,

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
#        [0, 0, 0]]], dtype=uint8)>, <tf.Tensor: shape=(), dtype=int64, numpy=1>)
#(<tf.Tensor: shape=(118, 118, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
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

#       ...,

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
#        [0, 0, 0]]], dtype=uint8)>, <tf.Tensor: shape=(), dtype=int64, numpy=0>)
#(<tf.Tensor: shape=(124, 121, 3), dtype=uint8, numpy=
#array([[[0, 0, 0],
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

#       ...,

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
#        [0, 0, 0]]], dtype=uint8)>, <tf.Tensor: shape=(), dtype=int64, numpy=1>)

# Notice that after we run this now, the shape has changed because the files are now shuffled.
# Note: If we leave the shuffle files parameter on false, we will get the same results every time we run it.

# In order for us to create the train validation and test set, we are going to make use of the take method
#and the skip method.

# So let's look at this example.

# We have this little dataset, we'll create it.

# And we want to print out the dataset after skipping seven values.

# This is how we will code the entire process.

dataset = tf.data.Dataset.range(10)
dataset = dataset.skip(7)
print(list(dataset.as_numpy_iterator())) # Output [7, 8, 9]

# Notice that with skip formula we are skipping the first 7 values in the values list with a range of 10.
# That's why we are returned the last 3 values.

dataset = tf.data.Dataset.range(10)
dataset = dataset.take(6)
print(list(dataset.as_numpy_iterator())) # Output [0, 1, 2, 3, 4, 5]

# Notice for the take formula, we are doing the opposite of the skip formula. You can see that we are only
#returning the first 6 values from a value list with a range of 10.

# Next we will use print(list(dataset.as_numpy_iterator())) after ever level.
# This will print out the dataset for the range and the take/skip separately.

# First we will define some percentages for train ration, validation ratio, and test ratio.

# We will also get the length of our dataset to make sure it is the correct range.

#TRAIN_RATIO = 0.8
#VAL_RATIO = 0.1
#TEST_RATON = 0.1

#dataset = tf.data.Dataset.range(10)
#print(list(dataset.as_numpy_iterator())) # Output [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#print(len(dataset)) # Output 10 

# Notice that we are returned the list with each element in our range of 10.
# Also notice that we are returned the length of that list, which is 10.

# Now that's set.

# To obtain the train dataset, all we need to do is simply take the first 80% of our dataset, using the .take
#function.

# This is how we will perform this.

# First we will define a dataset size

# Next we will muliptly 0.8/80% by our dataset size/length

# Lastly we will use the print(list(train_dataset.as_numpy_iterator())) to print out the return.

#dataset_size = len(dataset)
#train_dataset = dataset.take(int(0.8*dataset_size))
#print(list(train_dataset.as_numpy_iterator())) # Output [0, 1, 2, 3, 4, 5, 6, 7]

# Notice that using this .take method, we were returned the first 80% of the elements in our list of range 10.

# Next we will change the train ratio, validation ratio, and test ratio to see more examples.

#TRAIN_RATIO = 0.6
#VAL_RATIO= 0.2
#TEST_RATIO = 0.2

#dataset = tf.data.Dataset.range(10)
#print(list(dataset.as_numpy_iterator()))
#dataset_size = len(dataset)
#train_dataset = dataset.take(int(TRAIN_RATIO*dataset_size))
#print(list(train_dataset.as_numpy_iterator())) # Output [0, 1, 2, 3, 4, 5]

# Notice that we are returned the first 60% of the elements in our list of range 10.

# Also notice that we are left with the other 40% of elements remaining.

# Note: We did not change the length of the dataset, only the percentages on which we train, validate, and test
#on. 

# Now let's see how we can get this other 4 elements.

# To obtain those last 4 element, we will use these steps.

# So we're going to define our validation.

# And instead of taking the first 6 elements, we're going to skip the first 6 elements, this way we could get the last
#4 elements that we want.

#val_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))
#print(list(val_dataset.as_numpy_iterator())) # Output [6, 7, 8, 9]

# Notice that with this method we are returned the last 4 elements inside our list of range 10.

# But recall that VAL_RATIO is used on the 2 elements [6,7], as specified by the VAL_RATIO = 0.2

# So what we're going to do again, is after skipping and getting the last 4 elements, which essentially is made up
#of the VAL_RATIO set [6,7] and the TEST_RATIO set [8,9], we are now going to take out the first 2 elements of the
#the remaining 4 elements, which correspond to the VAL_RATIO set [6,7].

# To do that we'll use this method in the code we just wrote, val_dataset = val_dataset.take

# Notice that we are making our val_dataset equal to val_dataset.take.
# What are we taking?
# We are taking the first 2 elements, which could be obtained by having this int val.
# So we will add an int of our VAL Ratio times dataset size

#val_test_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))
#val_dataset = val_test_dataset.take(int(VAL_RATIO*dataset_size))
#print(list(val_dataset.as_numpy_iterator())) # Output [6,7]

# Notice that we are the 2 elements that make up our VAL_RATIO set [6,7]

# Now in order for us to get the last 2 elements that make up our TEST_RATIO [8,9],
#all we need to do is a skip.

# Notice that we are creating a new variable to store our data in, "val_test_dataset", on line 692 and 693.
# This is just for example purposes so that we break it apart and use it to target the TRAIN_RATIO elements that
#we want.

# Also notice that we will be using half the new variable to skip the 2 elements the make up our VAL_RATIO set [6,7]

#test_dataset = val_test_dataset.skip(int(VAL_RATIO*dataset_size))
#print(list(test_dataset.as_numpy_iterator())) # Output [8,9]

# Notice that we are returned the last 2 elements that make up our TEST_RATIO set [8,9]

# Now we have successfully obtained all of the elements in our list of range 10.

# We could always play around with the values to get a better feel for how things work.

# Let's suppose we have no VAL_RATIO for example.

TRAIN_RATIO = 0.6
VAL_RATIO= 0
TEST_RATIO = 0.2


#dataset = tf.data.Dataset.range(10)
print(list(dataset.as_numpy_iterator()))
dataset_size = len(dataset)
train_dataset = dataset.take(int(TRAIN_RATIO*dataset_size))
print(list(train_dataset.as_numpy_iterator())) 

val_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))
print(list(val_dataset.as_numpy_iterator())) 

val_test_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))
val_dataset = val_test_dataset.take(int(VAL_RATIO*dataset_size))
print(list(val_dataset.as_numpy_iterator())) # Output []

# Notice that because we specified that our VAL_RATIO is 0, we are returned an empty list when we try to retrieve
#the elements that make up our VAL_RATIO.

test_dataset = val_test_dataset.skip(int(VAL_RATIO*dataset_size))
print(list(test_dataset.as_numpy_iterator())) # Output [6,7,8,9]

# Notice that we are returned all 40% / last four elements in our list of range 10 to our TEST_RATIO, when previously
#we were only returned [8,9].
# This happened because we specified that the VAL_RATIO is 0, so by default, all the elements that were left
#out of the TRAIN_RATIO, which was 60% / 6 elements, were allocated to the TEST_RATIO.

# Now we can create a function from here.

# We'll call this function, the spits function.

# The splits function will take the dataset and ratios as parameters.

# Next we will add it's functionality, which will be everything we have just done.

# Note: We don't have have to define our dataset twice since it is already defined above in our code,
#so we can leave that part out.
# But we'll just comment it out here to show what we are referring to.

# We can also leave out the print statements.

# Lastly we will go ahead and return all of our datasets.

# So now we have this method that we've built.

# And from this, we have our datasets.  

# Before we run it, we will also define our datasets above and make them equal to the splits function and its
#parameters

def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    #dataset = tf.data.Dataset.range(10)
    dataset_size = len(dataset)
    train_dataset = dataset.take(int(TRAIN_RATIO*dataset_size))

    val_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))

    val_test_dataset = dataset.skip(int(TRAIN_RATIO*dataset_size))
    val_dataset = val_test_dataset.take(int(VAL_RATIO*dataset_size))
    return train_dataset, val_dataset, test_dataset

# Notice We moved the RATIOS down here and the defined datasets so that the splilts function could be defined properly.
# Note: The instructor actually moved his entire splits function up to acheive this, but for better clearity and
#visual understanding, we kept our splits function in place and just moved the RATIOS down along with the defined 
#datasets equaling the splits function and it's parameters.

TRAIN_RATIO = 0.6
VAL_RATIO= 0
TEST_RATIO = 0.2

train_dataset, val_dataset, test_dataset = splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

# Now we can print all of our datasets as one.
print(train_dataset, val_dataset, test_dataset) # Output

#<_TakeDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)> 
#<_TakeDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)> 
#<_SkipDataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>

# Notice that we have our three datasets.

# Lastly, we can also print our datasets as their numpy iterators using this method.

print(list(train_dataset.as_numpy_iterator()), list(val_dataset.as_numpy_iterator()), 
      list(test_dataset.as_numpy_iterator())) # Ouput

# [0, 1, 2, 3, 4, 5]
# []
# [6, 7, 8, 9]

# Notice that we got back all three of our datasets as their numpy iterators.
# Also notice that since we still have the VAL_RATIO as zero, we still get back an empty list as its elements,
#and the TEST_RATIO still gets returned with the last 40% / 4 elements as a result of this.