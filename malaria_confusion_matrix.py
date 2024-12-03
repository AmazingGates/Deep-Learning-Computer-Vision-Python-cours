import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC
from keras.optimizers import Adam
import sklearn
from sklearn.metrics import confusion_matrix
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
    
IM_SIZE = 224
lenet_custom_model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    Flatten(),

    NeuralearnDense(1000, activation = "relu"),
    BatchNormalization(),

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

history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)

test_dataset = test_dataset.batch(1)

test_dataset

lenet_custom_model.evaluate(test_dataset)

# In this section we will be going over the Confusion Matrix in more detail

# Up until now we have been using everything from the BinaryAccurcay up until the AUC.

# What if we plotted out the Confusion Matrix, and also the ROC plot. 

# To actually handle this process we will import over scikit-learn and seaborn (see lines 10 and 12)

# Also, from sklearn.metrics we will import confusion_matrix (see line 11)

# These imports are going to allow us plot out our confusion matrix.

# So we'll run this and visualize our confusion matrix.

# We'll have to get the label.

# That's the true values of the outputs.

# And then we'll go ahead and get the predicted values.

# So we'll have our labels and predicted values

# Let's start with the labels.

# Here we're going to have labels, and for now we're going to assign it to an empty  list.

# And then for x and y, in the test dataset, and we're also going to attach a .as_numpy_iterator to our test_dataset

# So for x,y we have the x and the y.

# From here we will append every output to these labels.

# So we'll have labels.append(y), and we can run this.

# Now we can print out labels.

# After printing out our labels we will see that we have our labels for our outputs.

# Now let's convert it into a simpler form where we only have the values of the outputs returned, and not the arrays.

# In order to perform this task we will have labels equal to np.array([]).

# And inside np.array we are going to create a list.

# In this list, for every element of this labels list, we're going to i the zeroth index.

# So we're always going to take the element, and not the array.

# Now we will print out the labels again after doing the transformation


labels = []
input = []

for x,y in test_dataset.as_numpy_iterator():
    labels.append(y)
    input.append(x)

print(labels)

labels = np.array([i[0] for i in labels]) # When we run this part of the code keep we may have to keep this part 
#commented because it may throw an error. We'll run it first to see.

print(labels)

print(np.array(input).shape)
print(np.array(input)[:,0,...].shape)

# Ok, now we have our labels., we could move ahead and work on our predicted values.

# To get the predicted values we have our lennet model dot predict.

# What we're going to pass into this predicted method is our input.

predicted = lenet_custom_model.predict(np.array(input)[:,0,...])

# We'll initialize the input right under our labels empty list, as an empty list also. (see line 195)

# We will also do the same thing we did for the labels by adding a dot append, but this time we will be appending x.
#See line 199

# Now we will run this.

# And then we are going to pre process our input.

# From there we are just going to print out the input, to see what it gives us or what kind of shape.

#print(np.array(input).shape)

# Notice what we get back as a shape.

# We get back the Item number plus the shape, for example, 'IteratorGetNext:2742' shape=(None, 224, 224, 3)


# So we've gotten the labels, now we're ready for the predictions.

# We printed the input using this, np.array(input).shape, now we'll take this, and pass it as a parameter into our
#predict(). (see line 242)

# Then we will print our predictions using the print(predicted). (see line 244)
 
#predicted = lenet_custom_model.predict(np.array(input).shape)

print(predicted) 
print(predicted.shape)
print(predicted[:,0].shape)
print(predicted[:,0])

# So notice what we get. we could print out a shape to better understand what this is.

#print(predicted.shape)

# Notice that this will get us back our shape of our prediction.

# So do we take this and make look like our labels?

# We could simply modify the print(predicted.shape).

# We'll modify it like this, print(predicted[:,0].shape), and run it again.

#print(predicted[:,0].shape)

# So now we could take out the shape print(predicted[:,0]), then it should look like the labels we've had.


# Now that we've gotten this, our next step is to use the confusion matrix from scikit and metrics.

# First we'll initailize our threshold and set it to 0.5

# Next we assign our variable of cm to confusion_matrix() with the parameters of lables, predicted greater than
#the threshold.

# So what we're sayig is that all values greater than the threshold are considered uninfected, and all values less
#than or equal to the threshold is going to be considered parasitized.

# 
  

threshold = 0.5
# threshold = 0.25
# threshold = 0.75

cm = confusion_matrix(labels, predicted > threshold)
print(cm)

# We will run this portion of the code first separately.

# These results should show us the number of true negatives and true positives, and the number of false negatives and
#false positives.

# Getting back to our model evaluation, we'll see exactly which of these values are the TP's, FP's, TN's, or FN's.

# So what scikit returns us is similar to what TensorFlow gives us, when we compare the numbers.

# We'll then see that how by modifying the threshold we can either reduce the FN or reduce the FP.

# So for example, we'll reduce the threshold from 0.5 to 0.25

# After running that we will see that we now have a number of FN reduced when we reduce the threshold.

# Now we'll try to increase the threshold to 0.75 and see how things are effected.

# We will see that our number of FN's has increased and our number of FP's has reduced because we increased the value 
#of the threshold.

# Now we focus on printing out our plots.

# We will run our plots using the code below


plt.figure(figsize=(8,8))

sns.heatmap(cm, annot = True,)
plt.title("Confusion matrix - {}".format(threshold))
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# After running the code, we should have the same confusion metrics but more elegantly plotted.

# Now we notice that when we're trying to say reduce the FN or reduce the FP, what we're doing is we're just
#picking up some values, like changing thhe threshold to 0.2, for example, and we would see its effect on our plot.

# We could also do that for the threshold value of 0.25 and see its results.

# We could continue to do this until we get the best results.

# But, this isn't very efficient since we're just trying out different values.

# Now the way that we could do this more efficiently is by working with the ROC plots where we'll be able to choose
#a threshold in a more efficient manner using the plots.

# And so we'll be able to reduce the number of FN, or number of FP without having to try out a bunch of different
#threshold values manully.