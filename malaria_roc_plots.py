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


# In this section we will be going over the steps and process of using ROC Plots.

# The very first thing we'll do is import the roc_curve, and that's part of the sk learn metrics.

# Now we will run this again, and make use of our method.

# We're going to output the number of false positives, true positives, and then the threshold, which we'll
#generate with the ROC plot by assigning it to roc_curve, which takes as a parameter (labels, predicted)

# Now if we print the length out, the length of fp, the length of tp, and the length of threshold, we should get 
#back the same exact numbers.

fp, tp, thresholds = roc_curve(labels, predicted)
print(len(fp), len(tp), len(thresholds))

# Note that the reason we need this is because when coming up with our ROC plot, we wanna have for each and every
#point the corresponding TP, FP, and then the Threshold, which will lead to that Tp, FP pair.

# So with that said, we're going to make use of that data now and then plot out the ROC curve.

# So let's get straight into that.

# We have our plot and then we do our plotting.

# We pass into the plot() fp and tp, which act as our x and y, respectively.

# From here we have our labels.(xlabel and ylabel)

# Now that we have this we could include the grid.

# We'll use the plt.grid

# And next we'll use the plt.show

plt.plot(fp, tp)
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.grid()

skip = 20

for i in range(0, len(thresholds), skip):
    plt.text(fp[i], tp[i], thresholds[i])

plt.show()

# Now we can run this and see what we get.

# How do we include the threshold?

# In order to include the threshold we are going to make use of matplot lib test method.

# So before the plt.show, we will add the plt.text(), and then we're going to pass in the fp, tp, thresholds.

# Now note that we actually have to do this for each and every point, which isn't possible since there will be too
#many text outputs which will cause congestion.

# So what we can do is skip some values.

# So what we'll say is for i in range, we're going to start from zero right up until the length of thresholds,
#and then we are going to be skipping some values by passing skip.

# Then we'll define our skip such that we will be skipping 20 values at a time.

# Once we skip this value we will pick within a given i for fp, tp and thresholds

# So we get the corresponding false positive rate, corresponding true positive rate, and corresponding threshold.

# Now that that's done we can run this.

# Once we run this and everything goes correctly we should see our ROC plot with the different thresholds.

# Now let's create a size and just focus on the curve portion of our plot, which actually matters the most
#because we wouldn't want to get into the other regions because our false positive rates will be too high
#or our true positives will be too low.

# So generally we'll try to ffocus on the zone around the curve of our plot.

# Now depending on the problem we are trying to solve, if our false positive rate is what matters the most,
#that is if we can't afford to have a high false positive rate, then we would tend to pick values around the curve.

# On a side note: If we have a problem like the one we're trying to solve, where Parasite = 0, and then Uninfected
#= 1, how our dataset was created and our model was built, then that means that this Parasite = 0, will be considered 
#as negative samples, and this Uninfected = 1, will be considered as positive samples.

# Whereas iin the real world, we'll tend to look at Unifected as negative, and parasitized as positive.

# So we have to be very careful with these terms and know exactly how our data and models have been built.

# And that's why in our case where we're trying to avoid situations where our model predicts a fake uninfected output,
#that is an output saying the patient who actually has the parasite, is uninfected. This would be the case of 
#a fake uninfected.

# This is a False Positive in our case, so we'll tend to minimize the number of FP.

# Now if our model was built in such a way that Parasitic = 1, and then uninfected = 0, then in this case
#we would be trying to minimize the number False Negatives instead, since uninfected would be considered as a negative.

# That said, coming back to our problem, since our data set was constructed in this way, we try to minimize the 
#number of False Positives at all cost.

# But while doing this, we have to ensure that the True Positive rate remains at a reasonably position.