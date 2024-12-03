import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger, EarlyStopping
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

#csv_callback = CSVLogger(
#    "logs.csv", separator=",", append=False
#)

csv_callback = CSVLogger(
    "logs.csv", separator=",", append=True
    )

es_callback = EarlyStopping(
    monitor = "val_loss", min_delta=0, patience=2, verbose=1,
    mode = "auto", baseline=None, restore_best_weights=False
)


history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1, 
                                 callbacks = [LossCallback(), csv_callback, es_callback])  


# From here we will be looking at the early stopping callback.

#   tf.keras.callbacks.EarlyStopping
# Stop training when a monitored metric has stopped improving

# To better understand early stopping, let's look at our plots from previously again.

# We'll look at the plot's model accuracy where we see how the train accuracy keeps increasing, while after a 
#certain point our models validation accuracy doesn't increase any further.

# So what we have here is something like this.

# We have the train accuracy which increases and goes towards one, and the validation accuracy which is something
#like this.


#   1 |       ______________ Train Accuracy
#     |      /_____________Validation Accuracy
#     |     //
#     |    //
#     |   //
#     |  //
#     | //
#     |//____________________


# Now in some other cases, we would even have situations where the validation line starts to drop.


#   1 |       ______________ Train Accuracy
#     |      /_____________Validation Accuracy
#     |     //             \
#     |    //               \
#     |   //                 \
#     |  //
#     | //
#     |//____________________


# Nonetheless, in this case we have our plot where it just knd of stablizes and doesn't increase any further.

# Now note that this type of situation is known as overfitting.

# In overfitting, the model starts to over fit the training data.

# So because the model has been trained on the training data, and not on the validation data, at certain points,
#the model stops, or ceases to generalize because the aim of the training process is not to come up with a model
#which only performs well on the training data.

# We're trying to come up with a model which performs well on any type of data, be it the train, the validation, or
#the test data.

# So if we're able to have a model to which does the same or which has the same performance with a train, and with
#a validation, and with a test, then that model is an ideal one.

# But in our case, we see that as we keep on training, the models parameters have been modified, to suit only the 
#training data.

# And this is very dangerous because at a certain point, we may feel like because we have a high training accuracy,
#our model is performing well.

# Whereas this isn't the case because when our model is shown new data, like in the validation data in this case,
#and the test data later on, this model wouldn't perform as well as it does now with the training data.

# So to avoid this kind of false measurement, we tend to stop the training once the overfitting status occurs.

# So this means that if we're training and then our validation seems to be constant, whereas the training seems 
#to kind of increase, then it would be better to stop the training at this point.

# After this point, the model parameters are just being modified to suit the training data and it really doesn't
#generalize, which is the case here because we're trying to extract some information from this data and make the
#model intelligent.

# So the model doesn't become more intelligent by only modifying its weights and parameters based on the data it has
#been trained upon. 

# It's intelligent because after being trained, it can perfrom well on data it has never seen before.

# So we have stopping early where after we notice that the validation accuracy doesn't seem to increase any further,
#we just kind of stio the training, and then use the model parameters from the number of epochs.

# So we could say after the peak training epochs have been reached, we stop the training.

# We can also do something similar for the loss.

# So if we have a loss, we would have something like this.

# This is our training loss 

#           L
#     |\
#     | (
#     | (
#     |  (
#     |  (
#     |   (
#     |    (_______________
#     |____________________(__ Epochs

# So we could have a situation where we have our training data, our training loss, which keeps reducing.

# Whereas for the validation, we would have something like this.

# This is our validation loss.

#           Loss
#     |\
#     | (
#     |  (                   _)
#     |   (___             _)
#     |       (           )
#     |        (_________/
#     |              
#     |______________________ Epochs


# Both of these plots together are one plot, and be a typical eample of overfitting.
#Note: They are only separated for visual purposes.

# In our case the model overfits, but not that much.

# In some cases we will have a situation where it starts to drop and where the loss starts to increase after a 
#certain point.

# The second plot is a visual of our validation loss, and the first plot is the visual for our training loss.

# Obviously the training loss will keep reducing because we are training it on training data.

# So what we're saying is at the point where the validation stops reducing, it's important to just stop the 
#training, which is known as early stopping.

# Now recall that the aim of callbacks is to be able to modify the training process, the evaluation process or
#the test process as a prediction process in an automatic manner.

# So with that being said we'll make use of the early stopping callback which will permit us to stop training
#automatically once we notice that a given parameter, let's say validation loss for example, doesn't drop any longer.

# Now we can stop writing the code that will handle this process for us.

# And then we apply it similarly to how we did with the csv callback.

# This is the code we will be using.

# tf.keras.callbacks.EarlyStopping(
#    monitor = "Val_Loss", min_delta=0, patience=0, verbose=0,
#    mode = "auto", baseline=None, restore_best_weights=False
#)

# Now we will create a variable called es_callback and set it eqaul to our equation.

# See line 390.

# Before we run we will look the significance of each and every argument inside the EarlyStopping.

# monitor - Quantity to be monitored

# min_delta - Minimum chnage in the monitored quantity to qualify as an improvement, i.e. an absolute change of
#less than min_delta, will count as no improvement.

# patience - Number of epochs with no improvement after training will be stopped.

# verbose - Verbosity mode.

# mode - One of {"auto", "min", "max"}. In the min mode, training will stop when the quantity monitored has stopped
#decreasing, in "max" mode it will stop when the quantity monitored has stopped increasing, in "auto" mode, the
#direction is automatically inferred from the name of the monitored quantity.

# baseline - Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement
#over the baseline.

# restore_best_weights - Whether to restore model weights from the epoch with the best value of the monitored
#quantity. If False, the model weights obtained at the last stop oftraining are used. An epoch will be restored
#regardless of the performance relative to the baseline. If no epoch improves on baseline, traininng will run 
#patience epochs and restore weights from the best epoch in the set.

# With all that explained, now we can move forward with our code.

# Well set the patience to equal 3, verbose to equal 1, and everything else can stay the same.

# Next we just need to include it in our history.fit.

# We'll add es_callback to the Callbacks method, after the csv_callback.(see line 397)

# Now we can run the training.

# After training for 8 epochs we see clearly how the early stop in callback stops the training process.

# Now let's examine why the training process has been stopped.

# If we take a look at the validaton loss, we would find that there was a drop in increase.

# Because we set the patience to 2, we have to get 2 consecutive loss increases before the training stops.

# For example, if we go from 0.255 to 0.243 to 0.266, the training will continue, because the loss decreased once
#and then increased once.

# But if the training loss went from 0.255 to 0.266 to 0.279, then the training would be stopped early, because 
#we have consecutive loss increases.

# So we now have the training process which has been stopped.

# With that said, we will now move on to the learning rate scheduling.