import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
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


# The next callback we will be looking at is the tf.keras.callbacks.ModelCheckpoint

# With the model checking callback we are able to save the keras model or model weights at some frequency.

# So unlike previously where we couldn't save our model until after training our model, now with this method we 
#can save our model during the training process.

# And that's thanks to this model checkpint callback.

# We'll start by copying the code template and pasteing it into our code.

# We'll be modifying the code to fit our needs.

# Instead of save_freq being set to after ever epoch, we will change it to after 3. That's if we set the save_best_only
#to true. Then we will check our models after every 3 epochs and save the best one, and if it's not the best we continue
#training. But for now, we will keep it set to "epoch".

# Next we will name our filepath "checkpoints/".

# Lastly we will the checkpoints_callback to our history.fit callbacks method.

# Now we can run our training.

checkpoint_callback = ModelCheckpoint(
    "checkpoints/", monitor="val_loss", verbose=0, save_best_only=True,
    save_weights_only=False, mode="auto", save_freq=3,   
)


# Now we will go over the documentation for each and every parameter.

# filepath - string or PathLike, path to save the model file. filepath can contain named formatting options, which will 
#be filled the value of epoch and keys in logs (passed in on_epoch_end). The filepath name needs to end with 
#".weights.h5" when save_weights_only=True or should end with ".keras" when checkpoint saving the whole model (default). 
#For example: if filepath is "{epoch:02d}-{val_loss:.2f}.keras", then the model checkpoints will be saved with the epoch 
#number and the validation loss in the filename. The directory of the filepath should not be reused by any other 
#callbacks to avoid conflicts.

# monitor - The metric name to monitor. Typically the metrics are set by the Model.compile method. Note:
    # Prefix the name with "val_" to monitor validation metrics.

    # Use "loss" or "val_loss" to monitor the model's total loss.

    # If you specify metrics as strings, like "accuracy", pass the same string (with or without the "val_" prefix).

    # If you pass metrics.Metric objects, monitor should be set to metric.name

    # If you're not sure about the metric names you can check the contents of the history.history dictionary returned by 
#history = model.fit()

    # Multi-output models set additional prefixes on the metric names.

# verbose - Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action.

# save_best_only - 	if save_best_only=True, it only saves when the model is considered the "best" and the latest best 
#model according to the quantity monitored will not be overwritten. If filepath doesn't contain formatting options 
#like {epoch} then filepath will be overwritten by each new better model.

# mode - 	one of {"auto", "min", "max"}. If save_best_only=True, the decision to overwrite the current save file is 
#made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be "max", 
#for val_loss this should be "min", etc. In "auto" mode, the mode is set to "max" if the quantities monitored are "acc" 
#or start with "fmeasure" and are set to "min" for the rest of the quantities.

# save_weights_only - 	if True, then only the model's weights will be saved (model.save_weights(filepath)), else the 
#full model is saved (model.save(filepath)).

# save_freq - 	"epoch" or integer. When using "epoch", the callback saves the model after each epoch. When using 
#integer, the callback saves the model at end of this many batches. If the Model is compiled with steps_per_execution=N, 
#then the saving criteria will be checked every Nth batch. Note that if the saving isn't aligned to epochs, the monitored
#metric may potentially be less reliable (it could reflect as little as 1 batch, since the metrics get reset every epoch). 
#Defaults to "epoch".

# initial_value_threshold - 	Floating point initial "best" value of the metric to be monitored. Only applies if 
#save_best_value=True. Only overwrites the model weights already saved if the performance of current model is better 
#than this value.


# After training for 10 epochs, We should have two logs.

# The first one that should tell us that our assets have been saved to checkpoint assets file which should have been 
#created and we should see that we're actually saving our model.

# And then we should have our second log that comes after the seed epoch.

# From this model checkpoint we will be looking at the reduced learning rate on plateau.



# In this section we will be going over the tf.keras.callbacks.ReduceLRonPLateau.

# This Reduces the learning rate when a metric has stopped improving.

# So if we start training with a fixed learning rate, and then say we've trained for over 100 epochs, and then
#after 110 epochs, our model stops improving, we'll adjust the learning rate by lowering it by a given factor.

# Here are the parameters and their definitions.

# monitor -	String. Quantity to be monitored.

# factor -	Float. Factor by which the learning rate will be reduced. new_lr = lr * factor.

# patience - Integer. Number of epochs with no improvement after which learning rate will be reduced.

# verbose -	Integer. 0: quiet, 1: update messages.

# mode - String. One of {'auto', 'min', 'max'}. In 'min' mode, the learning rate will be reduced when the quantity 
#monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing;
#in 'auto' mode, the direction is automatically inferred from the name of the monitored quantity.

# min_delta	- Float. Threshold for measuring the new optimum, to only focus on significant changes.

# cooldown	- Integer. Number of epochs to wait before resuming normal operation after the learning rate has been reduced.

# min_lr - Float. Lower bound on the learning rate.

# We're going to create our callback fairly easily and this time around we're going to monitor the validation accuracy.

# So if the validation accuracy doesn't increase, after 2 epochs(see patience), then our learning rate will be 
#reduced by a factor of 0.1.

# This means that if we have a learning rate of 0.01, for example, and the validation accuracy hasn't increased
#in 2 epochs, we are going to reduce the learning rate such that the learning rate becomes 0.01 times 0.1, which
#is the factor 0.1(see factor).

# Lastly we are going to add the plateau_callback to our history.fit(callbacks).

# Now we can train our model.

plateau_callback = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.1, patience=2, verbose=1
)

history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1, 
                                 callbacks = [plateau_callback])


# After training for 5 epochs, we can observe the change in our learning rate.