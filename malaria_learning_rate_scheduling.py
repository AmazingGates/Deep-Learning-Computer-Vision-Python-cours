import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler
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

#csv_callback = CSVLogger(
#    "logs.csv", separator=",", append=True
#    )

#es_callback = EarlyStopping(
#    monitor = "val_loss", min_delta=0, patience=2, verbose=1,
#    mode = "auto", baseline=None, restore_best_weights=False
#)


# So we could train our model and then after 10 epochs, we restart the training by modifying the learning rate in
#our optimizer, and then we'll do that again after we've trained a certain amount of epochs.

# But now the problem with this is we have to always be there to ensure that after the training of each set of
#epochs, we modify the learning rate manually.

# Now, what if we're able to do this automatically?

# As usual this is made possible by Tensorflow callbacks.

# With this callback, that's the learning rate schedule log callback, we could define a function which takes in the
#number of epochs and then modifies the learning rate based on the current epoch,or based on a mixture of the 
#current epoch and some predefined function.

# So as we can see here we have the learning rate schedule log, which takes in a rate schedule, and then we could
#specify the verbosity.

# tf.keras.callback.LearningRateSchedule(
# schedule, verbose = 0
#)

# Now this is an example of this scheduler method which has been defined.

# def scheduler(epoch, 1r):
#   if epoch < 10:
#     return 1r
#   else:
#     return 1r * tf.math.exp(-0.1)
#

# Here this example is saying that if the number of epochs is less than 10 then you're gonna use this predefined 
#learning rate.

# And in the case where the number of epochs is greater than or equal to 10, then we start to reduce the learning
#rate in an exponential manner.

# Now what we're saying here is we're modifying the learning rate such that after that, we have in before 10 epochs,
#we have this fixed learninig rate. That's it.

# And then after this we start to reduce this.

# So the learning rate starts dropping as we continue with the training.

# So now we don't really need to monitor the training manually because of the callback will automatically modify
#the learning rate for us.

# We could simply copy out this example which has been given to us and then make use of it in our training process.

# BUt first we have to import the LearningRateScheduler right after the EarlyStopping in The callbacks imports

# So here we have to include this text in our code.

# This is the scheduler we will copy into our code.

def scheduler(epoch, lr):
  if epoch < 3:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

# Next we will define the learningratescheduler method and pass in a parameter of scheduler.

# We will also assign it a variable of scheduler_callback.

scheduler_callback = LearningRateScheduler(scheduler, verbose = 1)

# Notice how the scheduler method takes in the current epoch, the current epoch number, and the lr.

# All of that was already given to us in the example we copied over, but we can modify it.

# We're going to change 10 epochs to 3. (see line 284)

# So after 3 epochs we are going to modify the learning rate.

# Then we will pass in a verbosity in our LearningRateScheduler method and set it to equal 1.

# Next we modify our history.fit so that the callback attribute only has scheduler_callback as a parameter.

history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1, 
                                 callbacks = [scheduler_callback])


# So we're expecting that below 3 epochs, we are given a learning rate, and anything above that we have a learning
#rate that decreases exponentially.

# We'll notice that as our training begins we have a learning rate of 0.009999, which is basically 0.01, which is 
#what we set our initial learnng rate to.

# And then as time goes on the learning rate will automatically decrease in value.

# We want our learning rates adjusted because they affect the speed and stability of training our process.

# In the beginning, higher learning rates are good for speed and getting us through the training process, but
#as the process moves along, especially for larger training processes, it's best to switch to a lower learning
#rate which is best for the stability of our training process.
