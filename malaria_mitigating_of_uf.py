import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer, Dropout
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L2, L1
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
dropout_rate = 0.2
regularization_rate = 0.01

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
    
lenet_custom_model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu", kernel_regularizer = L2(0.01)),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),
    Dropout(rate = dropout_rate),

    Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu", kernel_regularizer = L2(0.01)),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    Flatten(),

    NeuralearnDense(1000, activation = "relu"),
    BatchNormalization(),
    Dropout(rate = dropout_rate),

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


checkpoint_callback = ModelCheckpoint(
    "checkpoints/", monitor="val_loss", verbose=0, save_best_only=True,
    save_weights_only=False, mode="auto", save_freq=3,   
)


plateau_callback = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.1, patience=2, verbose=1
)

history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1, 
                                 callbacks = [plateau_callback])

# In this section we will be going over the process of Mitigating Overfitting and Underfitting

# We will be looking at different strategies for combating overfitting and underfitting.

# These strategies will include data augmentation, dropout, regularization, early stopping, smaler network usage,
#hyperparameter tuning, and normalization.

# So far in this course we've mentioned the terms overfitting and underfitting without really going into depth
#to what these two actually mean.

# For overfitting, we'll consider this plot of the loss versus the number of epochs, and the then the precision versus
#the number of epochs.


#   Loss                        Precision   ________________ training
#     |\                             |     /
#     |\\ = val                      |    /___________ validation
#     | \\        /                  |   //
#     |  \\______/                   |  //
#     |   \___________ t_loss        | //
#     |________________ Epoch        |//________________ Epoch


#   Loss

# With a loss versus the number of epochs, what we'll generally have when the model is overfitting is something like 
#this. (see loss plot above)

# So have the validation loss, and the training loss.

# So this is just a general way of looking at this though this may take different forms.

# Now with that said, we could see, we have these two sets and the loss, and their loss versus the number of epochs.

# Clearly we could see that the validation and the training set initially start up with a similar pattern.

# There are times when we may see that the validation performs better than the training set initially.

# But generally when a model overfits, what we'll have is this kind of model where at some point the model keeps
#doing well at a level of training, and starts doing very poory in the validation set.

# We could replicate this for the Precision epoch.

# This could be precision, accuracy, recall or some other metric we have chosen.

# And what we could have is something like this.

#   precision

# Obviously we saw in the practice, it isn't always like this, but something in this sense.

# So we would have the traing set, and the validtion set.

# So what goes on is our model keeps on doing well for the training, and then with the validation, at some point
#it starts doing very poorly.

# And so the danger with this is if we are considering working oly with a training set, we may feel like more
#epochs or training more epochs is good idea becuase we keep having these great results.

# Like suppose we have fixed our precision model so that it is training at a 99% accuracy rate.

# So we have a 99% precision on our training data and we think it's performing really great in the real world.

# But this isn't generally the case because this model has been overfitted on our training data.

# Our model has learned instead to modify its weights based on the training data instead of being able to extract
#useful information or some intelligence from the data which has been used in training this model.

# The main cause of overfitting is having a small data set and a large complex model which contains so many
#parameters.

# If a model has many parameters and we're giving it a small data set, then obviously it's gonna adjust its
#parameters such that on this small data set, it perfroms exceptionally well.

# Now in another case, we may have a moderate sized data set and a very large model.

# So at the end of the day, we notice that there has to always be a balance between the data set and the model size.

# This means that even if we increase the data size, and then we also increase the model size, our model may still
#risk overfitting.

# To better understand this concept of overfitting, let's take this example.

# Suppose we have three subjects to master school, Math, English, and Sports.

# But when kids get to school, they are only taught mathematics.

# So the children get to school and fro  the first year to the last year they are only taught mathematics.

# It's clear that when we evaluate these children, or when we pick a child at random and evaluate that child in
#mathematics, they would tend to have a better or above average result in mathematics compared to children 
#from other schools.

# But when tested on subjects like English or Sports that may not be the case.

# This is an example of overfitting.

# The model will do very well on things it was trained on.

# But when it comes to subjects that it hasn't seen before, it will perform poorly.

# Now it's clear that to adjust this situation, the children have to be taught all the subjects, because that is 
#the balance the children need to be well rounded.

# And obviously this balance will come in a way that the children will now perform better in the other two subjects.

# And because they have had part of the time they used to study math allocated for english and sports, they may
#perform slightly less than before in this subject, but what's important to notice this time around, they can
#now express themselves better in english or practice some sports.


# Now moving on to underfitting, it turns out that here our model becomes way to simple for it to even be able
#to extract information from our data.

# So we may have our validation data and then our training data.

# And then there is this huge gap between our current loss and the minimum possible loss.

# We could also have this at a level of let's say Accuracy.

# Then we have validation and training.

# And let's say the maximum of our accuracy is 100 percent.

# Then let's say we have a threshold of 1 right above our training.

# Our model is still too simple that we just end up with a 0.6, or 60% accuracy.

# In this kind of situation our data or the relative size of our data as compared to the model may be too large.

# So we could even have a situation where this data is smaller than what we had in the overfitting model.

# So our data could be small, but if our model is way too simple, let's say we have a very small model, then we may
#face this problem of underfitting.

# It also turns out that sometimes we may have a situation where we have a very complex model, but that model still
#underfits.

# That is because that model hasn't been built in a way that it could extract useful information from this data.



#   Loss                         Accuracy   
#     |\                             | 100%    
#     |\\ = val                      |________ 1  
#     | \\        /                  |   ________________ training
#     |  \\______/                   |  /___________ validation
#     |   \___________ t_loss        | //
#     |                              |//________________ Epoch
#     |
#     |
#     |
#     |
#     |________________ Epoch


# Now if we could recall in the section where we're predicting the car price, we had a situation where in a model
#we used a simple single dense layer with just two parameters and a fixed data set.

# But onced we increased, or, stacked up more dense layers, we found that we we're able to get better training and
#validation mean average error values.

# There are several ways in which we could mitigate the problem of overfitting.

# The very first one we'll look at is that of collecting more data.

# So it's impoortant to lay hands on as much data as possible.

# This data has to be representative of what the model would see in real life.

# And this data should be as diverse as possible.

# Even after collecting more data, to solve this problem of overfitting, we could use data augmentation.

# Now, what is data augmentation about?

# Suppose we have an image of a cell, which happens to be parasitized.

# Instead of having just this in our data set, we could have the image modified, such that we now have more data 
#to train on.

# So imagine initially we had let's say, 20,000 images, for exaample.

# Now, after doing data augmentation, after modifying each and every image, we now have a data set of 80,000.

# We're considering 80,000 because we're supposing that each image is going to be flipped.

# So we would take an image and rotate it, we have a new view of the image, which is still parasitized obviously.

# And we flip it agin for another angle.

# We can flip an image a total of 3 times before it is at its originally state. 

# And if we do that for all 20,000 images, that is how we get our 80,000 images.

# It should also be noted that there are many other data augmentations strategies for this kind of image data.

# So apart from flipping as we've just done, we could crop just a portion, we could add some noise to this data,
#we could modify the contrast, we could modify the brightness and carry on so many other operations.

# And there is no particular data augmentation strategy which works for all problems.

# This means that when we have a particular problem, we will have to try different augmentation strategies
#and then be able to select the one which works for our data.

# With that said, we'll look at the dropout.

# To better ubderstand the notion of dropout, we'll consider the simple neural network below.

#   O
#       O
#   O
#       O   O
#   O
#       O
#   O


# Now, if we could recall, the reasons why we have models which overfit is because we are working with very
#complex models with many parameters.

# Now, in order to reduce the complexity of our neural network, what we could do is take off the interaction between
#the second neuron in the first layer (first neuron on line 213) and the entire input layer (4 neurons), for example.

# So this means that when training our model, we are only going to consider two neurons in our first hidden layer
#instead of 3.

# So all the connections to that particular neuron become useless.

# Now, this has the effect of simplifying our network as to what we have as output.

# Now, that's after carrying out the dropout operation, looks like this.

#   O
#       
#   O   O
#           O
#   O   O
#  
#   O

# This is what our new neural network looks like.

# This particular case is an example of a dropout.

# We drop our ration 0.3, or let's say 0.333

# That's because we are dropping out exactly one third of the neurons in our first hidden layer.

# Now let's say for example that we performed the dropout again.

# And if ratio = two thirds, what we'll be left with is this.

#   O
#
#   O
#       O   O
#   O
#
#   O

# With this method we can leave from a very complex model to a simplified model via this dropout operation.

# And this has an overall effect of mitigating overfitting.

# The next step we could take is that of regularization.

# To better understand regularization, suppose we have this model with weights WJ.


#   ____________
#  |            |
#  |    W       |
#  |     j      |
#  |            |
#  |____________| 

# So say we have n weights, and these weights are free to take up any value.

# As we've seen previously, the fact that these weights can take up any value may lead to overfitting.

# As now, the weights can be adjusted to fit on the traininng data in a very perfect manner.

# So this means that we could have a model which picks out each and every point.

# Whereas if we restrain the weights to stay in a given range, then we may end up with a simplified model
#because it doesn't have as much freedom as the other, unrestricted model.

# So with regularization, our aim is to ensure that we minimize the loss, then we could include the weights
#in the computation of the loss.

# This means that we now have our loss, which is now equal to the loss we would have normally.

# Plus the regularization constant times the sum of the weights of each and every weight square.

# This is known as L2 regularization.

# Whereas with L1 regularization where we are summing up the absolute value of each andevery weight.

# For now, we're just going to explain how rerularization helps with mitigating that problem of overfitting by 
#restraining the weights in a given range.

# So let's have our Loss equal L(initial loss), plus R.

# Loss = L + R.

# Now if our aim is to minimize the loss, then obiously the L will be minimized and the R will be minimized.

# And so if we're trying to minimize this sum (w,j,2) in general, then it would have the overall effect of
#restraining the weights in a given range.

# Especially as we know that when we square very large values, these values become even larger.

# And so to avoid this, our weights will tend to take up smaller values which fall on the smaller range.

# The L2 regularization is also known as weight decay.

# It should be noted that the main difference between the L2 regularization and L1 regularization is that in 
#trying to restrain the range of values which the weights can take up, the L1 regularization has the negative 
#effect of making many of those weights take up values around zero.

# Those values are small, or take up many zero values.

# So this will lead to sparse models as compared to the L2 regularization.

# And that's why in practice we generally use the L2 regularization.

# With that being said, now we will move on to Early stopping, which we've seen already.


# In early stopping, as we have seen, in our model, let's say we have validation precision and training precision,
#for example, which keeps on increasing.

# And then we have a limit of 1 or 100%.

# So we have the precision and we have our epochs.

# And after a while the validation precision starts dropping, and this starts dropping because our model is now
#trying to over feed on the data it has been trained on.

# And so we have to stop training once we notice that tha validation performance isn't improving any longer.

# So this means that after a certain number of epochs we are going to stop the training.

# And we've seen this already in the previous section.

# So now we've seen it both in theory and practically.


# Now we will look at Smaller Networks.

# So another thing we could do is reduce the size of our network, or use a less complex network.

# Our next step will be to properly tune our hyper parameters.


# Our hyper parameters include batch-size, dropout-rate, regularization-rate, and learning-rate.

# These hyper parameters can effect our model and dictate whether the model will overfit or not.


# Now, we'll look at the batch-size.

# Training with a larger batch-size may speed up our training process, but working with smaller batch-sizes
#have a regularization effect which help reduce overfitting.

# And so according to Yann LeCun, "friends don't let friends use batch-sizes larger than 32."


# Now for the dropout-rate, we've seen this already.

# Increasing the dropout-rate means we are making the model simpiler. 


# The regularization-rate, 

# Increasing the regularization-rate means we're reducing the effect of overfitting.


# And then finally, we have our learning-rate.

# Picking too small of a learning-rate may lead to overfitting.


# So in general we have some hyper parameters to tune, and they are not limited to these, as we have many other
#hyper parameters depending on our problem.

# Now the fact that normalization introduces extra parameters, which bring in some noise in the model, has
#that regularization effect which help reduce overfitting.

# So if we're including batch normalization in our model, then we could feel free to reduce the dropout rate.

# Since this normalization layer already brings in the regularization effect.


# From here we look at ways of mitigating and the problem of over - under fitting.

# So with underfittinh we could have more complex models.

# We could also collect more data.

# We see that this solution falls in the tool that's both overfitiing and underfitting.

# This is a good thing to collect more data, or more clean and representative data.

# So from here we could also imrove the training time.


# Now we will see how the dropout could be implemented with tensorflow.

# So we have our dropout layer, which takes as arguments the dropout rate, which will look like this rate = "",
#the noise_shape = "", and the seed = ""

# tf.keras.layers.Dropout(
#     rate, noise_shape=None, seed=None
#)

# To better understand how and why we need the seed, is simply because in the case where one reproducible experiment.

# So if we want to apply dropout to a layer with a dropout rate of 0.2, then we'll be taking off one neuron out 
#of five neurons to make our model simplier and avoid overfitting.

# Now in doing so, we make a random choice as to which neuron is removed.

# If we want to fix the choice, so that the experiment can be reproducible, then we can set the seed so that each time
#we run the experiment it's going to choose exactly the same neuron to remove.

# Getting back to the code, the way we could use this is by first importing the dropout layer (see line 6).

# And then w ecould include dropout in our lenet_model, under the maxpool.

# So we would have Dropout and set it to a rate of 0.2.

# To make things easier we will initialize dropout_rate and set to 0.2. This is way we don't have to specify the 
#details everytime we implement it. 

# To see where we initialized dropout_rate see line 59.

# To see where we implemented it in our code see lines 114 and 124.

# Note: We are only adding the dropout to our first conv2d and first dense layer, but we could always add more
#dropouts as needed, and also increase or decrease the dropout rate as needed.

# From here we can run our model

# This is where we can see the dropout and see its effect on our model.

#Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# conv2d (Conv2D)             (None, 222, 222, 6)       168       
#
# batch_normalization (BatchN  (None, 222, 222, 6)      24
# ormalization)
#
# max_pooling2d (MaxPooling2D  (None, 111, 111, 6)      0
# )
#
# dropout (Dropout)           (None, 111, 111, 6)       0
#
# conv2d_1 (Conv2D)           (None, 109, 109, 16)      880       
#
# batch_normalization_1 (Batc  (None, 109, 109, 16)     64        
# hNormalization)
#
# max_pooling2d_1 (MaxPooling  (None, 54, 54, 16)       0
# 2D)
#
# flatten (Flatten)           (None, 46656)             0
#
# neuralearn_dense (Neuralear  (None, 1000)             46657000
# nDense)
#
# batch_normalization_2 (Batc  (None, 1000)             4000      
# hNormalization)
#
# dropout_1 (Dropout)         (None, 1000)              0
#
# neuralearn_dense_1 (Neurale  (None, 100)              100100
# arnDense)
#
# batch_normalization_3 (Batc  (None, 100)              400       
# hNormalization)
#
# neuralearn_dense_2 (Neurale  (None, 1)                101
# arnDense)
#
#=================================================================
#Total params: 46,762,737
#Trainable params: 46,760,493
#Non-trainable params: 2,244
#__________________________________________________________________


# Notice that our dropout doesn't have any parameters (see line 707).


# From here we will be looking at the tf.keras.regularizers.L2

# This is a regularizer that appiles a L2 penalty.

# So we'll be going over the process of implementing the L2 regularizer and the L1 regularizer.

# This is the sample code that we will be working with and modifying.

#tf.keras.regularizer.L2(
#    12 = 0.01
#)

# The 12 specifies the regularization rate.

# Before we move forward we will go back to our Conv2D and specify the kernel_regularizer, which will be eqaul to
#tf.keras.regularizer.L2(0.01) (see line 111).

# We can implement the L2 (and the L1 for future use) without the tf.keras.regularizer by importing it (see line 11).

# We can also add the kernel_regularizer = L2(0.01) to our second conv2D (see line 116).

# We can also add these to the dense layers, but note, we didn't add it the the last dense layer (which is the output 
#layer)

# This is how we implement the weight decay with tensorflow.

# And we could always modify the parameters.

# We will initialize regularization_rate to 0.01.

# Now we can run or model.

# From here we will go ahead and retrain our model.