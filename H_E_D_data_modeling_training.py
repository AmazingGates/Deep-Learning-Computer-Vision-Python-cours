import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer, Dropout, Rescaling, Resizing
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC, CategoricalAccuracy, TopKCategoricalAccuracy
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L2, L1
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import cv2


train_directory = "Human Emotion Images/train"

val_directory = "Human Emotion Images/test"

CLASS_NAMES = ["angry", "happy", "sad"]

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "LEARNING_RATE": 0.001,
    "N_EPOCHS": 3,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 100,
    "N_DENSE_2": 10,
    "NUM_CLASSES": 3 
}


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',
#    label_mode='int',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=99,
)


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    labels='inferred',
#    label_mode='int',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
)


for i in validation_dataset.take(1):
    print(i)


plt.figure(figsize=(12,12))

for images, labels in train_dataset.take(1):
    for i in range(16):
        ax = plt.subplot(4,4, i+1)
        plt.imshow(images[i]/255.)
        #plt.title(tf.argmax(labels[i], axis=0).numpy())
        plt.title(CLASS_NAMES[tf.argmax(labels[i], axis=0).numpy()])
        plt.axis("off")
        plt.show()


train_dataset = (train_dataset.prefetch(tf.data.AUTOTUNE))

validation_dataset = (validation_dataset.prefetch(tf.data.AUTOTUNE))

resize_rescale_layers = tf.keras.Sequential([
    Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    Rescaling(1./255)
])



# In this section we will go over the process of modeling and training our data.

# We will be building our own me image in  that allows us to pass in inputs(images) and tells us whether the 
#classification(Happy, Sad, Angry) of the input(image).

# In this session we are going to start with the lenet me image in  which which we used in the previous me image in , and 
# #then move on to even more complex and better computer vision models.

# This is the code that we will be using from the previous me image in .

lenet_model = tf.keras.Sequential([
    #InputLayer(input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3)),
    InputLayer(input_shape=(None, None, 3)),
    resize_rescale_layers,

    Conv2D(filters=CONFIGURATION["N_FILTERS"], kernel_size=CONFIGURATION["KERNEL_SIZE"], strides=CONFIGURATION["N_STRIDES"],
           activation="relu", kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D (pool_size=CONFIGURATION["POOL_SIZE"], strides=CONFIGURATION["N_STRIDES"]*2),
    Dropout(rate = CONFIGURATION["DROPOUT_RATE"]),

    Conv2D(filters=CONFIGURATION["N_FILTERS"]*2 + 4, kernel_size=CONFIGURATION["KERNEL_SIZE"], strides=CONFIGURATION["N_STRIDES"],
           activation="relu", kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D (pool_size=CONFIGURATION["POOL_SIZE"], strides=CONFIGURATION["N_STRIDES"]*2),

    Flatten(),

    Dense(CONFIGURATION["N_DENSE_1"], activation="relu", kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dropout(rate = CONFIGURATION["DROPOUT_RATE"]),

    Dense(CONFIGURATION["N_DENSE_2"], activation="relu", kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),

    Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax"),

])

lenet_model.summary()


# Now we are going to get to our training.

# We will start by designing our loss function.

# This is how we wil start that function.

# Unlike before where we had the Binary, this time we will be using the Catergorical.

# One of the parameters we will have is the from_logits, which is set to False by default.

# We can visit the tensorflow website to read the documentation for all of the parameters descriptions.

# When we have from_logits = False, it simply means that we're supposing that what is going to get into this
#loss function is going to be a logits tensor, which is going to be the case here because we're using the softmax
#activation. 

# In the case that we didn't have the softmax activation, then we would have needed to set the from_logits to 
#True, such that what gets into Categorical Cross-entropy is a logits tensor.

# Now let's go ahead and test this example here that we will get from the tensorflow website.

# y_true = [[0, 1, 0], [0, 0, 1]]
# y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
# Using "auto"/"sum_over_batch_size"
# cce = tf.keras.losses.CategoricalCrossentropy()
# cce(y_true, y_pred).numpy()


#loss_function = CategoricalCrossentropy(
#    from_logits=False
#)

loss_function = CategoricalCrossentropy()

y_true = [[0, 1, 0], [0, 0, 1]]
#y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.05, 0.85]]
#y_pred = [[0, 1.0, 0], [0.0, 0.0, 1]]
#y_pred = [[0.0, 0, 1.0], [0.0, 1.0, 0.0]]
# Using "auto"/"sum_over_batch_size"
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred).numpy())

# If we ran this now we would see that we would get back a value of 1.1769392.

# Now this value here tells us how close the me image in 's prediction is to the true values of y.

# Now let's modify our me image in 's prediction such that it's very close to the true value of y.

# So here we modify the second list in our y_pred.

# We will change it to [0.1, 0.5, 0.85]. See line 170 for modification

# Notice that all our list for y_true and y_pred sum to 1.

# Now we will run this again to see the difference.

# The value we get back this time is 0.1069061

# Our value drops by almost a factor of 10.

# So this shows us that these two are very close to each other. 

# Now let's change both the list in our y_pred to [[0.0, 1.0, 0], [0.0, 0.0, 1.0]] and see what we get.

# This is the new value we have 1.192093e-07, which is practically zero because the true and pred are so close.

# Now if we change the y_pred to [[0.0, 0, 1.0] and [0.0, 1.0, 0.0]], we will notice that the true value (1)
#location is in a completely different location than the predicted.

# Now let's run this again to see what we get.

# This is the value we get now 16.118095.

# This number is so large because the positions of the y_true and the y_pred are in two different locations.

# So this shows us how the Categorical CrossEntropy actually works.

# Moving forward we can remove the from_logits = false as a parameter, since it is aleady the default setting, 
#there is no need to specify it as a parameter. (see line 169 for update)

# With that done, let's move on to the metrics of our me image in .

# This is how we will define our metrics.

# Our metric is going to be CategoricalAccuracy, and we will give it a name of accuracy.

# And we will also have the TopK CategoricalAccuracy, which we will give a value of k = 2, and we will give it
#a name of top_k_accuracy

# 


metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k = 2, name = "top_k_accuracy")]

# Now before we move on let's explain the TopKCategoricalAccuracy metric.

# So unlike with the "accuracy", where we have these four stations for example.

#
# (0.1) (1)       (0.8) (0)
# (0.4) (0)       (0.1) (1)
# (0.5) (0)       (0.1) (0)
#   P   EP          P    EP

#
# (0.90) (1)       ( 0 ) (0)
# (0.05) (0)       (0.7) (1)
# (0.05) (0)       (0.3) (0)
#    P   EP          P    EP

# The P columns are what the me image in  predicted, and the EP columns are what the models were expected to predict.

# The accuracy rate will be computed as such.

# Taking a look at our first example, we see that highest value in our P column is the 0.5, and the highest value
#in our EP column is the 1. 

# This would be considered a no match, which would give us a zero. 

# This is considered a no match because the highest value of the P column is in a different position than the
#highest value of the EP column.

# Following that logic we can get the output scores for the next 3 examples.

# That would look something like this.

# 0 + 0 + 1 + 1

# Next, since we have four different examples, we will take the correct answers and divide by 4, which will look
#like this.

#     1 + 1
#---------------
#       4

# And then multiply by 100

#     1 + 1
#--------------- x 100
#       4

# This will give us an accuracy of 50%.

# Now, to get back to the TopKCategoricalAccuracy we will use the same four examples.

#
# (0.1) (1)       (0.8) (0)
# (0.4) (0)       (0.1) (1)
# (0.5) (0)       (0.1) (0)
#   P   EP          P    EP

#
# (0.90) (1)       ( 0 ) (0)
# (0.05) (0)       (0.7) (1)
# (0.05) (0)       (0.3) (0)
#    P   EP          P    EP

# With the TopKCategoricalAccuracy, we're interested in just our first highest predictions.

# We are not interested in making sure that the highest prediction matches the highest expected prediction.

# What we are interested in is if any of the two highest predictions matches the highest expected predictions.

# If they do, we will consider that a correct prediction.

# So let's look at the first example. We would consider that an incorrect prediction because none of the two 
#highest values of the P column matches the highest value of the EP column.

# Now if we look at the second example we would consider that a correct prediction. 

# This is because one of the highest two values of the P column matches the highest value of the EP column.

# So following this logic this would be the output value for each example.

# 0 + 1 + 1 + 1

# The follow steps would look like this

# 1 + 1 + 1
#----------- x 100
#     4

# This will return us an acurracy of 75%.

# Now we can go ahead and compound our me image in .

# This is how we will compound our me image in .

lenet_model.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)

# Now that our me image in  is compile we can run our  history.

# This is how we will implement that.

history = lenet_model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1,
)

# Now that we are done with the history, we will plot out our loss curves for the validation and the training.

# This is how we will do that.

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"])
plt.show()

# As we can notice, the training and validation losses both drop together.

# Next we will do the same for the accuracy.

# This is how we will do that.

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend(["train_accuracy", "val_accuracy"])
plt.show()

# As we can notice, the training and validation accuracies both rise together.

# Note that our image isn't overfitting, and our models metrics keep increasing, so what we could do is increase
#the number epochs to get better results.

# Next we will evaluate the our models performance using the lenet_model.evaluation, with a parameter of validation
#dataset.

# Running this will give us our loss, accuracy, and topk accuracy.

lenet_model.evaluate(validation_dataset)

# This is the evaluation of model.

#213/213 [==============================] - 183s 830ms/step - loss: 0.3929 - accuracy: 0.8582 - top_k_accuracy: 0.9744

# Now we're ready to test out this model on an image in our testing dataset.

# This is how we are going to that using the (Open CV library) cv2.imread method that will take our image as a parameter
#and read it.


test_image = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test\happy/62821.jpg_brightness_2.jpg")

# Then we will convert this into a tensor, and also specify the data type.

# This is the process we will use to do that.

im = tf.constant(test_image, dtype = tf.float32)

# From here we will pass this information into our model directly because the way we have designed this model is 
#we put the resizing and the rescaling tool in the model, so we don't have to do that outside of the model.

# But before we call our lenet model to pass in this information, we will add one dimension, since we're passing 
#this input in the model as batches.

# So we add the battch dimension here using the tf.expand.

# And then we will pass in parameters of image and axis.

# This is how we will do that.

im = tf.expand_dims(im, axis = 0)

# So once we have this, let's call our lenet model which takes in that image as a parameter.

# And print that out to see what our model gives us.

# This is how we will do that.

# Before we we run this we will modify our lenet model input layer.

# We will change the configuration to none for both instances. (see line 109)

print(lenet_model(im))

# Another thing we can do is print out the class.

# We can do that by modifying our print method like this.

print(CLASS_NAMES[tf.argmax(lenet_model(im),axis = -1).numpy()[0]])

# Notice that instead of the lenet model, we are using the tf.argmax, and specified our axis. We also converted
#this to a numpy() with an index of 0. And then we used the CLASS_NAMES to get the name

# So we'll look for the class with the highest probability of being true.

# This method will return the name of the class with the highest probability of being correct.

# With that being done, we will do one last test.

# This time we will use an example from the sad classification and see how our model handles it.

# This is the code we will run for our second example using a selection from the sad classification.


test_image2 = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test/sad/17613.jpg")
im2 = tf.constant(test_image2, dtype = tf.float32)
im2 = tf.expand_dims(im2, axis = 0)
print(lenet_model(im2))
print(CLASS_NAMES[tf.argmax(lenet_model(im2),axis = -1).numpy()[0]])

# Now we'll do something similar to when we had our model label each one of our images, but this time we will return
#what the model predicts.

# This is the code we will use.

# Notice that it's a modified version of the code we used above to get the labels for our images.

# For this version we will be using the validation dataset instead of the train dataset.

# To see the original see line 75.

# We will also modify the label of the title of the plot. (See line 80 for original)

# Our modified label version will have a true label and a predicted label, plus the lenet model which gets passed a 
#parameter of tf expanded dimensions of images with an index of i, with an axis of 0


plt.figure(figsize=(12,12))

for images, labels in validation_dataset.take(1):
    for i in range(16):
        ax = plt.subplot(4,4, i+1)
        plt.imshow(images[i]/255.)
        #plt.title(tf.argmax(labels[i], axis=0).numpy())
        plt.title("True Label - :" + CLASS_NAMES[tf.argmax(labels[i], axis=0).numpy()] + "\n" + "Predicted Label - : " 
                  + CLASS_NAMES[tf.argmax(lenet_model(tf.expand_dims(images[i], axis = 0)),axis = -1).numpy()[0]])
        plt.axis("off")
        plt.show()

# Now we can run this to see what we get.

# Notice that we now get back the the True labels for our images, and the predicted labels also.



# The next thing we'll do is plot out the confusion matrix.

# This is how we will do that.

# We'll start by going through our validation data.

# The we have predicted, which gets append with the lenet model, that takes in as a parameter of im.

# Then we will store our labels in a variable.

# But before we start this code we will initialize both predicted and labels and set them to empty lists.

predicted = []
labels = []

for im, label in validation_dataset:
    predicted.append(lenet_model(im))
    labels.append(label.numpy())


# Before moving on we will convert the label to a numpy(). See line 504

# Now we can print outour labels.

print(labels)

# Now that we have all of our labels we will flatten them out.

# We will do that by printing out the argmax of the labels and printing up until the last batch using this index [:-1],
#also specifying the last axis.

# This is how we will do that

print(np.argmax(labels[:-1], axis = -1))

# From here we could actually flatten now.

# This is how we will do that.

print(np.argmax(labels[:-1], axis = -1).flatten())

# We could print out the length of this return to see how many elements we actually have.

# This is how we will do that.

print(len(np.argmax(labels[:-1], axis = -1).flatten()))

# What we want to do is compare all of these labelss with the models predictions.

# So we'll print the label and the predicted side by side to that.

# This is how we will run them.

print(np.argmax(labels[:-1], axis = -1).flatten())
print(np.argmax(predicted[:-1], axis = -1).flatten())

# Now that we have this set we can define our pred.

# This is how we will do that.

pred = np.argmax(predicted[:-1], axis = -1).flatten()

# These are the different predictions of the model.

# We will do the same for the labels.

lab = np.argmax(labels[:-1], axis = -1).flatten()

# These are what the model was supposed to predict.

# Now we will code our confusion matrix by using the same cm that we created in the previous model, just modified
#to fit the code we are running now.

cm = confusion_matrix(lab,pred)
print(cm)
plt.figure(figsize=(8,8))

sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix - {}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# This should return our confusion matrix.

# Notice that when we look at the printed values of our cm, we see that the diagonal values are the highest.

# The high values indicate the model predicting the accurate label for the image.

# Let's take a look at the chart below to get a better understanding.

#           Happy   Angry   Sad - Predicted
# Happy |  622    |  299   |  599
#       |         |        |
# Angry |  12     | 2819   |  181
#       |         |        |
# Sad   |  38     |  138   |  2076
#
# Actual


# Also, when looking at our actual cm chart, the lighter the color grid indicates a higher value, and the darker 
#the color indicates a lower value.

# That's pretty much how we handle generating our Confusion Matrix, which is a very important aspect.

# Now we can go back and address that last batch that we instructed our code to print up until, using this method
#labels[:-1].

# What we could to is some concatenation.

# This is how we will do that.

print(np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()]))

# Notice that we used a np.concatenate method on our np.argmax, then we copy this piece of code inside that method
#np.argmax(labels[:-1], axis = -1).flatten(), and did we re-added it the same code after the flatten(), separated by 
#a comma.

# The difference is that in the code we re-added, we will include the last batch we previously left out.

# We can do that by specifing it like this labels[-1]. see line 600.

# Also notice that we added [] around both our np.argmax parameters.

# Now we will do the samething for the predicted.

print(np.concatenate([np.argmax(predicted[:-1], axis = -1).flatten(), np.argmax(predicted[-1], axis = -1).flatten()]))

# Next we will use the same concatenate on our lab and predict varaiables.

# This is how we will do that.

pred = np.concatenate([np.argmax(predicted[:-1], axis = -1).flatten(), np.argmax(predicted[-1], axis = -1).flatten()])


lab = np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()])

# Now we have successfully used the Cofusion Matrix and all of its conponents.

# Now we can move to tyue next topic.