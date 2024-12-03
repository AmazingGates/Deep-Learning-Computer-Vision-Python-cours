import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer, Dropout, Rescaling, Resizing, RandomFlip, RandomContrast, RandomRotation, MaxPooling2D, GlobalAveragePooling2D, Activation, Add
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC, CategoricalAccuracy, TopKCategoricalAccuracy
from keras.optimizers import Adam
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L2, L1
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import cv2
import tensorflow_probability as tfp


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


### tf.keras.layer augment
augment_layers = tf.keras.Sequential([
#    RandomRotation( factor = (0.25, 0.2501),),
    RandomRotation( factor = (-0.025, 0.025),),
    RandomFlip(mode = "horizontal", ),
    RandomContrast(factor = 0.1),
])

def augment_layer(image, label):
    return augment_layers(image, training = True), label

def box(lamda):

    r_x = tf.cast(tfp.distributions.Uniform(0, CONFIGURATION["IM_SIZE"]).sample(1)[0], dtype = tf.int32)
    r_y = tf.cast(tfp.distributions.Uniform(0, CONFIGURATION["IM_SIZE"]).sample(1)[0], dtype = tf.int32)

    r_w = tf.cast(CONFIGURATION["IM_SIZE"]*tf.math.sqrt(1-lamda), dtype = tf.int32)
    r_h = tf.cast(CONFIGURATION["IM_SIZE"]*tf.math.sqrt(1-lamda), dtype = tf.int32)

    r_x = tf.clip_by_value(r_x - r_w//2, 0, CONFIGURATION["IM_SIZE"])
    r_y = tf.clip_by_value(r_y - r_h//2, 0, CONFIGURATION["IM_SIZE"])

    x_b_r = tf.clip_by_value(r_x - r_w//2, 0, CONFIGURATION["IM_SIZE"])
    y_b_r = tf.clip_by_value(r_y - r_h//2, 0, CONFIGURATION["IM_SIZE"])

    r_w = x_b_r - r_x
    if(r_w == 0):
        r_w = 1

    r_h = y_b_r - r_y
    if(r_h == 0):
        r_h = 1

    return r_y, r_x, r_h, r_w


def cutmix(train_dataset_1, train_dataset_2):
    (image_1, label_1), (image_2, label_2) = train_dataset_1, train_dataset_2

    lamda = tfp.distributions.Beta(0.2,0.2)
    lamda = lamda.sample(1)[0]

    r_y, r_x, r_h, r_w = box(lamda)
    crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
    pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])

    crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
    pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])

    image = image_1 - pad_1 + pad_2

    lamda = tf.cast(1- (r_w*r_h)/(CONFIGURATION["IM_SIZE"]*CONFIGURATION["IM_SIZE"]), dtype = tf.float32)
    label = lamda*tf.cast(label_1, dtype = tf.float32) + (1-lamda)*tf.cast(label_2, dtype = tf.float32)

    return image, label


train_dataset_1 = train_dataset.map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)
train_dataset_2 = train_dataset.map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)

mixed_dataset = tf.data.Dataset.zip((train_dataset_1, train_dataset_2))


training_dataset = (
                    mixed_dataset
                    .map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE))


#train_dataset = (train_dataset
#                 .map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)
#                 .prefetch(tf.data.AUTOTUNE))

validation_dataset = (validation_dataset.prefetch(tf.data.AUTOTUNE))

resize_rescale_layers = tf.keras.Sequential([
    Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    Rescaling(1./255)
])


lenet_model = tf.keras.Sequential([
    #InputLayer(input_shape=(CONFIGURATION["CONFIGURATION["IM_SIZE"]"], CONFIGURATION["CONFIGURATION["IM_SIZE"]"], 3)),
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

loss_function = CategoricalCrossentropy()

y_true = [[0, 1, 0], [0, 0, 1]]
#y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.05, 0.85]]
#y_pred = [[0, 1.0, 0], [0.0, 0.0, 1]]
#y_pred = [[0.0, 0, 1.0], [0.0, 1.0, 0.0]]
# Using "auto"/"sum_over_batch_size"
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred).numpy())


metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k = 2, name = "top_k_accuracy")]


lenet_model.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)


history = lenet_model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1,
)


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"])
plt.show()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend(["train_accuracy", "val_accuracy"])
plt.show()

lenet_model.evaluate(validation_dataset)

test_image = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test\happy/62821.jpg_brightness_2.jpg")

im = tf.constant(test_image, dtype = tf.float32)

im = tf.expand_dims(im, axis = 0)

print(lenet_model(im))

print(CLASS_NAMES[tf.argmax(lenet_model(im),axis = -1).numpy()[0]])


test_image2 = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test/sad/17613.jpg")
im2 = tf.constant(test_image2, dtype = tf.float32)
im2 = tf.expand_dims(im2, axis = 0)
print(lenet_model(im2))
print(CLASS_NAMES[tf.argmax(lenet_model(im2),axis = -1).numpy()[0]])


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


predicted = []
labels = []

for im, label in validation_dataset:
    predicted.append(lenet_model(im))
    labels.append(label.numpy())


print(labels)

print(np.argmax(labels[:-1], axis = -1))

print(np.argmax(labels[:-1], axis = -1).flatten())

print(len(np.argmax(labels[:-1], axis = -1).flatten()))

print(np.argmax(labels[:-1], axis = -1).flatten())

print(np.argmax(predicted[:-1], axis = -1).flatten())


pred = np.argmax(predicted[:-1], axis = -1).flatten()


lab = np.argmax(labels[:-1], axis = -1).flatten()


cm = confusion_matrix(lab,pred)
print(cm)
plt.figure(figsize=(8,8))

sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


print(np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()]))

print(np.concatenate([np.argmax(predicted[:-1], axis = -1).flatten(), np.argmax(predicted[-1], axis = -1).flatten()]))

pred = np.concatenate([np.argmax(predicted[:-1], axis = -1).flatten(), np.argmax(predicted[-1], axis = -1).flatten()])

lab = np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()])



# In this section we will be implementing a resnet34 code from scratch with Tensorflow 2.

# We are going to construct our resnet 34 while making use of model subclassing.


class CustomConv2D(Layer):
    def __init__(self, n_filters, kernel_size, n_strides, padding = "valid"):
        super(CustomConv2D, self).__init__(name = "custom_conv2d")

        self.conv = Conv2D(
            filters = n_filters,
            kernel_size = kernel_size,
            activation = "relu",
            strides = n_strides,
            padding = padding)

        self.batch_norm = BatchNormalization()

#    def call(self, x, training):
    def call(self, x, training = True):

        x = self.conv(x)
#        x = self.batch_norm(x)
        x = self.batch_norm(x, training)

        return x




class ResidualBlock(Layer):
    def __init__(self, n_channels, n_strides = 1):
        super(ResidualBlock, self).__init__(name = "res_block")

        self.dotted = (n_strides != 1)

        self.custom_conv_1 = CustomConv2D(n_channels, 3, n_strides, padding = "same")
        self.custom_conv_2 = CustomConv2D(n_channels, 3, 1, padding = "same")

        self.activation = Activation("relu")

        if self.dotted:
            self.custom_conv_3 = CustomConv2D(n_channels, 1, n_strides)

#    def call(self, input):
    def call(self, input, training):

#        x = self.custom_conv_1(input)
        x = self.custom_conv_1(input, training)
#        x = self.custom_conv_2(x)
        x = self.custom_conv_2(x, training)

        if self.dotted:
#            x_add = self.custom_conv_3(input)
            x_add = self.custom_conv_3(input, training)
            x_add = Add()([x, x_add])
        else:
            x_add = Add()([x, input])

        return self.activation(x_add)



class ResNet34(Model):
    def __init__(self,):
        super(ResNet34, self).__init__(name = 'resnet_34')
        self.conv_1 = CustomConv2D(64, 7, 2, padding = "same")
        self.max_pool = MaxPooling2D(3, 2)

        self.conv_2_1 = ResidualBlock(64)
        self.conv_2_2 = ResidualBlock(64)
        self.conv_2_3 = ResidualBlock(64)

        self.conv_3_1 = ResidualBlock(128, 2)
        self.conv_3_2 = ResidualBlock(128)
        self.conv_3_3 = ResidualBlock(128)
        self.conv_3_4 = ResidualBlock(128)

        self.conv_4_1 = ResidualBlock(256, 2)
        self.conv_4_2 = ResidualBlock(256)
        self.conv_4_3 = ResidualBlock(256)
        self.conv_4_4 = ResidualBlock(256)
        self.conv_4_5 = ResidualBlock(256)
        self.conv_4_6 = ResidualBlock(256)

        self.conv_5_1 = ResidualBlock(512, 2)
        self.conv_5_2 = ResidualBlock(512)
        self.conv_5_3 = ResidualBlock(512)
        
        self.global_pool = GlobalAveragePooling2D()

        self.fc_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation = "softmax")

#    def call(self, x):
    def call(self, x, training = True):
        x = self.conv_1(x)
        x = self.max_pool(x)

        x = self.conv_2_1(x, training)
        x = self.conv_2_2(x, training)
        x = self.conv_2_3(x, training)

        x = self.conv_3_1(x, training)
        x = self.conv_3_2(x, training)
        x = self.conv_3_3(x, training)
        x = self.conv_3_4(x, training)

        x = self.conv_4_1(x, training)
        x = self.conv_4_2(x, training)
        x = self.conv_4_3(x, training)
        x = self.conv_4_4(x, training)
        x = self.conv_4_5(x, training)
        x = self.conv_4_6(x, training)

        x = self.conv_5_1(x, training)
        x = self.conv_5_2(x, training)
        x = self.conv_5_3(x, training)

        x = self.global_pool(x)

        return self.fc_3(x)


# From here we can define our resnet34.

# This is how we will do that.

resnet_34 = ResNet34()

# The next thing we will do is call our resnet 34 model with some parameters, like this.

#resnet_34(tf.zeros([1, 256, 256, 3]))
resnet_34(tf.zeros([1, 256, 256, 3]), training = False)

# Once we have called our model we can get the summary like this.

resnet_34.summary()

# This should return our model summary and show us the deatails of our model so far.

# With that being done we are going to move on to the training. But this time around we will include
#some check points.

# We will ensure that as we train we save our best model weights.

# This is how we will implement our model training with the checkpoints.

checkpoint_callback = ModelCheckpoint(
    "best_weights", 
    monitor = "validation_accuracy",
    mode = "max", 
    verbose = 1, 
    save_best_only = True,
)

# We started with our checkpoint callbcak, and if we need a reminder we can check back on previous modules
#for more details on the checkpoint callback.

# So we have our callback which will permits us total weights for our best performing weights.

# We also have a monitor that will monitor the validation accuracy.

# Before we move forward we will go back to the def call() which had the training parameter. 

# This is located in our custom conv2d class.

# We also have a batch normalization.

# It should be noted that with this batch normalization layer we have to specify whether we are 
#in training mode or in inference or testing mode. 

# The reason why we are doing this is because the parameters of the batch norm layer will react 
#differently or behave differently in these two different modes.

# This means that during the training, True means this layer will normalize the inputs with the 
#mean and variance with the current batch of inputs.

# False means we're in inference mode and the layer will normalize the input using the mean and
#variance of its moving statistics learned during training.

# So this means that we have a layer that we'll call [Alia], for example.

# So we have the parameter [Alia], and during training our layer updates these parameters, but then
#during inference we do not want to update these parameters as they were learned during training so
#we have to set the training to False when we are not training the model, or when we are evaluating
#the model, or testing the model.

# What that simply means is that we'll have to add training to our x = self.batch_norm(x). See line 348

# Once we do that we will manually set training to True inside our def call() parameters. See line 344

# Now let's go back to our residual block.

# We will manually add training to our def call() as a parameter with self and input. see line 370

# We will then add training to our x = self custom conv 1, 2, and, 3 as a parameter. See lines 373, 375, 379

# Now we will get back to our conv network and manually add training to our def call() and set it to true
#so we can use it for the rest of the residual blocks. See line 419

# Now we will get back to our conv network and manually add training to every self starting from
#self conv 2 1. See lines 423 - 441

# So now when we are not in training mode we could specify the training parameters such that the 
#batch norm layer parameters aren't modified.

# That's done.

# This time around we are going to add the training in our resnet 34() with the tf.zeros parameter,
#and set it to False. See line 457

# Now that that is set let's get back to our loss function.

loss_function = CategoricalCrossentropy()

# Then we will have our metrics.

metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k = 2, name = "top_k_accuracy")]

# Once we have our metrics we can compile our model, but this time around we will set a higher learning rate.

resnet_34.compile(
    optimizer = Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]*10),
    loss = loss_function,
    metrics = metrics)

# Now that we have compiled our model we can begin our training, using the history.

# We will also include the callbacks to add our model checkpoints.

# Also notice that we are specifying that we want to run our model for 9 epochs. See line 560.

history = resnet_34.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"]*3,
    verbose = 1,
    callbacks = [checkpoint_callback]
)

# Now our model is ready to be trained by running our code.

# Once the training is complete we can generate our model accuracy plot.

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"])
plt.show()

# Once we are done with our plots we can run our evaluation.

#resnet_34.load_weights("best_weights")

resnet_34.evaluate(validation_dataset)

# Now what if we load our best model and evaluate it.

# We will actually write this piece of code before we evaluate, so we will evaluate twice. 

# The first time is the original model, and the second time is with the best model.

# We'll leave the best model code commented out until we evaluate for the second time. See line 579

# The best weights parameter on line 579 should be in string form.

# Once we evaluate our model for the second time we can compare the accuracies between the two evaluations.

# Now we will test this and see the results.

test_image2 = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test/sad/17613.jpg")
im2 = tf.constant(test_image2, dtype = tf.float32)
im2 = tf.expand_dims(im2, axis = 0)
print(lenet_model(im2))
print(CLASS_NAMES[tf.argmax(lenet_model(im2),axis = -1).numpy()[0]])


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

# Now that we've retested our new model we will run the confusion matrix again.

predicted = []
labels = []

for im, label in validation_dataset:
    predicted.append(lenet_model(im))
    labels.append(label.numpy())


print(labels)

print(np.argmax(labels[:-1], axis = -1))

print(np.argmax(labels[:-1], axis = -1).flatten())

print(len(np.argmax(labels[:-1], axis = -1).flatten()))

print(np.argmax(labels[:-1], axis = -1).flatten())

print(np.argmax(predicted[:-1], axis = -1).flatten())


pred = np.argmax(predicted[:-1], axis = -1).flatten()


lab = np.argmax(labels[:-1], axis = -1).flatten()


cm = confusion_matrix(lab,pred)
print(cm)
plt.figure(figsize=(8,8))

sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
