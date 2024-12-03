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
    "N_DENSE_1": 1024,
    "N_DENSE_2": 128,
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


resnet_34 = ResNet34()

#resnet_34(tf.zeros([1, 256, 256, 3]))
resnet_34(tf.zeros([1, 256, 256, 3]), training = False)

resnet_34.summary()




# In this section we will be going over Feature Extraction

# In this section we will be treating transfer learning and fine tuning. 

# Transfer learning can be applied in several domains like computer vision, natural language processing,
#and speech.

# In order to better understand the usefulness of transfer learning, we have to take note that deep learning
#models work best when given a lot of data. 

# And so this means that if we have a data set of only a hundred data points, then we're most going to have
#a poorly performing model.

# But what if it's possible to train our model on a million data points, for example, and then use that model
#or adapt that in such a way that we can now train it on smaller data sets that will give us greater results.

# This is very possible with transfer learning, and that's what we shall be treating in this section.

# How is it possible to train a model made of a millon data points and then use that model on a smaller data
#set, which is obviously different from our million points data set.

# The answer to this question lies in the convnets.

# With the convnets we generally have two main sections.

# The first section is the feature extractor.

# And when we get towards the end on the convnet, we have the classifier.

# So the very first thing the convnet will want to do is extract low level features.

# And then as we get towards the end of the convnet, we focus on extracting more high level features.

# The classifier permits us to pick between a set of options, which one the model thinks the image actually
#is.

# Now because the convnet works this way, it means that if we have two data sets which are similar, then we 
#could build some sort of feature extractor which will extract features from the very large data set, and then
#because the weights have been tuned or have been trained in such a way that they extract features correctly, then
#when we pass in the very small data set, the extraction section of the model will do its job. 

# That is extracting useful features from the smaller data set.

# We would see that because the two data sets are similar, it's going to a geat job at extracting those features.

# So we will not need a very large data set in order to extact features from a small data set.

# And so in fact, what we're saying is we have a model, let's look at the example model below.

#   _______________
#  |               |
#  |               |
#  |               |
#  |               | 
#  |               |
#  |               |
#  |_______________|
#
#   _______________
#  |               |
#  |_______________|

# So we have a model, which is the whole example, and then out of the model we have a feature extractor unit, which
#is the larger portion of the model.

# And then we have the classifier unit of the model, which is the smaller portion.

# So now we have this small data set which we're going to pass into our model.

# Then we are going to get the outputs from our feature extractor unit.

# Now it should be noted that we generally use the concept of transfer learning when we a very small data
#set.

# And obviously since deep learning models perform best with larger data sets, we want to get the best out
#of them, and so we want to use the transfer learning when we have a small data set as we've said, and we 
#have this model which has been pre-trained to extract these useful features from those kinds of images.

# So this simply means that we should have two similar data sets. 

# Another advantage of using or working with transfer learning is that we get to gain in terms of training 
#compute cost.

# That means that the model which is pre-trained may have been trained for say three days, for example, and
#now all we need to do is just get the pre-trained model and then apply transfer learning on our own
#specific and smaller task.

# And so when we're running on limited budget we find that working with pre-trained models is going to be
#really helpful.

# Now apart from transfer learning we also have fine tuning which is quite similar in the sense that unlike
#with the transfer learning where we have the feature extractor's weights, which are fixed, and then during 
#training we update the weights of the classification section with fine tuning, what we could do is also 
#update the weights of the feature extractor section.

# Now generally we start fine-tuning from the top.

# So we'd have the input and then we would have the final layer.

# We would see that we'll start fine-tuning from the final layer to the initial layer.

# While carrying out fine-tuning, one thing we need to note is that we have to use a very small learning 
#rate.

# The reason why we need a very small learning rate is to avoid disrupting the weight values which have
#taken very much time to attain.

# And so as we do the fine-tuning we're gonna update the weights but very slowly.

# What we mean when we say update the weights very slowly is that we're gonna choose a very small learning
#rate, and then observe how that affects our models performance.

# At this point we'll get to code section of this module where we'll look at some pre-trained models.



# The model we will be looking at is the efficient net v4. 

# We are using this one because it has a slightly lower number of parameters compared to the resnet50 and 
#it outperforms the resnet50 by a very large margin.

# So we're going to pick this efficient net v4, and if there's one thing we can do with tensorflow is 
#we can simply use the models without having to code them out from scratch.

# We can copy and paste the code directly from the documentation.

# We will call it the back bone.

# We're not ging to include the top recall so we'll have to set it to False because it is True by default.

# The weights have been pretrained on the imagenet so we can leave as is.

# We're not going to use the input sensor so we can comment that out.

# Next we have the input shape which we will modify or configure to our image size configuration.

# Since we are not using the top as specified above we will be using the classes so we can comment that out.

# We will also be commenting out the classifier and the kwargs.

# The pooling layer we want use is the max avaerage, but we will comment it out for now and specify that later on.

# 

backbone = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights='imagenet',
#    input_tensor=None,
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3),
#    pooling=None,
#    classes=1000,
#    classifier_activation='softmax',
#    **kwargs
)

# Now that we have that we will do a freeze.

# What we will be doing is freezing the backbone in such a way that the weights aren't updated during
#training.

# We can do this by simply setting backbone trainable parameter to false.

backbone.trainable = False

# That's all we need to do to freeze our model.

# Now that we've frozen our model the next thing to do is to add the classification layer.

# Then we're going to define the input with the image size.

# Then we will pass in our backbone, which has been set to frozen for the training.

# Next we will move on to the Global Average Pooling.

# Then we have a Dense layer, which is set to 1024.

# The next thing we will have is the batch normalization

# This will be followed by the Dense layer 2, which is set to 128.

# And finally we will have one last dense layer with an activation of softmax.

# Also this dense layer won't be number but will have a parameter of num classes, which is set to 3.

# 

model = tf.keras.Sequential([
    Input(shape = (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3)),
    backbone,
    GlobalAveragePooling2D(),
    Dense(CONFIGURATION["N_DENSE_1"], activation = "relu"),
    BatchNormalization(),
    Dense(CONFIGURATION["N_DENSE_2"], activation = "relu"),
    Dense(CONFIGURATION["NUM_CLASSES"], activation = "softmax"),
])

model.summary()


# Now with this we have our model set, with minimal code,and we can go ahead and start our training.

loss_function = CategoricalCrossentropy()

metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k = 2, name = "top_k_accuracy")]


model.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)


history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1,
)

# When the training is over we can ahead and evaluate our model.

model.evaluate(validation_dataset)

# Now we will test our image out with our new model.

# Before we can test our image, we have to resize it. See line 687

test_image3 = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test/sad/17613.jpg")
test_image3 = cv2.resize(test_image3, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
im3 = tf.constant(test_image3, dtype = tf.float32)
im3 = tf.expand_dims(im3, axis = 0)

# Next we can print out the classification of the test of our image.

print(model(im3))

# Then we can print out the image classification label

print(CLASS_NAMES[tf.argmax(model(im3),axis = -1).numpy()[0]])

# Next we will reprint the first 16 images in our dataset again and see how accurately this model
#predicts their classifications

plt.figure(figsize=(12,12))

for images, labels in validation_dataset.take(1):
    for i in range(16):
        ax = plt.subplot(4,4, i+1)
        plt.imshow(images[i]/255.)
        #plt.title(tf.argmax(labels[i], axis=0).numpy())
        plt.title("True Label - :" + CLASS_NAMES[tf.argmax(labels[i], axis=0).numpy()] + "\n" + "Predicted Label - : " 
                  + CLASS_NAMES[tf.argmax(model(tf.expand_dims(images[i], axis = 0)),axis = -1).numpy()[0]])
        plt.axis("off")
        plt.show()


# Next we will get the confusion matrix metrics for this new model's image predictions.

predicted = []
labels = []

for im, label in validation_dataset:
    predicted.append(resnet_34(im))
    labels.append(label.numpy())


print(np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()]))

print(np.concatenate([np.argmax(predicted[:-1], axis = -1).flatten(), np.argmax(predicted[-1], axis = -1).flatten()]))

pred = np.concatenate([np.argmax(predicted[:-1], axis = -1).flatten(), np.argmax(predicted[-1], axis = -1).flatten()])

lab = np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()])


cm = confusion_matrix(lab,pred)
print(cm)
plt.figure(figsize=(8,8))

sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


# One thing we need to remember is that our dataset was not that small so we may not see the change 
#or the difference between training from scratch using transfer learning.
