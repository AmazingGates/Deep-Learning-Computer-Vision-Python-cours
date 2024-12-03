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


backbone.trainable = False


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


model.evaluate(validation_dataset)


test_image3 = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test/sad/17613.jpg")
test_image3 = cv2.resize(test_image3, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
im3 = tf.constant(test_image3, dtype = tf.float32)
im3 = tf.expand_dims(im3, axis = 0)


print(model(im3))


print(CLASS_NAMES[tf.argmax(model(im3),axis = -1).numpy()[0]])


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

# In this section we will be going over the concept of Finetuning

# Before we get into the finetuning we will first convert our code which was built with sequential API,
#to a functional API.

# Here's our converted model.

# It's basically the same thing where we have the input, a backbone which takes in the input and produces
#output, with our Global Average Pooling, dense layer, batch norm, dense layer, and our final dense layer
#which is our output layer.

# Then we would have our fine tuned model.

# We can then view our fine tuned model summary which should be identical to what we had already with the 
#pre-trained model.

# After running this we should have the same exact model that we previously built with the sequential API.

# Now we want to fine tune our model with all the layers which were frozen and not trained, we now want to
#make them trainable.

# So we'll simply go back and set the backbone trainable to True.

#backbone.trainable = True
#backbone.trainable = False

# Then we are going to set the trainable to false inside our back bone function 

input = Input(shape = (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3))

x = backbone(input, training = False)

x = GlobalAveragePooling2D()(x)
x = Dense(CONFIGURATION["N_DENSE_1"], activation = "relu")(x)
x = BatchNormalization()(x)
x = Dense(CONFIGURATION["N_DENSE_2"], activation = "relu")(x)
output = Dense(CONFIGURATION["NUM_CLASSES"], activation = "softmax")(x)

finetuned_model = Model(input, output)

finetuned_model.summary()

# Before moving forward we should note that setting layer trainable to False is different than setting
#training to False.

# When we set a layers trainable to False we are simply saying we do not want to update the weights 
#when training.

# When we set training to False, it means we want to work in inference mode..

# In the case of the batch norm, the gamma and the beta are trainable parameters and so when we say
#layer trainable equal False, it means that they are not going to be updated during training, but on
#the other hand the mean and variance aren't trainable parameters, instead they are parameters which
#adapt to the training data.

# That is why when we add inference mode, that's when we set training to equal False, we do not want
#to disrupt the mean and the variance values we got during the training based on the training inputs.

# And so as we saw already, the mean and variance when in inference mode will simply be the moving
#average of the mean and standard deviation of the batches it has seen during training.

# So clearly these two concepts, layers trainable equal False and training equal False, are different
#and have different functionalities.

# Nonetheless it should be noted that in the case of the batch norm, setting trainable to False on
#the layer means that the layer will be subsequently run in inference mode.

# Now although weve seen that the two aren't the same, we should also note that setting trainable on
#a model containing other layers will recursively set the trainable value of all inner layers.

# If the value of the trainable attribute is changed after calling compile() on a model, the new 
#value doesn't take effect for the model until compile() is called again.

# With that being said we can move on to the dropout.

# The dropout layer doesn't have any trainable parameters, but remember the way that the dropout
#works is like this:

# : let's say we have some inputs and we were to pass them to the dropout layer, at the end some of
#the inputs won't be considered, as specified by the dropout rate.

# Let's say that we had a dropout of 0.5. This would indicate that half of our inputs would not be 
#considered.

# So we see that with inference when we are actually trying to test our model, we do not need to dropout
#any of the neurons/inputs.

# So generally the dropout also takes in the training parameter where when we set training to True, it 
#means we are in training mode and so we could actually dropout some of the values, whereas when we 
#set the training to False then we are in inference mode and so we do nothing. We allow the inputs
#to pass without any modifications.

# And we could also see cleary that the trainable layer doesn't apply because dropout doesn't have 
#trainable parameters, whereas with the training we could decide whether it's true or false, indicating
#training or inference modes. 

# So in fact what we're saying is that we have a model, which has a backbone, and it has a head for
#classification, we apply transfer learning by freezing all of our backbone so no parameter in the model
#is updating during training.

# Then we move on to fine tuning, where in finetuning we want to update the parameters with a very small
#learning rate, and then we also want to avoid a situation where the mean and variance statistics 
#which we got during the training process will be upset during the fine tuning process.

# And so the batch norm is kind of like a special layer where even during the fine tuning where we want
#or where we set the trainable to true, that is we want to update the weights during the training,
#we do not want to modify or upset the batch norms mean and variance.

# And so we are going to set the training to False (see line 343), so it still behaves as if it were
#in inference mode.

# Getting back, we have our training which we will start by setting our backbone to False. see line 575
