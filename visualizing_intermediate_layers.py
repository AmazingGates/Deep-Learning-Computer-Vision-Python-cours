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

# In this section we will be Understanding the Blackbox [Human Emotions Detection].

# We will start with Visualizing Intermediate Layers.

# We will visualize the convolutional neural networks feature map.

# One very imortant part of building robust deep learning models involves understanding how these models
#work, or understanding what goes on in the different layers.

# And so in this section we'll focus on taking a model which has already been pre-trained and then 
#generating these feature maps, so we get to see exactly what goes on under the hood.

# The pre-trained model we will be using here will be the vgg16.

# So we'll simply copy the sample code from the tensorflow documentation and begin our code.


# We won't be using anything after the input shape so we commented it out.

# We will be commenting out the input tensor also.

# Also we will set the include top to False.

# The input shape will get defined with our configuration im size.

# Now we will assign our updated function to a variable called vgg_backbone.

vgg_backbone = tf.keras.applications.VGG16(
    include_top = False,
    weights= 'imagenet',
#    input_tensor = None,
    input_shape = (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3),
#    pooling = None,
#    classes = 1000,
#    classifier_activation = 'softmax'
)

# We can check the vgg backbone by running a summary on it.

vgg_backbone.summary()

# Once we have our summary we can move on to the next step, where we are going to create a model which will
#permit us to visualize the feature maps

# Now to explain how this works, let's recall that we have the vgg.

# So we have our vgg,and what the vgg does is takes in an input image.

# So we have an input image and then it produces a single output.

# Since we only have one single output and we are interested in visualizing the hidden layers
#that go inside the vgg model, what we'll do now is we'll create is reate a new model which instead
#has many different outputs. 

# These differet outputs will come from the different hidden layers.

# So we could take a single layer from our hidden layers, and turn that into an output.

# We can do the same thing with every hidden layer in our model.

# So basically, the hidden layers of our feature maps will become our outputs.

# So we'll now have an input with zero params, that can have many different outputs, Instead
#of just a single output.

# We have about 17 different outputs now(as shown in our summary).

# We may also decide to pick specific outputs, so we may want to take only half the conv layers.

# Only the outputs of the conv layers, so we might want to omit the max pool layers.(See summary in
#command line).

# This all depends on us, and we'll see how we can do this when we get back to the code.

# This is where we will define our is_conv.
def is_conv(layer_name):
    if "conv" in layer_name:
        return True
    else:
        return False
    
# This is where we will define our is_conv to only count the max pool layers.
#def is_conv(layer_name):
#    if "pool" in layer_name:
#        return True
#    else:
#        return False

# Next we're going to start by building a feature_map.

# And then we'll put it in a list.

# Inside the list we will have a "for statement" that will get us every layer in our vgg backbone.

#feature_maps = [layer.output for layer in vgg_backbone.layers[1:]]
feature_maps = [layer.output for layer in vgg_backbone.layers[1:] if is_conv(layer.name) == True]

#feature_maps = [layer.output for layer in vgg_backbone.layers[1:4]]

# So we'll take the layers and from there we will build a new model.

# We'll call this new model "Model" and store it to variable called features map model.

# From here the Model function takes as parameters the vgg_backbone.input as inputs, and the
#feature map which will be stored in our outputs.

feature_map_model = Model(
    inputs = vgg_backbone.input,
    outputs = feature_maps
)

# Then we'll view the features map models summary.

feature_map_model.summary()

# Once we see the result we will see that they similar to what we have.

# Let's say we for example that we picked from layer one to layer four, see line 309, 
#our summary would come back shortend.

# This is because we are telling our model that we only want to get the layers 1 up until 4.

# And the difference in the models is that now we have many outputs and have just one single input,
#unlike before when we had one input abd one output.


# Now let's move on to passing an input through this model.

# To do this, what we want to do is take an input image, and pass it into our model.

# And now since our model outputs the different feature maps, with the different hidden layer outputs 
#will now be able to visualize what's going on inside the vgg model.

# Now to get this output we are going to use something similar to the testing that we've seen already.

# Recall how we carried out the testing, where we take an image and test on it using the test_image.

# We will simply copy and paste the test_image here so we can reuse it, and pass it to our model to get
#the output.

# But now in our case, we will reduce the model.

# We will use the model with our newly created feature map model.

# This is how we will do this.


test_image = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test/sad/17613.jpg")
test_image = cv2.resize(test_image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
im = tf.constant(test_image, dtype = tf.float32)
im = tf.expand_dims(im, axis = 0)

f_maps = feature_map_model.predict(im)


# This is where we will check our length
print(len(f_maps))

# Now we will check our features map and its length.

# Then we want to print out the features maps shape.

for i in range (len(f_maps)):
    print(f_maps[i].shape)

# All of the returns that we get back should start from one.

# This is because we don't want to include the input layer.

# So we'll get every layer starting from one right up until the last layer.

# Now we've picked the conv layers and the max pool layers, and with these we can now visualize 
#the different feature maps.

# Now we'll take note of the length by printing it out. See line 383.

# The length will represent how many different outputs we will have.

# Next we will modify the feature_map in such a way that we only do this if the conv layer 
#name is True. see line 322 for modification.

# If it is True, we are going to attach this to the outputs.

# What this means is that we are only going to take the conv layers as part of our output.

# Then we would define our is_conv. See line 302

# The is_conv will take as parameter the layer name. Then we're going to say that if a "conv"
#name in layer, return True.

# else return False.

# We can also do the same thing for the max pool. 

# We can modify our model to only count the max pool layers and excclude the others. See line 309

# Now we set our model back to count only the conv layers and move forward.

# Next we will carry out the final visualization.

# We are going to go through each and every feature map.

# So for i in range of the length of the feature map, we create the figure, and then we specify the 
#figure size wil be 256 by 256.

# And since we're going through each and every feature map, it is important for us to get the feature size.

# So we want to get the values for each feature map.

# Now with this we simply have f maps, we have i, and then we pick one feature map to get the shape of.

# Next we will get the number of channels.

# So we'll have the number of channels equal to the feature maps i shape 3.

# 3 because this is a zero one two.

# So this is how we get the number of channels.

# Now that we have this set already, we want to be able to visualize this in such a way that all the 
#channels are lined in a single line.

# To do this we will create another array called a joint maps.

# We'll use the numpy to initialize and then to size.

# We'll also add a height, which is 256.

# Then we'll times feature size times number of channels.

# That's how we do that, and so with that we now have our joint maps initialized.

# The next thing to do will be to fill in the values of our outputs and features in this array.

# Now we understand how the joint maps was created, we will move ahead to fill in the information for
#all the different features in the giant maps array.

# So next we'll go through the different channels.

# To do this we'll use another for loop using a for j in range of number of channels.

# Basically the number of channels we have that, and then we fill in the joint maps.

# The way we'll do this is we'll keep the height fixed.

# So we'll have the heights fixed and then in the width dimension, we'll fill in the information in
#such a way that as we go from one channel to another, we are going to skip steps of 256.

# So we'll have feature size, which is 256, times j, then feature size times j plus one.

# This will allow us to fix our height.

# Once we have this we will go ahead and fill in the data we have.

# We'll have the feature maps and then we take in i.

# If we consider the case where j equals zero.

# That means we picked a zone from 0 to 256.

# Then we're going to take a particular feature map, by way of the [i], and then once we pick the particular
#feature map we can go ahead and set the outputs to the values of the feature maps while selecting the 
#particular channel.

# So now when j equals zero for example, we take the zeroth channel and pick out all the values which 
#come before and then pick out now the zeroth channel.

# That's it. Now we have our joint maps which has now been created.


for i in range(len(f_maps)):
    plt.figure(figsize= (256, 256))
    f_size = f_maps[i].shape[1]
    n_channels = f_maps[i].shape[3]
    joint_maps = np.ones((f_size, f_size * n_channels ))

    axs = plt.subplot(len(f_maps), 1, i + 1)

    for j in range(n_channels):
        joint_maps[:, f_size*j : f_size*(j + 1)] = f_maps[i][..., j]


# Now we are ready to plot our images.

# We'll start with the plt imshow method which will take as a parameter joint maps.

# If we want to pass all the joint maps this going to be very it would be very ram consumming, so we just
#select all the heights and then we'll pick some values.

# We'll pick from 0 to 512, for example, and run this.

# But before we run this we want to set the different axis points.

# We'll do this above the j loop.

# The axis will hold the plt sublots length of the feature maps.

# So basically if we have certain feature maps then we're gonna have different subplots. see line 503

# Now once we have all of this set we can run this.


plt.imshow(joint_maps[:,0:512])
plt.axis("off")
plt.show()

