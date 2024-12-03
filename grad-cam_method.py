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

test_image = cv2.imread("C:/Users/alpha/Deep Learning FCC/Human Emotion Images/test/sad/17613.jpg")
test_image = cv2.resize(test_image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
im = tf.constant(test_image, dtype = tf.float32)
img_array = tf.expand_dims(im, axis = 0)

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


vgg_backbone = tf.keras.applications.VGG16(
    include_top = False,
    weights= 'imagenet',
#    input_tensor = None,
    input_shape = (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3),
#    pooling = None,
#    classes = 1000,
#    classifier_activation = 'softmax'
)


vgg_backbone.summary()



# In this section we will be going over the Grad-Cam Method.

# The Grad-Cam Method will allow us to visually explain how the deep neural networks work.

# This was first developed in the paper by Ram Persad et al and entitiled "Grad-Cam, Visual Explanations
#from deep networks via Gradient-based Localization."

# First we'll see an original image (image of Dog and Cat) as input.

# Then we'll see that image with a Grad-Cam ouput.

# Now the task we have is a classification task where we are trying to see whether there's a cat or a dog
#in the image.

# And then with the Grad-Cam for the "Cat", we're able to detect the portion of the image which influenced
#the model to say that there's a cat in the image.

# And also, for teh Grad-Cam Dog, we have the part which shows us that portion influences the model the
#most in detecting or seeing that there is a Dog in the image.

# Now we also have the Grad-Cam for different models.

# We see a Grad-Cam for the vgg model we have.

# We would also have a resnet model Grad-Cam.

# We would see that the two different models can produce two different Grad-Cams.

# Though generally they should be similar, we said with a resnet we'll have a larger surface as
#compared to a vgg, which will have a smaller surface.

# We would also see that for the "Cat" image.

# Now with that being said, let's implement the Grad-Cam technique which permits us to generate these
#kinds of visualizations which tells us what part of the input influenced the model in making certain
#decisions.

# Grad-CAM Overview from the documentation:
#   :Given an image and a class of interest (e.g "tiger cat", or any other type of differentiable output)
#as input, we forward propagate the image through the CNN part of the model and then through task-specific
#computations to obtain a raw score for the category. The gradients are set to zero for all classes except
#the desired class (tiger cat), which is set to 1. The signal is then backpropagated to the rectified
#convolutional feature maps of interest, which we combine to compute the coarse Grad-CAM localization 
#(blue heatmap) which represents where the model has to look to make the particular decision. Finally,
#we pointwise multiply the heatmap with guided backpropagation to get Guided Grad-CAM visualizations
#which are both high-resolution and concept-specific.

# Now that we have an understanding, we will dive back into the code to use the Grad-CAM practically.


backbone = tf.keras.applications.efficientnet.EfficientNetB5(
    include_top = False,
    weights = "imagenet",
    input_shape = (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3)
)

backbone.trainable = False

input = Input(shape = (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3))

x = backbone.output

x = GlobalAveragePooling2D()(x)
x = Dense(CONFIGURATION["N_DENSE_1"], activation = "relu")(x)
x = Dense(CONFIGURATION["N_DENSE_2"], activation = "relu")(x)
output = Dense(CONFIGURATION["NUM_CLASSES"], activation = "softmax")(x)

pretrained_model = Model(backbone.inputs, output)

pretrained_model.summary()

# Now that we know our model can compile, let's move forward and make this model out of the full model
#we have.

# In order to do create the last conv layer which is essentially the model where we would get the output
#to be a rectified conv feature maps.

# We have the keras model which takes as input the pretrained model inputs, so it's quite similar 
#to what we had already where we just take the inputs to be the backbone inputs.

# Now the inputs are the full pretrained model inputs, but, then we notice that the output is the last
#conv layer output.

# What is the last conv layer.

# The last conv layer is the layer whose name is given as top activation.

# We'll notice that that is the last conv layer where we have the feature maps, which is 8 by 8 by 2048.

# From there we move on to the global average pooling.

# Then we have our Dense layers one and two.

# And then we have our final Dense layer which is our output layer.

# So everything from the top activation and above is our initial conv layer, or the conv 
#convolutional neural network.

# Everything below is considered the classifier unit. 

# And so now that we have the name of the last conv layer, that's a top activation, we're going
#to make use of that to produce our output.

# We see that the last conv layer is simply the pretrained model, and we get the layer whose name
#is last conv layer name, which is top activation.

# So now that we have our last conv layer as we've said before, we are now able to have the last
#conv layer model or the initial cnn model which has as input the image and as output the feature
#maps.

# So let's run this and see what we get.

# This is the code we will be running.

last_conv_layer_name = "top_activation"
last_conv_layer = pretrained_model.get_layer(last_conv_layer_name)
last_conv_layer_model = tf.keras.Model(pretrained_model.inputs, last_conv_layer.output)

last_conv_layer_model.summary()

# Now we can view our last conv layer model summary.

# This is where we will see that our input is our image and the output is our feature map.


# Now that we've built this first model from our overall model, let's go ahead and check out our
#classifier model.

# So for the classifier model we'll see that we have our inputs.

# The input now is going to be our feature maps.

# This is the code we will use to implement our classifier model.

#classifier_layer_names = [
#    "global_average_pooling2d",
#    "dense",
#    "dense_1",
#    "dense_2"
#] 

classifier_layer_names = [
    "global_average_pooling2d",
    "dense_3",
    "dense_4",
    "dense_5"
]


classifier_input = Input(shape = (8, 8, 2048))
x = classifier_input
for layer_name in classifier_layer_names:
    x = pretrained_model.get_layer(layer_name)(x)
classifier_model = Model(classifier_input, x)

# The next step will be to compute the partial derivatives of the output with respect to the features.

# Now we could compute these partial derivatives or these gradients making use of the gradient method 
#which takes in the top class channel, and the last conv layer output.

# This is the code we can use to help us compute the partial derivatives.

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    print(top_pred_index)
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)

grads.shape

# Once we have the gradients the next thing we could do is simply obtain the mean values at every position.

# To help us do that we will make use of the reduced mean while specifying the axis.

# This is the code we will be using.

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

# Now let's run this and print our pooled grads shape to see what we get.

# This is how we will get our pooled grads shape with print.

print(pooled_grads.shape)

# This shape is basically a vector now where each position is a mean of a single channel.

# We have a 2,048 dimensional vector, which is what we are seeing when we see the shape of 2,048.

# The next thing we will want to do is take our dimensional vector, and multiply it by our features maps.

# We can carry out this multiplication by using a for i in range method.

# This is the code that we will using with our in rangr method.

last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(2048):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

# So for each and every one of the different positions we have, we are going to take the corresponding
#last conv layer output, which is essentially the features maps.

# Now we can run this to obtain our last conv layer output.

# Now that we have this, we can print out the shape of last conv layer.

# This is how we will do that

print(last_conv_layer_output.shape) 

# The next thing we will be doing is obtaining the heat map.

# We are going to sum up the values at different positions for all the channels,

# So let's suppose that we have something like this for example, 

# This is the code we wil use to handle this process.

#Note: When we sum this, we are going too specify that the axis is a channel axis. 

heatmap = np.sum(last_conv_layer_output, axis = 1)

# The next thing we want to do is to run our code to make sure it is compiling.

# Once that is done we can move to the next step which is visualizing our heatmap.

# This is the code that will help us do that.

heatmap = tf.nn.relu(heatmap)
plt.matshow(heatmap)
plt.show()

# At this point we will run our code again to make sure that everything is still compiling.

# Once we are sure that our code is still running smooth, we can move forward.

# The next thing we will want to do is resize our image.

# To do this we will use an open cv resize method.

# This is the code we will use.

resized_heatmap = cv2.resize(np.array(heatmap), (256, 256))
plt.matshow(resized_heatmap)
plt.show()

# We will once again compile our code at this point.

# The reason we are using these heatmaps is because we will be adding them to our image,
#or purpose them in such a way that we have an output which shows us better clear regions of
#the image where we have the highest contribution to our output, which is that of a happy person
#(in this case).

resized_heatmap = cv2.resize(np.array(heatmap), (256, 256))
plt.matshow(resized_heatmap*2555+img_array[0,:,:,0]/255)
plt.show()

# So the model predicts that the person is happy (in this case), and now we know which parts of the 
#image contributed the most to that prediction.

# Now we will try it out with a different image and modifying this slightly so that we could get 
#the image from our array.

resized_heatmap = cv2.resize(np.array(heatmap), (256, 256))
plt.matshow(resized_heatmap*100+img_array[0,:,:,0]/255)
plt.show()

# We should see a much larger reion of hotzones on the heatmap now over our image.