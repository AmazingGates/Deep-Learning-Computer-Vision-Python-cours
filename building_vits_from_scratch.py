import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer, Dropout, Rescaling, Resizing, RandomFlip, RandomContrast, RandomRotation, MaxPooling2D, GlobalAveragePooling2D, Activation, Add, Embedding, LayerNormalization, MultiHeadAttention
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
    "NUM_CLASSES": 3,
    "PATCH_SIZE": 16
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

last_conv_layer_name = "top_activation"
last_conv_layer = pretrained_model.get_layer(last_conv_layer_name)
last_conv_layer_model = tf.keras.Model(pretrained_model.inputs, last_conv_layer.output)

last_conv_layer_model.summary()

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

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    print(top_pred_index)
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)

grads.shape

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

print(pooled_grads.shape)

last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(2048):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

print(last_conv_layer_output.shape) 

heatmap = np.sum(last_conv_layer_output, axis = 1)

heatmap = tf.nn.relu(heatmap)
plt.matshow(heatmap)
plt.show()

resized_heatmap = cv2.resize(np.array(heatmap), (256, 256))
plt.matshow(resized_heatmap)
plt.show()

resized_heatmap = cv2.resize(np.array(heatmap), (256, 256))
plt.matshow(resized_heatmap*2555+img_array[0,:,:,0]/255)
plt.show()

resized_heatmap = cv2.resize(np.array(heatmap), (256, 256))
plt.matshow(resized_heatmap*100+img_array[0,:,:,0]/255)
plt.show()


# In this section we will be going over the process of building ViTs from scratch.

# In the previous section we saw how the Transformer model was used for NLP tasks could be used
#in computervison with proper preparations.

# And so in this section we'll see how to convert patches of an image and then 
#carry out the linear projections and pass the output from those linear projections into the 
#Transformer encoder to then create or to then train an end-to-end model which learns how to see
#whether an input image is that of an angry person, a happy person, or a sad person.

# Given that we've been working with 256 by 256 images, if we have to split up thse images into
#16 by 16 images, then we would have 256 different iamges.

# This is because if we have 256 images and then break them into 16 by 16, which would give us 
#16 by 16 pixels, then we would have to go 16 times vertical and 16 times horizontal to form
#a 256 by 256 image.

# And so from here we would have 256 different patches.

# So what we'll use to create these patches will be the extract patches method.

# Now the extract patches method takes in the (images, sizes, strides, rates, padding) as parameters.

# Before we look at an example, let's go over some of the args.

# 1. images - A 4-D Tensor with shape [batch, in_rows, in_cols, depth]

# 2. sizes - The size of the extracted patches. Must be [1, size_rows, size_cols, 1]

# 3. strides - A 1-D Tensor of length 4. How far the centers of two consecutive patches are in the 
#images. Must be [1, stride_rows, stride_cols, 1]

# 4. rates - A 1-D Tensor of length 4. Must be [1, rate_rows, rate_cols, 1]. This is the input 
#stride, specifying how far two consecutive are in the input. Equivalent to extracting patches
#with patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1), followed by subsampling 
#them spartially by a factor of rates. This is equivalent to rate in dilated (aka Atrous) convolutions.
 
# 5. padding - The type of padding algorithm to use.

# 6. name - A name for the operation (optional)

# So let's look at how this works.

# Let's take a look at the example output below.

# This output will help us to picture how this works.

# *      *      *      4      5      *      *      *      9      10
# *      *      *      14     15     *      *      *      19     20
# *      *      *      24     25     *      *      *      29     30
# 31     32    33      34     35     36     37     38     39     40
# 41     42    43      44     45     46     47     48     49     50
# *      *      *      54     55     *      *      *      59     60
# *      *      *      64     65     *      *      *      69     70
# *      *      *      74     75     *      *      *      79     80
# 81     82     83     84     85     86     87     88     89     90
# 91     92     93     94     95     96     97     98     99     100

# tf.image.extract_patches(images=images,
#                          sizes = [1, 3, 3, 1],
#                          strides = [1, 5, 5, 1]
#                          rates = [1, 1, 1, 1]
#                          padding = "VALID"
#)

# The tf image method above will be the first piece of live code we will be implementing in
#this module.

# We will save this method to a variable called patches.

# Then we will pass in a tf.expand dimension method and pass test iamges and axis = 0 as parameters
#to be stored in a variable called images. 

# Next we will have our 16 by 16, which will be the patch size.

# The stride is also 16 because we want to have it match the value of the sizes, which will
#allow us to have compact patch extractions.

#patches = tf.image.extract_patches(images = tf.expand_dims(test_image, axis = 0)
#                                   sizes = [1, 16, 16, 1],
#                                   strides = [1, 16, 16, 1],
#                                   rates = [1, 1, 1, 1],
#                                   padding = "VALID"
#)

# With that being done, let's configure our patch size.

patches = tf.image.extract_patches(images = tf.expand_dims(test_image, axis = 0),
                                   sizes = [1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
                                   strides = [1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
                                   rates = [1, 1, 1, 1],
                                   padding = "VALID"
)

# Now we should have our patches, but to check, we will print out our patch shapes to makee sure
#everything is working properly.

print(patches.shape)
#patches = tf.reshape(patches, (patches.shape[0], 256, 768))
patches = tf.reshape(patches, (patches.shape[0], -1, 768))
print(patches.shape)

# Now let's explain why we have the shape size of (1, 16, 16, 768).

# Recall that we have a 256 by 256 by 3 image, meaning that we're going to have 3 channels with the
#(256, 256, 3).

# Since we're dealing with patches, it's preferable that we should get our 16 by 16 patches first from 
#the 256 by 256 images we have.

# This will give us a 16 by 16 patch for each channel we have.

# Given that each and every one of those patches is 16 by 16, and 16 by 16 is 256, we'll see that
#if we pick a given patch like the ones we have, which are 16 by 16, then the third dimension will
#be 768, simply because for each patch we have 256 pixels per channel.

# So what we have is 16 by 16 = 256 pixels, which make up our entire patch.

# The same thing applies each of our 3 channels.

# This is how we get the value of 768 in our shape size.

# Now to plot this out we are going to go through each and every patch.

# So we'll take it both vertically and horizontally to create the subplots 16 by 16
#because we'll have 16 by 16 different subplots.

# Then for each subplot we'll have an image where we'll pick a given patch out of the 256 patches.

# We pick the patch and then we're going to reshape it because when we pick the patch, we're left
#with the 768 value patch.

# We're left with those pixels, which have been flattened out to the 768 dimensional vector 
#that we calculated.

# This is how we will use and reshape our plots.

#plt.figure(figsize = (8, 8))
#k = 0
#for i in range (16):
#    for j in range (16):

#        ax = plt.subplot(16, 16, k + 1)
#        plt.imshow(tf.reshape(patches[0, i, j, :], (16, 16, 3)))
#        plt.axis("off")
#        k += 1
#        plt.show()


# Next we will create our patch encoder.

# To do this we'll have to convert or reshape our patches.

# Recall that in the paper, after creating the patches we have to take each and every one of them
#and consider them as one element of the sequence.

# To help us do this we will reshape our patches in such a way that each and every one of them
#will be considered as a part of our whole sequence.

# This is how we will carry out the reshaping process. see line 432.

# Then we want to check our new shape by printing it. see line 433.

# Now each of the 256 will be passed to the Transformer.

# Now we can plot this modified version of our images.

# This is the code we will use to create our patch encoder.

plt.figure(figsize = (8, 8))
for i in range (patches.shape[1]):

        ax = plt.subplot(16, 16, i + 1)
        plt.imshow(tf.reshape(patches[0, i, :], (16, 16, 3)))
        plt.axis("off")
        plt.show()

# With this being done, we are now ready to craete our patch encoder.

# This patch encoder layer will be similar to the kind of layers we've been creating so we
#can just copy and paste the code we will be using.

# We will be copying and pasting the reduced function we used previously.

# This is the code we will be using.

class PatchEncoder(Layer):
    def __init__(self, N_PATCHES, HIDDEN_SIZE):
        super(PatchEncoder, self).__init__(name = "patch_encoder")

        self.linear_projection = Dense(HIDDEN_SIZE)
#        self.postional_embedding = Embedding()
        self.positional_embedding = Embedding(N_PATCHES, HIDDEN_SIZE)
        self.N_PATCHES = N_PATCHES
    def call(self, x):
        patches = tf.image.extract_patches(
            images = x,
            sizes = [1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
            strides = [1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
            rates = [1, 1, 1, 1],
            padding = "VALID")
#        patches = tf.reshape(patches, (patches.shape[0], -1, patches.shape[-1]))
        patches = tf.reshape(patches, (CONFIGURATION["BATCH_SIZE"], 256, patches.shape[-1]))
        embedding_input = tf.range(start = 0, limit = self.N_PATCHES, delta = 1)
        output = self.linear_projection(patches) + self.positional_embedding(embedding_input)
        patches.shape[-1] # This will make our patches more dynamic. The value we used is for the last
        #dimension of all our patches.

#        return x
        return output

# We'll call this our patch encoder layer. 

# Now we have our patch encoder.

# This patch encoder will be responsible for converting our image into patches, and then projection,
#and adding the positional encoding patch encoder.

# And now we have the linear projection which we will create. 

# We will do that by saving our Dense Layer function to a variable called linear_projection.

# To see how we can do this see line 524.

# We can also select our hidden size dimension, or embedding dimension.

# Now since our input is already set at 256 by 768, this is our hidden size dimension, for now.

# What we could do is we could convert 768 into 512.

# Inside our Dense function we will pass the hidden_size dimension as a parameter.

# We will also pass the hidden_size dimension parameter to the constructor method, see line 521

# Now we have our linear projection and our hidden size.

# Now that we have all of this we are ready to add up the positional embeddings.

# We should also note that we are not going to take into consideration the extra input that we have
#([0, *]).

# So we would add all of our positional embeddings (excluding the extra input) onto on our
#Linear Projected Patches.

# The positional embeddings will now be constructed using Tensorflow embeddings layer.

# So we would have our positional embedding, or encoding, as described in the Paper we refrenced.

# This will be used the variable we use to store our Embedding layer. see line 525

# When we look at the documentation, the Embedding method is described as such:
# Turns positive integers (indexes) into dense vectors of fixed size.

# Now getting back to the paper, we will be looking at our linear projections, which we will 
#be adding to the positioning coding.

# The linear projection we are going to get from this process, if we include the batch dimension,
#we'll have an output batch size by number of patches.

# We can represent that like this (B, Np). We're using Np because obviously we're refering to
#the number of patches we have our image broken down in to. 

# Each patch is going to be a vector.

# So it's number of patches by that last dimension, which is 768.

# This is what our complete shape looks like (B, Np, 768)

# Now we have our positional embeddings, and our Embeddings Layer.

# Now let's check out how the Embeddings layer is constructed.

# First, the Embedding layer, as we saw from the documentation, turns positive integers
#into dense vectors of fixed size.

# Getting back to the reference paper.

# We would see that at some point we would have our linear projections, which we need to be
#added up to the positioning coding.

# Now the linear projection we're going to get from this if we include the bacth dimensions
#will have an output batch size by number of patches. 

# As we stated previously, we will call this B by Number of Patches by last dimension of the shape.

# (B, Np, 768)

# Note: We can always modify our last dimension in such a way that will be easier for programmer
#or user to understand.

# We can can display our modified shape like this, (B, Np, H). Here, the H inside our shape stands
#for Hidden Dimension.

# So let's say we have a B of 1, or batch that equals 1.

# We would have a one as a representation for the batch dimension, then the number of patches we
#would have is 256.

# This is because if we break up our 256 by 256 image, we would get patches of 16 by 16 pixel
#patches, which will give us 256 different patches.

# This is how our shape can be displayed now, (1, 256, 768).

# Now with the embedding layer, what we'll be able to get will be another tensor, which is like this
#(1, 256, 768), in the output of the embedding layer.

# So as we would see, it takes the indices and then turns them into dense vectors, which we're
#interested in.

# Now let's look at an example that we can test out with some live code.

#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Embedding(1000, 64, input_length = 10))
# The model will take as input an integer matrix of size (batch, input_length),
#and the largest integer (i.e. word index) in the input should be no larger than
#999 (vocabulary size).
# Now model.output_shape is (None, 10, 64), where "None" is the batch dimension.

#input_array = np.random.randint(1000, size = (32, 10))
#model.compile("rmsprop", "mse")
#output_array = model.predict(input_array)

# Note: Since we are working with integers we have no need for the vocabulary, so we'll
#use number of patches instead.

# If we were dealing with NLP (Natural Language Processor), Vocabularty would be our focus.

# Now we can print out our new shape to see what we have from this new input.

#print(output_array)
#print(output_array.shape)

# As we can see, our input takes this, (32, 10), and gives us a vector as an output.

# So basically what we have are some indices.

# In this case let's say that we have 10 indices.

# Let's also say that we just have the 10 indices and no batch dimension.

# So these 10 indices now could be projected into its two dimensional version, where each
#indice will be represnted by a vector.

# So just like the patches where we had patches represented by vectors, each indice is also 
#represented by vectors. 

# So basically, they are no longer represented by an index.

# That's because the random values takes values from 0 to 1000, which is why it can no longer
#support the index which will be converted into a vector.

# The size of the vector will be based on the value we set in our tf.keras.layers.Embeddings(), 
#which in our case was (64). see line 643

# This means that each one of our 10 indices will be converted into a 64 dimensional vector.

# That's why we would go from the 10, to the 10 by 64 shape we get after we exclude the batch
#dimension.

# So what we'll be doing in our case now, is modifying the 64 dimensional we set, and change
#it to the 768 dimensional.

# We will also modify our 1000 parameter and changge it to 256.

# Then we will modify the size of the input, which is going to be some random input.

# We'll set the size to 32 by 256, since we have 256 patches.

# We can also remove the input_length from the Embeddings().

# The reason why we took off input_length and why we would need can be found in the documentation,
#but we will not be needing it here.

# Now we expect to have an output which is 32 by 256 now by 768.

# Let's modify our size one more time and set it to 1 by 256 (1 = batch).

# Now let's run this and see what we get.

model = tf.keras.Sequential()
#model.add(tf.keras.layers.Embedding(256, 768))
model.add(tf.keras.layers.Embedding(1000, 768))
input_array = np.random.randint(1000, size = (32, 256)) # Correction, 1000 was the correct word index count, 
#not 256. (I put 256 in yesterdays episode which is incorrect.)
model.compile("rmsprop", "mse")
output_array = model.predict(input_array)

print(output_array.shape)

# Now let's take what we have here and modify our patch encoder. 

# We will be taking the parameters from the Embedding() from line 713, and adding it the parametes 
#of the Embedding() on line 520.

# Instead of an actual number, we will pass in N_PATCHES as a parameter. See line 520.

# Also, instead of the 768, we will pass in the variable HIDDEN_SIZE.

# Now that we have our N_PATCHES and HIDDEN_SIZE added, next we will modify our self.linear
#projector(patches) by adding (+ self.postional_embedding()), see line 537.

# Also, this will take an input of length number of patches.

# So we would be modifying our positional code, which is in our def call() by adding additional
#embedding_input. see line 536

# Inside this new addition we will have a start which is set to 0.
 
# Then we will have a limit which we will set to our N_PATCHES.

# So basically we will be starting at some zero value and ending with some n value, such that the 
#length of the tensor we're going to create is going to be equal to the number of patches.

# So our limit is equal to N_PATCHES intthat sense.

# In that sense, since we already have number of patches, from here we could add self.N_PATCHES
#to our encoder. see line 527

# Note: We added the self.N_PATCHES to our encoder so that we can successfully add it as a
#parameter and assign it hte variable of limit.

# Also, we will add a delta as a parameter to our embedding_input method, which we will
#set to 1. see line 536

# a (delta =) creates a sequence of numbers that begins at start and extends by increments
#of delta up to but not including limit.

# Next we will add a parameter of embedding_input to our self.positional method. see line 537

# Now we can simply return our output. see line 542

# Note: It should be noted that the embedding layer here is similar to the dense layer, but 
#with a dense layer when we have an input let's say x, the input x gets multiplied by the
#weights, and then we add the bias.

# So this is how we get the output. 

# But with this embedding layer, it is a simple metric small application, so when we take an input
#x we multiply it by the weights and we get the output.

# Now we can run this to set our modifications and move forward.


# Now that we have our patch encoder set we can test it out.

# This is how we will do that.

patch_enc = PatchEncoder(256, 768) # This will get two parameter values, number of pacthes and hidden size.

# From here, we can use this command method to pass in an image and see what we get.

# This command method will also get a shape as a parameter

patch_enc(tf.zeros([1, 256, 256, 3]))

# So now when we pass the input image we expect to get some output of shape batch size by number of
#patches by the hidden size.


# And now that we've built this up to the point where we have our embedded patches, let's go ahead
#and build this transformer encoder.

# First we will start with a layer of normalization multi-head attention, and then we add the input
#to the output.

# Then we have the layer of normalization again, then the multi layer perceptron.

# And again we have an addition, and then we get the output. 

# This is what this looks like visually in a graph.


#          |
#         (+)
#          |
#        [MLP]
#          |
#        [Norm]
#          |
# [Multi Head Attention]
#          |
#        [Norm]
#          |
#   [Embedded Patches]

# Now we can build our code from here.

# To build our code for our encoder, we're going to paste out the patch encoder and modify it
#to use as our Transformer Encoder.

# This transformer is made up of the first norm layer and the second norm layer.

# Next is the multihead attention layer with its two parameters(N_HEADS, HIDDEN_SIZE)

# Now is time for the dense layers.

# We will have two dense layers.

# Now that we've specified the dense layers we can get our output.

# We've built our transformer encoder and now we can pass the input into the different layers.

# We have our input.

# Then input goes into the layer norm 1.

# So we'll have our output (x = layer_norm_1) which takes in the input as a parameter.

# Then our output is input.

# Remember that according to the documentation paper we have the addition ( (+) ) input, so we have to
#specify that in the code directly.

# Now we can return our output.

# This is what that will look like.

class TransformerEncoder(Layer):
    def __init__(self, N_HEADS, HIDDEN_SIZE):
        super(TransformerEncoder, self).__init__(name = "transformer_encoder")

        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()

        self.multi_head_att = MultiHeadAttention(N_HEADS, HIDDEN_SIZE) 

        self.dense_1 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)
        self.dense_2 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)

    def call(self, input):
        x_1 = self.layer_norm_1(input)
        x_1 = self.multi_head_att(x_1, x_1)
#        x_1 = self.mutli_head_att(x_1, x_1) # Error Version

        x_1 = Add()([x_1, input])

        x_2 = self.layer_norm_2(x_1)
        x_2 = self.dense_1(x_2)
        output = self.dense_2(x_2)
        output = Add()([output, x_1])

        return output

trans_enc = TransformerEncoder(8, 768)
trans_enc(tf.zeros([1, 256, 768]))


# Now we have our transformer encoder.

# Now we can test it.

# This is how we will start that process.

#print(trans_enc(tf.zeros([1, 256, 768])))

# Now that we have our Transformer Encoder we can start the process of Building Our ViT(Visual Transformer)

# To start this process we can begin by copying and pasting the ResNet model that we used prviously
#to modify and build our ViTs Model.

# This is the code that we copied to be modified.

# The first thing we will do is change the Resnet Model to a ViTs Model.

# Then we're building in such a way that it takes in the number of heads and the hidden size,
#then from the patch encoder the number of patches.

# From here we would have our patch encoder, which we will define with the Patch Encoder that we
#built already.

# The PatchEncoder will take two parameteres, number of patches, and hidden size.

# Now that we've added the Patch Encoder we can add our Transformer Encoder.

# Recall that we have several Transformers that need to be Encoded, and we need to define them.

# The Transformer Encoder will be made up of a list of different layers, so we need to add
#number of layers as a parameter to our ViT Model.

# The length of this list will be determined by the number of layers.

# We'll use a for loop inside our list to specify the details.

# We'll also have these two parameters (number of heads, hidden size) before the for loop inside our 
#Transformer Encoder.

# The input also goes through the Patch Encoder so we'll pass it as a parameter to our def call function.

# Then it gets passed through the Patch Encoder directly.

# Then we will get the output x

# Once we have the output x what we are going to do is loop through each and every transformer encoder
#layer.

# We will do this by using a for i in range function.

# Note: We must remember to define the number of layers in the visual transformer model.

# Once we set our number of layers we define our for loop.

# Inside the for loop we will have an x, which will be the output for the transformer encoder 
#layers.

# Once we get the output we will proceed to flatten it.

# Once we have Flattened everything out we will define our dense layers, which make up the MLP
#head.

# The Dense() will have a certain amount of units, which will be specified by the number of dense
#units parameter.

# We will add the number of dense units parameter to our Vit Model, along with the activation.

# Then we can copy and paste this to create our dense_2.

# Then we'll have our x = dense_1 which will take x as parameter.

# We will do the samething for dense_2

# Now our final output dense layer has to consider the number of classes.

# So just as what we have been doing so far in this course, we will ensure that our output dense 
#layer has the number of classes number of units in the output.

# Now we can set our return

# Now we can run this.

# Once we run this with no errors we can test it out using the ViT()

# We can print out the vit summary to check out the model.


class ViT(Model):
    def __init__(self, N_HEADS, HIDDEN_SIZE, N_PATCHES, N_LAYERS, N_DENSE_UNITS):
        super(ViT, self).__init__(name = 'vision_transformer')
        self.N_LAYERS = N_LAYERS
        self.patch_encoder = PatchEncoder(N_PATCHES, HIDDEN_SIZE)
        self.trans_encoder = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
        self.dense_1 = Dense(N_DENSE_UNITS, tf.nn.gelu)
        self.dense_2 = Dense(N_DENSE_UNITS, tf.nn.gelu)
        self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation = "softmax")
       
    def call(self, input, training = True):

        x = self.patch_encoder(input)

        for i in range(self.N_LAYERS):
            x = self.trans_encoder[i](x)

        x = Flatten()(x)

        x = self.dense_1(x)
        x = self.dense_2(x)

        return self.dense_3(x)

#vit = ViT(8, 768, 256, 4, 1024) # These are just the values of the parameters we entered previously.
vit = ViT(N_HEADS = 4, HIDDEN_SIZE = 768, N_PATCHES = 256, N_LAYERS = 2, N_DENSE_UNITS = 128)
#vit(tf.zeros([1, 256, 256, 3]))
vit(tf.zeros([32, 256, 256, 3]))

vit.summary()

# Notice that we have 283,478,787 different parameters for our ViT model.

# Now let's reduce this.

# We can say that N_HEADS = 4, HIDDEN_SIZE = 768, N_PATCHES = 256, N_LAYERS = 2, N_DENSE_UNITS = 128.

# Now that we have this model, we'll modify our tf.patches.reshape() by adding a CONFIGURATION
#of the specified Batch Size. See line 536

# The reason why we are doing this is because when we are training, our batch size is going to be
#known, so we want to configure it correctly.

# Now we will run our vit summary again to update our changes.

# Note: Before we run our summary again we must modify our vit(tf.zeros) to reflect that we now have
#32 Batches, instead of 1. see line 995

# Now we can move forwardwith the process of compiling and training our model.

# This is how we will do that.

vit.compile(
    optimizer = Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)

history = vit.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1
)



# Timestamp 25:40:00 

#iTraceback (most recent call last): File "C:\Users\alpha\Deep Learning FCC\building_vits_from_scratch.py", 
#line 1026, in <module> history = vit.fit( ^^^^^^^^ File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\utils\traceback_utils.py", 
#line 70, in error_handler raise e.with_traceback(filtered_tb) from None File "C:\Users\alpha\AppData\Local\Temp\__autograph_generated_filefa37cl1d.py",
#  line 15, in tf__train_function retval_ = ag__.converted_call(ag__.ld(step_function), (ag__.ld(self), ag__.ld(iterator)),
#  None, fscope) ^^^^^ ValueError: in user code: File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\
# engine\training.py", line 1284, in train_function * return step_function(self, iterator) File "C:\Users\alpha\AppData\
# Local\Programs\Python\Python311\Lib\site-packages\keras\engine\training.py", line 1268, in step_function ** 
# outputs = model.distribute_strategy.run(run_step, args=(data,)) 
# File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\engine\training.py", 
# line 1249, in run_step ** outputs = model.train_step(data) 
# File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\engine\training.py",
#  line 1051, in train_step loss = self.compute_loss(x, y, y_pred, sample_weight) 
# File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\engine\training.py",
#  line 1109, in compute_loss return self.compiled_loss( File
#  "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\engine\compile_utils.py",
#  line 265, in __call__ loss_value = loss_obj(y_t, y_p, sample_weight=sw)
#  File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\losses.py", 
# line 142, in __call__ losses = call_fn(y_true, y_pred)
#  File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\losses.py", 
# line 268, in call ** return ag_fn(y_true, y_pred, **self._fn_kwargs) 
# File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\losses.py", 
# line 1984, in categorical_crossentropy
#  return backend.categorical_crossentropy(
#  File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\backend.py",
#  line 5559, in categorical_crossentropy target.shape.assert_is_compatible_with(output.shape) 
# File "C:\Users\alpha\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\backend.py",
#  line 5559, in categorical_crossentropy target.shape.assert_is_compatible_with(output.shape) y 
# target.shape.assert_is_compatible_with(output.shape) target.shape.assert_is_compatible_with(output.shape) 
#ValueError: Shapes (None, 256, 256, 3) and (32, 3) are incompatible    
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          