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


# In this section we will be going over the process of understtanding ViTS

# We will be going over the topic of ViTS, or Visual Transformers.

# We will be using these Transformer Networks to solve problems in computer vision, more specifically,
#in the task of image classification.

# Up to this point we've seen different convolutional neural networks like the Lenet Model, the
#VGG Model, the Resnet Model, the Mobile Model, the Efficient, and the Net Model.

# Now we will be looking at the Vision Transformers.

# The Vision Transformers were first developed and introduced in a paper titled 
#"An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale."

# In this section we will be looking at the Transformer, how has it been constructed and 
#how it works, also how and why Transformers perform as well as their Convolutional
#Neural Network counterparts.

# The very first point we want to note here is the useage of the Transformers for Computer Vision
#tasks has been developed in very recent times.

# We could see that from the date of the published paper the Authors say that while the 
#Transformer architecture has become the defacto standard for natural language processing
#task, its application to computer vision remains limited.

# In vision, attention is either applied in conjunction with convolutional networks, or used
#to replace certain components of convolutional networks while keeping their over structure in place.

# We show that reliance on the convolutional neural network is not necessary and a pure 
#transformer(without CNN's) applied directly to sequences of image patches can perform very well on 
#image classification tasks.

# When trained on large amounts of data and transferred to multiple mid-sized or small image
#recognition benchmarks(ImageNet, CIFAR-100, VTAB, etc.)

# Vision Transformer (ViT) attains excellent results compared to state of the art convolutional networks
#while requiring substantially fewer computational resources to train.

# Next we will take some time to explain a few terms that have specific meanings in Program Training.

# To better understand the Transformer and the role it has to play in the ViTs architecture, we 
#would have to go back in time to understand why it was first developed.

# In 2017, a paper entitled "Attention Is All You Need", was first developed by Vaswaniya Al 
#and it has turned out to be one of the most influential papers in the mordern deep learning era
#with the developement of the Transformer Architecture we have today.

# At the heart of the Transformer Architecture we have the Self Attention model.

# More specifically, in the paper they used the scaled up product attention that we would see,
#but then as we said, the whole purpose of the domain in which these kinds of architectures or 
#those kinds of networks were built was for natural language processing.

# The question then becomes how does this work in natural language processing?

# To understand how and why the attention and also the Transformers are used in Natural Language
#processing we'll take the following example, which is that of translation, which we are used to 
#already doing in google translate.

# For this example we will be translating text from English to French.

#           English                                                       French
# [The weather today is amazing.]                           [Le temps aujourd'hui est incroyable]

# Initially the kinds of deep learning techniques which were used in solving these kinds of problems, 
#that is taking us from one language to another, were the recurrent neural networks.
 
# The way the recurrent neural network works is quite simple.

# We'll start by putting the text in the text box.

# Let's take a look at our example sentences again.

#  []    []      []  []    []                                []  []      []       []      []
# [The weather today is amazing.]                           [Le temps aujourd'hui est incroyable]

# Notice that we've added some extra blocks to our example sentences.

# These blocks are Recurrent Neural Network blocks.

# RNN's were one of the first learning based models used in natural language processing tasks like
#the case with this translation.

# The way this works is we have our initial text, or we have our source, that's the English text, then
#we have the target, which we to generate, which would be the French.

# So initially we have the input and the output which we are going to train on, and later on when we
#pass in some random input we expect to get a reasonable output.

# The way that our input is structured is that each word is called a token. 

# For example, we have 5 tokens in our input sentence.

# Each token gets converted into a vector, and then gets passed to the RNN block.

# We can carry out some simple computations like multiplication and addition, and that information
#will be passed from one block to another, hence the term recurrent neural network.

# The importance of passing the computations from one block to another is that each token's computations
#will depend on all the previous tokens.

# Once we are done passing information from one block to the next, we are going to take the last block,
#which should hold all of the computations from the previous blocks and pass it to the next section
#of RNN blocks.

# This is our Encoder Block (The RNN where the computations are coming from, also where our source text is).

# The next section of RNN blocks will be our decoder (The RNN where the computations are going, also
#where the source is translated).

# And again, a similar process is repeated when we are in the decoder.

# We will have computations, which will produce an output, which will get passed to the next block,
#and so on so forth up until the final output.

# But then, the problem with this technique is that imagine we have a very long text, then we might happen
#that it becomes too difficult for information to flow from block to block, which is important for
#previous context to be passed to neighboring blocks.

# Without the context from previous blocks, this problem will lead to poor results.

# Another problem is that each time we train we have to pass the information from one block to
#another sequentially.

# Because the information gets passed sequentially, it makes it difficult for us to implement 
#parallelization very efficiently.

# So This makes the training of these kinds of neural networks very difficult.

# To tackle the issue with long term dependencies, attention networks were developed.

# So instead of depending on just the final output we get from thhe hidden layer, which gets
#passed to relay information from the source to the target language, what we'll do is for each
#and every unit we have, each and every recurrent neural network block we have, we are going 
#to take into consideration inputs from each and every block.

# For example, inputs from each and every block of our encoder will be passed to the first 
#block of the decoder.

# We then have an "Attention" layer which processes the inputs from the different source RNN blocks 
#and produces an output vector which is now passed as input into the target RNN block.

# And now we have the source for the first block of the target RNN.

# Now, instead of having just the computations from the first block of the target go into the second block,
#what we will do is have another "Attentiion" layer.

# So again, inputs from each and every source block will get passed into an "Attention", which will then 
#get fed into the second block of the target RNN, along with all the computations from the first block.

# After doing this we would see that each and every block in our target RNN pays attention to each and 
#every block from the source RNN.

# When we read through the paper entitled, "Neural Machine Translation by jointly learning to align
#and translate".

# That is a famous Badenau et al paper you can see some of the attention maps.

# In these attention maps we can observe and clearly see which words align most with one another.

# At this point we are going to move on from the attention and have a look at self attention.

# To better explain the self attention we'll consider a whole different problem, which is that of 
#sentiment analysis.

# So we want to have the model take in an input, and classify the sentiment of the input.

# We want to know if the statement about the weather is a positive or negative one.

# So we have the model which takes in inputs like this.
                               
# [The weather today is amazing.]

# And then we have this model.

#   /|\ 0,1
#    |___________________________
#    |                           |
#    |                           |
#    |                           |
#    |                           |
#    |___________________________|

# We want this model to output, or tell us whether the staement is positive or negative.

# Notice, for the self attention we are not going to need the RNN blocks.

# What we're passing from the source input to the model are vectors.

# We will have a vector for each token/word.

# These vectors will have a sequence length.

# The sequence length of these vectors is 5.

# So we have a sequence length by let's say, embedding dimensions (S, E) metrics,
#which we get from tge model.

# Let's explain.

# Let's suppose that the sequence is 5 as we've said, and then the embeddings dimension is 3.

# We would have these metrics (5, 3)

# These are the metrics that will get passed into the self attention layer.

# All these vectors which we pass into the self attention unit are going to be designed in a way
#that words which look alike are going to be close to each other while words that are opposites 
#are going to be far away from each other.

# Now, since we're working in 3 dimensions it means we'll have 3 values for each vector.

# This is a visual representation of our 3 dimensions.

# First we'll look at the word happy, which would get plotted close to a word like smile.

# This would be the complete opposite of the word, "sad", for example.

# "sad" might get plotted close to a word like angry.

# The next set of words that we may see appear near eaach other are "the" and "is".




#       |         *happy
#       |         *smile
#       |
#       |
#       |
#       |________________________
#      /                    *the
#     /                     *is
#    /     
#   /      
#  /  *sad       
# /   *angry      


# Now let's get back to our model where we have our 5 by 3 (5,3) input which is passed into the 
#self-attention layer.

# So we could for example, have a metric set that is 5 by 3 (5, 3), and each word will have its own
#embeddings, So we would have some values there.

# Now suppose that we're working in a three dimensional embedding.

# We would see that each and every word has its own embedding.

# That would look something like this.

# [0 0 0] - The
# [0 0 0] - Weather 
# [0 0 0] - Today 
# [0 0 0] - Is
# [0 0 0] - Amazing

# So these are the different words we have and at this point we'll implement a special type of
#attention called a "dot product attention".

# This is where we'll take the 5 by 3 (5, 3) metric set and multiply it by the transpose of a matrix 
#which has the shape of of our (5, 3) metric set matrix.

# So we'll take our (5, 3) metric set matrix, which we'll call the query, and multiply it by 
#the transpose of the key.

# The key is going to be (3, 5), since it is going to have the same shape as the query.

# This is the shape of our key visually.

#   Key

# [0 0 0 0 0] 
# [0 0 0 0 0] 
# [0 0 0 0 0] 

# So now we have our query, and we have a (3, 5) matrix.

# This is a visual representation of what we will be multiplying.

#   Query                               Key

# [0 0 0] - The                     # [0 0 0 0 0] 
# [0 0 0] - Weather 
# [0 0 0] - Today                   # [0 0 0 0 0] 
# [0 0 0] - Is
# [0 0 0] - Amazing                 # [0 0 0 0 0] 

# The product of these will give us a (5, 5) matrix.

# After getting this (5, 5) matrix, we could pass it through a softmax layer.

# Now we've looked at the softmax layer in previous sessions, but one thing we should
#note here is that once we have the (5, 5) matrix, it produces an attention map similar
#to what we have seen before where we have the source input vertical and horizontal.

# That would look something like this.

#       The  weather  today  is  amazing
# The

# weather

# today 

# is 

# amazing

# With this formula, words which are most similar to each other in certain context are going 
#to have the highest values.

# From here we have this (5, 5) matrix, which now when multiplied by our (5, 3) matrix will give
#us a (5, 3) matrix.

# Generally we call the matrix which gets multiplied by the attention matrix the value.

# So we have The Query, The Key, and The Value.

# From this we can see that we would have an input which is (5, 3), and now we would also
#see that we have an output of (5, 3).

# This (5, 3) will then get passed through some fully connected layers, and then we'll have
#an output, or a fully connected layer with one neuron in the output which will tell us
#whether an input statement is a positive statement or a negative statement. 

# And so now we've seen that we've completely gotten rid of the ruccurent neural network blocks.

# As of now we're just making use of the self-attention blocks to extract information from our 
#inputs.

# Now one of the first papers if not the first paper which made use of just the attention, and 
#getting rid of the RNN's was the "Attention Is All You Need" paper and it happens to be one
#of the most influential papers in mordern day deep learning.

# With understanding how the transformer encoder works, let's now get into this unit where we break
#an image into different patches.

# Let's look at the visual example of this below.

#   Viision Transformer (ViT)

#  _______
# |_CLASS_|
# | Bird  |                   ______
# | Ball  | <--------------- | MLP  |
# | Car   |                  |_Head_|
# | ...   |                     |
# |_______|                     |
#                               |
#                       ________|____________________________________________________________
#                      |_______________________Transformer_Encoder___________________________|
#                          |     |     |     |     |     |     |     |     |     |
# Patch + Position ---> [0,#] [1, ] [2, ] [3, ] [4, ] [5, ] [6, ] [7, ] [8, ] [9, ]     
#   Embedding              n     |     |     |     |     |     |     |     |     |
#       |                     [___Linear___Projection___of___Flattened___Patches___]
# *extra learnable               |     |     |     |     |     |     |     |     |
# [class] embedding              |     |     |     |     |     |     |     |     |
#       |                        |     |     |     |     |     |     |     |     |
# [A][B][C]
# [D][E][F]-------------------->[A]   [B]   [C]   [D]   [E]   [F]   [G]   [H]   [I]   
# [G][H][I]

# Model Overview: We split an image into fixed-size patches, linearly embed each of them,
#add position embeddings, and feed the resulting sequence of vectors to a standard Transformer
#encoder. In order to perform classification, we use the standard approach of adding an extra
#learnable "classification token" to the sequence.

# To better understand how and why we make use of patches, let's not forget that what the 
#Transformer encoder takes in is some input sequence.

# Our initial input sequence were words that could be represented by embedding vectors.

# Then our words combined with embed vectors gets passed into the Transformer.

# Since our input is the image, in order for us to represent it in patches, we'll have to break it up.

# So what we could do is if we look at first sight, we have this image...

# [A][B][C]
# [D][E][F]  
# [G][H][I]

# Let's suppose the image has a shape of 256 by 256, by say, 3 channels (256, 256, 3).

# Then we could take each and every pixel here, we'll exclude the 3 channels for now, so we'll
#focus on the 256 by 256 pixels.

# For each pixel in the 256 by 256 image we would have a vector representing that pixel.

# So each pixel would have its own vector representation.

# But we shouldn't forget that unlike previously where we had only five words now we have 256 
#times 256 words, because if we an image like the one from our example, and we have to get each
#and every pixel, then we'll have 256 by 256, which is more than 65,000 diffrent vectors which
#we'll have to pass to our Transformer.

# And so before in our attention model where we had an attention map which was 5 by 5, recall
#that we saw already that when we had the input sentence with 5 words we had a 5 by 5 vector
#attention map, now we have a 65,000 by 65,000 attention map.

# We see that working with these kinds of matrices and memory isn't very feasible, so instead
#of going pixel to pixel, The Authors decided to go patch by patch.

# The authors chose to work with 16 by 16 patches. So each patch has a shape of 16 by 16.

# And so given that we have 16 by 16, if we have any particular patch, we would have 256 pixels
#in that patch.

# Basically what we are saying is that each patch has 256 pixels in it.

# So unlike with the words where we had a 5 by 3, (five words that are each represented by a 3
#dimensional vector), each patch is represented by the 256 dimensional vector.

# When we look at the computer vision, we see that we have a 256 by 256.

# Now when working with the Transformer, we may not want to work with the 256 dimensional vectors,
#maybe we want to work with let's say, 512 dimensional vectors, for example.

# In that case we would have to do linear projection of the flattened patches, such that we leave 
#from 9 by 256 and go to 9 by 512. (9 is the length of our sequential line, and 256 is the number of
#pixels in each patch, initially, before we went through linear projection and moved to a 512 
#dimensional vector.)

# We now get to 9 by 512.

# 512 will be the embedding dimension for our Transformer.

# Remember that prwviously we had an embedding dimension of 3.

# So linear projection allows us to work flexibly as we can now decide on what size we want for
#one embedding dimension.

# With that being said, we now have an output of 9 by 512, and we're ready to pass it to our Transformer
#encoder.

# But just before passing this, we would add the position embeddings.

# Now the reason why we even have to do this is because unlike with the conv nets where the way the 
#convolutional neural networks work is that for computing the feature maps it takes into consideration
#locality.

# So this means that we see these two portions, the patch and the position, when passed with a conv
#filter will produce a certain output, and so this means that pixels that belong to a small locality
#will be used to produce the output, and this clearly gives the CNN's an upper hand over the Transformers
#as when trying to understand an image.

# The positions of particular pixels actually matter, so this means that CNN's already have an 
#inductive bias due to the way they actually work.

# And so to give a helping hand to the Transformer network we will need the postion embedding which 
#gives the Transformer encoder an idea of the location of each and every patch which is passed in.

# But again it should be noted that this will have to be learned automatically by the model.

# If we notice, we have an extra input in patch positions. 

# The [0,#] is the extra input that doesn't come from the Linear Projection of Flattened Patches.

# The reason we have this extra input is simply because we do not want the situation where after
#going through the encoder we pick one of the outputs that came from the Linear Projection of Flattened 
#Patches to be used as the MLP head, or to be used for the fully connected network in the classification 
#unit ([1, ] [2, ] [3, ] [4, ] [5, ] [6, ] [7, ] [8, ] [9, ]).

# So to avoid this sort of bias where we would be picking one of the classifiers, the authors add the
#extra learnable class embedding ([0,#]), whose output will be passed into the MLP head and then 
#will be used for classification.

# Another important note to remember here is that the Transformer Encoder or Visual Transformers, are some
#sort of a hybrid architecture because we may choose not to pass in those image patches directly but instead
#pass those image patches through the convolutional neural network, then get the output embeddings and pass 
#it in directly.

# Instead of the image patches it should be noted that the multi-layer perceptron contains two fully
#connected layers with a GELU non-linearity.

# The type of normalization is the layer normalization as we have mentioned already, and when talking
#about the layer normalization, if we consider some inputs where we have the sequence length, or the
#sequence dimension, then we have the features or the embeddings, or like a vector actually.

# So we would have the different vectors, and then we would have the batch dimension.

# So basically what we're saying is we have a sequence length or we have the different sequence 
#vectors which have been passed into some layer and then instead of doing or carrying out 
#normalization throughout the batches as in the case of the batch norm, we're going to 
#carry out the normalization for each and every vector. 

# And that's the reason why we do not use the batch norm with the Transformers, because of the fact
#that the batch statistics for NLP data has a very large variance throughout training. 

# This variance exists in the corresponding ingredients as well.

# And so to avoid this kind of situation, it's perferable for us to carry out the normalization on the
#features instead.

# Before we move on to some experiments let's look at how the ViTS are being used in the real world.

# So actually, the ViTs are pre-trained on very large data sets and fine-tuned to smaller downstream tasks.

# Obviuosly when fine-tuning we remove the pre-trained head and replace it with a head which corresponds to
#our number of classes.

# So this means that initially we may have a thousand class heads, and then we move to k classes, or let's
#say 3 class heads.

# To better understand why we have a d by k output, let's get back to our Vision Transformer.

# So after the inputs have been passed in, we have an output sequence length, plus one.

# Remember the plus one is going to be the extra input we mentioned previuously.

# So let'sjust say that we have a one by d output.

# If we're considering all, the sequence length will be a sequence length by d output.

# The d is our embedding dimension which we had fixed from the linear projection.

# So we have the 1 by D, and then we pass it through to the MLP head.

# Obviously it becomes simply d neurons.

# So now we have d neurons, since it's just one by d, and then we have an output.

# Let's say we have a thousand classes, then we'll have the fully connected layer which brings all
#the d inputs to the k outputs, or in this case to the 1000 outputs.

# Now where we want to fine tune we're going to take off the 1000 outputs and replace it with k outputs.

# Now we have k outputs, then we initialize the weights of the fully connected layer.

# The Authors also make mention of the fact that during fine tuning it is better to work at higher
#resolutions.

# This means that the model could be trained at 256 by 256, and then later on fine tuned with 
#512 by 512 images.

# And since they keep the patch size the same, this results in a larger effective sequence length.

# Now let's visualize this statement.

# So we have this input which we'll say is 48 by 48. and when we divide this or break it up into

# [A][B][C]
# [D][E][F]  
# [G][H][I]

#  And when we divide this or break it up into three parts we have 16 16 16 by 16 16 16.

# So we have 16 by 16 patches.

# Now if we want to fine-tune on the hiher resolution image, then let's say the higher resolution image
#is let's say 96 by 96, so we could have something like this.

# [] [] [] [] [] []
# [] [] [] [] [] []
# [] [A][B][C][] []
# [] [D][E][F][] []  
# [] [G][H][I][] []
# [] [] [] [] [] []

# If we now fine-tune on the 96 by 96 image, and we still maintain the fact that the patches will be 
#16 by 16, then this means that instead of having 3 patches, we're going to have 6 now.

# So now we have 6 patches across and down, which will give us a total of 36 different patches instead
#of 9 patches as we had previously.

# That's why we are reminded that the sequence length will increase, and that's so long as they can fit
#the meomry.

# Now due to this modification the pre-trained position embeddings may no longer have meaning, so they 
#therefore perform 2d interpolation of the pre-trained position embeddings according to the allocation
#in the original image.

# In the experiments we could see the different models. 

# These are the ViT-Base, ViT-Large, and the ViT-Huge.

# This is what the entire experiment table chart looks like.

#   Model       Layers      Hidden Size D       MLP Size        Heads       Params
#-------------------------------------------------------------------------------------
#   ViT-Base      12            768              3072            12          86M
#   ViT-Large     24            1024             4096            16          307M
#   ViT-Huge      32            1280             5120            16          632M

# This table Details the Vision Transformer Variants.

# Hidden Size = embeddings
# MLP Size = Fully Connected Layers
# Heads = Attention Heads
# Params = Parameters

# The experiments were carried out on a JFT 300 median data set.

# From the experiment graph below, we can see that the 14 by 14 version of the ViT outperorms 
#the ResNet 152.

#-----------------------------------------------------------------------------------------------------
#                        Ours-JFT        Ours-JFT        Ours-JFT        BiT-L       Noisy Student
#                       (ViT-H/14)      (ViT-L/16)      (ViT-L/16)   (ResNet152x4) (EfficientNet-L2) 
#-----------------------------------------------------------------------------------------------------
# ImageNet                88.55           87.76           85.30         87.54          88.4/88.5*
# ImageNetReal            90.72           90.54           88.62         90.54          90.55
# CIFAR-10                99.50           99.42           99.15         99.37            -
# CIFAR-100               94.55           93.90           93.25         93.51            -
# Oxford IIIT Pets        97.56           97.32           94.67         96.62            -
# Oxford Flowers-102      99.68           99.74           99.61         99.63            -
# VTAB(19 Tasks)          77.63           76.28           72.72         76.29            -
#-----------------------------------------------------------------------------------------------------
# TPUv3-core-days         2.5k/           0.68k           0.23k          9.9k           12.3k

# Now this performance, although not largely greater than that of the ResNets requires less computation
#resources to train as we see here in the experiment graph, we have 2500 TPU core days required
#to train the model as compared to the ResNet which requirs 9900 TPU core days. 

# Also from some example demostration plots we can see that when we increase the number of pre-training
#samples, the model which performs the best is the ViT, which outperforms the ResNet, whereas for 
#a reduced number of samples RestNets outperforms the ViT.

# In another example plot, the smaller the patch size we have, the better the results. 

# Now to better understand the reason why as we increase the data set size the ViTs start to outperform
#the Convnets.

# We have to recall that when working with convnets like the ResNet, there is some inductive bias in the
#sense that the fact the ResNet takes as input a two dimensional image already gives the convnet a 
#helping hand when it comes to extrating features.

# And so even with relatively smaller data sets, these convnets can make sense out of the input image.

# Now with the Transformers which are some sort of generic neural network, the model doesn't get to
#see the image in its natural form.

# What it sees is some patches which have been converted to some vectors.

# And so at the very beginning or with small data, the Transformer model finds it difficult to make 
#much sense out of the patches, but as soon as we increase the data set to considerable amounts,
#the Transformer model now free of the inductive bias can even do better than the confidence, and 
#interesting enough we'll notice that after training a Transformer model, the position embeddings
#we call which are added onto the patch embeddings before passing it to the Transformer actually
#learn on their own to encode the position of the patches. 

# We can also visualize what the model sees by looking at the Attention maps.

# We will notice that after training the model, we see we have the Attention which are pixels, which
#pay much more attention to one another, as compared to the other pixels, which would be the pixels 
#from the input.

# In summary, to understand or to visualize what goes on when training the CNN and ViT model
#side by side, we'll see that the ViTs, as we increase the data size, or rather when we start
#with small data sets, we have less accuracy than the CNN model.

# While for the CNNs, we already have reasonable accuracies even with small data sizes.

# Then as we keep increasing the data size, we see the accuracy of the ViT model increasing, 
#while for the CNNs, at some point plateauing starts to happen.

# This plateauing simply comes from the fact that the CNNs are limited by the inductive biases,
#whereas these Transformers which are more generic neural networks are free to learn even from
#the larger data sets.  

# Timestamp 24:44:00