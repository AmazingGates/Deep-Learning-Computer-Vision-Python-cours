import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer, Dropout, Rescaling, Resizing, RandomFlip, RandomContrast, RandomRotation
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


# In this section we will be going over the concepts and the understanding of data augmentation.

# We are going to see the effect of augmenting our data artificially without actually getting to add an element 
#in our data set, and then seeing how this affects our model performance.

# In the session we will see how to augment our data and see how this technique of data augmentation helps
#in making the model even more performant.

# Data augmentation promotes diversity in our data set.

# Data augmention helps us fight overfitting by providing different views of the data points.

# We will be using keras.layers to augment our data.

# Now we will implement this and start our code.

# We will initialize this above our training data set on line 86.

### tf.keras.layer augment

# If we read the documentation in the Tensorflow  we would see that there are many different methods we could use
# #to augment our data.

# For now, we will only use the three we have listed above.

# Generally, we would have to test each method separately and see which one's work well with our model.

# Now that we have this  we're going to include the mapping.

# We are going to add the mapping to the training dataset and pass a parameter of augment_layers, and a parameter of
# num_parallel_calls = tf.data.AUTOTUNE. See line 98

# Note: We will only do this for the train dataset, not the validation dataset.

# Now that is done we will define our augment_layer, which will return augments_layers with image, training = True
#as parameters. It will also get a label at the end of the code. This code will be intialized on line 93 so that
#we can have access to all of the variables that come after. see line 98 for example.

# Now that we have implemented augmentation into our code, we may have noticed that the performance of our model
#has dropped.

# To understand why, we can look at the visualization of our data set. which will show us that after carrying out the
#rotation, we have images that are rotated at very unusual angles, compared to the standard position of the images
#we have in our validation training set.

# So we have to ensure that when carrying out the random rotation operation we limit the angle at which we could 
#carry out this rotation.

# So with that said, we have to specify that we only want image rotations that tilt to one side or the other, slightly,
#not completely upside down or completely turned to one side or the other.

# We can specify the degree of the tilt by adjusting the value in the random rotation method.

# The documenation tells us that the value in the random rotation gets multipled by 2pi. (pi = 180degrees). So 
#basically we are multiplying our value by 360. So for example if we take the default number that we have in our Random
#Rotation (25) and multiply it by 2pi, we would get an equation like this, .25 * 2pi or .25 * 360. In this case 
#both answers are 90. Which indicate a tilt of 90 degrees.

# So what we are going to do is limit the rotation to avoid over tilting.

# For example, we will change our value in the random rotation from 0.25 to -0.025. see line 88

# Notice that we have -0.025 to 0.025, which simply means that our image can only randomly tilt between -90 and 90
#degrees in either direction.

# It's either direction because if it tilts towards negative, it's tilting towards left, if it tilts towards positive,
#it's tilting towards right.

# Now that we have this we can go ahead and retrain our data to see what we get.

# With that being done, we are now going to use another data augmentation strategy called cut mix.

# Cut mix isn't like the other augmentation stragety where we just modify a single image, with cut mix, we actually 
#combine images.

# First we will import tensorflow_probability as tfp

# We will apply the cut mix code under the def augment_layer. 
 
# We started the mix cut code with a def box(lamda) see line 96.

# Once we aree done with the box(lamda) we started our def cutmix, see line 

# After we applied the code, we created train_dataset_1 and train_dataset_2. which we then combind inside a variable
#called mixed_dataset. see line 121 and 122 for train_datasets and see line 124 for mixed_dataset.

# We then created a new variable called training_dataset where we passed in the mixed_dataset as a parameter. see line
#127

# Now when we run our code we should see that our model is performing without overfitting.
