import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import Accuracy, RootMeanSquaredError
from keras.optimizers import Adam


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


func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image") 

x = Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")(func_input)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides= 2, padding = "valid")(x)

x = Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides= 2, padding = "valid")(x)


x = Flatten()(x)

x = Dense(1000, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)

func_output = Dense(1, activation = "sigmoid")(x)

lenet_model_func = Model(func_input, func_output, name = "Lenet_Model")

lenet_model_func.summary()


# Here we will go over the process of creating Custom Layers.

# We could recall from the previous sections, the way the dense layer is built.

# It is built in such a way that this as our dense layer, and then we have an input, which we will call x, then we 
#we have a certain M times X plus C.

# So our M is actually the weights.

# So the weight times x plus the bias.

# Let's call that B and then this is will now equal our output.

# So notice that we have an output of Y, Y equals MX + C

# So the symptom means if we want to recreate the dense layer, then we have to take this into consideration, or the 
#definition of our layer from scratch into consideration.


#          _____________________
#         |                     |
#         |                     |
#         |                     |
#  x ---> |        MX+C --------|---> Y = MX+ C
#         |                     |
#         |        WX+B         |
#         |_____________________|
#                    D

# With that said, we could define a neural learn dense.

# So it's like our custom dense, neural and dense.

# It's going to be a class

# It's going to inherit from layer.

# And from here we will define our init method.

# Then we have our super function, which take as parameters the NeuralearnDense and self.

# Then we have our .init that gets attached.

# Now from here we may have noticed, whenever we're creating a dense layer, we generally had to specify at least 
#the activation.

# We also need to specify the number of output units first.

# With that said, we have to take that into consideration when building our neural and dense layer.

# So inside of our def __init__(), we will pass in output_units as a parameter. 

# Then we will define self dot output_units, which are equal to output_units

# From this point on, we will be building our layer.

# In order to build this layer, we have to take into consideration this definition. See line 125 - 133

# But this definition isn't very clear like this.

# We'll break it up to mkae it easier to understand.

# So suppose we have this input of shape batch size of let's say, number of features. (B,F)

# So suppose that is our input.

# What happens here is this input is going to be multiplied by these weights. (WX+B), which happens to be a matrix.

# Now we take this, (B,F), and multiple it by that matrix.

# And for the multiplication to be valid, we need to ensure that the number of columns we have matches the number of
#rows in the matrix.

# But then what are the dimensions of this matrix?

# We have to note that this matrix has to be defined insuch a way that we have a shape of (F).

# This shape of (F) must match the number of output units.

# So if we want the number of output units, for example to be 1, we specify it like this (F,1).

# And that's why when defining the dense layer, we don't need to specify the (F) in (F,1), becuase (F) is gotten
#automatically from the number of columns in the inputs (B,F). 

# If we don't take (F) from the number of columns in the input, we will get an error.

# TensorFlow automatically takes (F) from (B,F) and then collects the input we pass into the dense layer.

# So when we specify the dense, by passing a number of outputs (1 as we specified for the example), the weights matrix
#(W) is automatically defined by the input we pass into the dense as the number of outputs.

# So basically, the dense affects the rows we have.

# But then what we pass in as an argumnet here is going to give us the number of columns we're going to have for the
#matrix.

# With that said, we have F by 1 (F,1), and we're going to multiple that, and then we'll have B by 1 (B,1).

# So now we understand how we get this output.

# And then we have plus B by 1.

# (B,1) + (B,1) 

# Our fisrt (B,1) comes from multiplying (B,1) by (F,1)

# Our second (B,1) comes from the bias.

# So that's it. 

# Now once we add this up we have output of B by 1 (B,1), and that's our Y. 

# That's the shape of our Y.

# Now we can move on to building.

# We'll start by defining the build method, which just gets a parameter of self for now.

# Next we will define our weights with self.Weights and define our weights.

# We can do that by making self.Weights equal to the self dot add weights method.

# This (add_weight) actually comes with the layer class.

# So we're able to call this (add_weight) because we're inheriting from the layer class.

# Next we specify the shape

# We are going to have our numer of rows, so n rows, which is going to come from the inputs.

# And then we are going to have our number of columns, which is going to come from the output units we specified.

# To do that we are going to pass in self.output_units

# So we get a number of output units.

# How do we obtain this number of inputs units?

# For now we have the weights.  

# Next we will add the bias by using self.biases, and this is also a weight.

# We Will specify the number of output_units without n_rows, since this one dimensional, it will only have one 
#parameter.

# Now we will define our weights and bias matrix.

# Next we will build our call method, which will have two parameters, (self, input_features)

# Now that we have that, what we're gonna do here is simply return the matrix modification (matmul), as we've seen 
#already with the weights, we wil return the tf.matmul() with the parameters input_features, and self.weights.

# The reason that the input_features comes before the self.weights here is because it's based off the equation of
#input (B,F) times the weight (F,1). If we tried to run the weights (F,1) times the input (B,F), we would det an 
#error.

# And then we add up the biases as parameter using self.biases

# Now we can go back to n_rows. 

# The way we're going to get this number of rows is going to be easy.

# Inside our def build(), all we need to do is specify that we have the input_features_shape, which is going to
#automatically come from the def call() parameter input_features.

# And then to get the number of rows, all we need to do is add input_features.shape as a parameter to self.weights =
#self.add_weight() by replacing the n_rows parameter.

# And then we get that last dimension from our definition (see lines 125 - 133) 

# Recall from our definition that we have B by F plus B by F, (B,f) + (B,F).

# And so here we need the weights, it needs to be F by output (F,O).

# So to get the F from (F,O), we just need to take the input (B,F), and then get the last element of that input (F),
#and there is the number of columns we have.

# That is how we get the input_features_shape.

# We specify the index where we took it from [-1] (The last element from the input (B,F), which is (F), our number 
#of columns)

# So that's how we obtain this input_features_shape value automatically.

# The next thing to do is to specify that it's trainable.

# So we have to specify that the weights are trainable, because in some cases we may want that the weights 
#shouldn't be trainable.

# We will specify the weights to be trained by passing trainable = True as a parameter into the self.weights().

# Then we will add the samething to the self.bias()

# Now apart from that, we could randomly initialize our weights and biases.

# To do that we will add initializer = "random_normal" to both the self.weights and self.bias, before the trainable 
#parameter.

# Now we will run this and get our neural learn dense layer.

# After we run this we will integrate it.

# We can do that simply by using our sequential API.

# Basically we'll copy and paste the lenet_model sequential API and integrate it into our neura learn dense layer.

# Next, instead of dense layer, we will have neura learn dense layer.

# So we see that we can create our own custom dense layers with TensorFlow.

# Also we will use a Try and Except to catch a possible error with the activation. This is because we don't take 
#into consideration the activation.

# And so what we could is, we could go back into our call method and modify things.

# We'll specify if that activation equals relu, return tf.matmul(input_features, self.weights, self.biases)

# Then we'll specify elif, activation is sigmoid we'll return tf.matmul(input_features, self.weights, self.biases)
#still, but modified.

# Then we'll do the same thing for the Else, return tf.matmul(input_features, self.weights, self.biases)

# Now we can start our modifications.

# Firts, for the "relu" we'll have tf.nn(tf.matmul(input_features, self.weights) + self.biases)

# And if the activation is "sigmoid", we'll have tf.math.sigmoid(tf.matmul(input_features, self.weights) + self.biases)

# The Else retrun stays the same tf.matmul(input_features, self.weights) + self.biases

# So we have successfully modified our code in such a way that we have integrated our activations.

# Note: our activation will be ==, which means equals, and not =, which means assigned.

# Also note that we must now specify activation as a parameter in our def __init__()

# Then add self.activation = activation

# Next we will update our if and elif to self.activation

# If we ran the code like this we will get an error. So to fix this we will include shape = () to both our self.weights
#self.biases.

# We're still not done yet. If we try to run the code like this will get another error.

# To fix the problem this time we will need to change our variables.

# So instead of self.weights, we're going to self.w.

# And instead of self.biases, we're going to have self.b.

# We will also need to change the variables inside of our return functions parameters to reflect the change.

# But instead of having to write out the code a bunch of times, we will create a new variable inside of our 
#call() and call it pre_output.

# This pre_output will be assigned to tf.matmul(input_features, self.w) + self.b.

# From here we can set the return tf.nn.relu, the return tf.math.sigmoid, and the return to pre_output.

# Now running this should be fine and we can move on.

# After running we can see that we have successfully created our model.

# The same exact model we've been building from the start.

# But this time around we're using a custom Dense layer, which is our neural dense layer.

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
    
IM_SIZE = 224
lenet_custom_model = tf.keras.Sequential([
    InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
    Conv2D(filters = 6, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    Conv2D(filters = 16, kernel_size = 3, strides= 1, padding = "valid", activation = "relu"),
    BatchNormalization(),
    MaxPool2D(pool_size=2, strides= 2, padding = "valid"),

    Flatten(),

    NeuralearnDense(1000, activation = "relu"),
    BatchNormalization(),

    NeuralearnDense(100, activation = "relu"),
    BatchNormalization(),

    NeuralearnDense(1, activation = "sigmoid"),
])

lenet_custom_model.summary() 

# Now we'll go ahead and compile this model and then train it.


lenet_custom_model.compile(optimizer = Adam(learning_rate=0.01),
              loss = BinaryCrossentropy(),
              metrics = "accuracy")

history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)

# We notice that this training process looks quite similar to the training sessions we have done previously, but 
#this time we are using our custom layer.