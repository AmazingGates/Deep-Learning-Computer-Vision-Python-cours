import tensorflow as tf # For our Models
import pandas as pd # For reading and processing Data
import seaborn as sns # For Visualization
import keras.layers 
from keras.layers import Normalization, Dense, InputLayer
import matplotlib.pyplot as plt
import keras.losses
from keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import numpy as np


# Here we will build off of our Task Understanding to start our Data Preparation

# Before diving into our data, let's take a look at the big picture. 

# Here we have a model which is fed from an input and an output.
#   X ---> MODEL <--- Y

# Later we can have an input fed to obtain an output
#   X ---> MODEL ---> Y

# So initially we have inputs that we feed into our model (X,Y)
# After the model is trained, we want to feed an input to obtain an output (X -> Y)



#    X(hp)  |   Y(prices K$)
#   109     |   8
#   100     |   9.3
#   113     |   7.5
#   97      |   8.87
#   206     |   15.16
#   124     |   8.31
#   162     |   12.8
#   150     |   11.50
#   80      |
#   156     |

# Now let's get into preparing our data so that it can be fed into our model.

data = pd.read_csv("training for ML Model.csv") # This method is used to read csv files
print(data.head()) # Output
# v.id  on road old  on road now  years      km  rating  condition  economy  top speed  hp  torque  current price
#0     1       535651       798186      3   78945       1          2       14        177  73     123       351318.0
#1     2       591911       861056      6  117220       5          9        9        148  74      95       285001.5
#2     3       686990       770762      2  132538       2          8       15        181  53      97       215386.0
#3     4       573999       722381      4  101065       4          3       11        197  54     116       244295.5
#4     5       691388       811335      6   61559       3          9       12        160  53     105       531114.5

# Notice when we run print(data.head()), we get back the first 5 rows of data from our csv.

# We can also do things like get the shape of our data output
print(data.shape) # Output (1000, 12)
# Notice that we got back a 2-d shape, with 1000 rows, and 12 columns.
# Note: There are 1000 rows of data in our csv. data.head only retturns the first 5.

# Here we will use our seaborn import as sns

#print(sns.pairplot(data[["v.id", "on road old", "on road now", "years", "km", "rating", "condition", "economy", 
#                   "top speed", "hp", "torque", "current price"]], diag_kind="kde")) # Output
# <seaborn.axisgrid.PairGrid object at 0x000001D98C4A0D90>

# Note: For some reason the pairplot was not able to populate in vsc code. 


# Here we will convert our data into a tensor

tensor_data = tf.constant(data)
print(tensor_data) # Output
#tf.Tensor(
#[[1.000000e+00 5.356510e+05 7.981860e+05 ... 7.300000e+01 1.230000e+02
#  3.513180e+05]
# [2.000000e+00 5.919110e+05 8.610560e+05 ... 7.400000e+01 9.500000e+01
#  2.850015e+05]
# [3.000000e+00 6.869900e+05 7.707620e+05 ... 5.300000e+01 9.700000e+01
#  2.153860e+05]
# ...
# [9.980000e+02 6.463440e+05 8.427330e+05 ... 1.130000e+02 8.900000e+01
#  4.058710e+05]
# [9.990000e+02 5.355590e+05 7.324390e+05 ... 1.120000e+02 1.280000e+02
#  7.439800e+04]
# [1.000000e+03 5.901050e+05 7.797430e+05 ... 9.900000e+01 9.600000e+01
#  4.149385e+05]], shape=(1000, 12), dtype=float64)
# Notice that this is our csv data concerted into a tensor

print(tensor_data.shape) # Output (1000, 12)
# Notice that this is the shape of that tensor


# Here we will shuffle our data so that no biased is shown for any particular data, making the inputs random.

#tensor_data = tf.random.shuffle(tensor_data)
#print(tensor_data[:5]) # This will return 5 random inputs from our csv data. Output
#tf.Tensor(
#[[9.730000e+02 6.226600e+05 7.411920e+05 6.000000e+00 1.263800e+05
#  3.000000e+00 1.000000e+00 9.000000e+00 1.450000e+02 1.020000e+02
#  1.110000e+02 1.728595e+05]
# [4.860000e+02 6.849240e+05 7.370910e+05 7.000000e+00 1.324980e+05
#  1.000000e+00 5.000000e+00 1.100000e+01 1.640000e+02 6.900000e+01
#  9.400000e+01 1.743990e+05]
# [1.000000e+00 5.356510e+05 7.981860e+05 3.000000e+00 7.894500e+04
#  1.000000e+00 2.000000e+00 1.400000e+01 1.770000e+02 7.300000e+01
#  1.230000e+02 3.513180e+05]
# [6.910000e+02 6.342340e+05 8.823930e+05 4.000000e+00 1.157600e+05
#  1.000000e+00 4.000000e+00 1.500000e+01 1.450000e+02 5.100000e+01
#  1.380000e+02 2.955535e+05]
# [3.900000e+02 6.421810e+05 7.663900e+05 2.000000e+00 1.381750e+05
#  1.000000e+00 5.000000e+00 1.000000e+01 1.930000e+02 9.300000e+01
#  1.000000e+02 1.533755e+05]]
# Notice that if we compare the return we have here to the return from tensor_data = tf.constant(data), we see that
#we have 5 random selections from our data because of the formula tensor_data = tf.random.shuffle(tensor_data).


# Now that we shuffled our data, we are ready to break the data up, such that we have the inputs, represented by
#x, and the output, represented by y. (x,y)

# First we'll start by getting the x
# We want to select all of our rows and a few selected columns.
# We are going to collect from position 3 up until position 10. 
# That should be from columns (Years - Torque)

x = tensor_data[:,3:-1] # This is how we target the data we want as specified in the overview.
print(x.shape) # Output (1000, 8)
# Notice that our shape is not 1000 by 12 anymore, but it is 1000 by 8 now because we didn't want all
#of the columns in our csv.

# Now we will print 5 rows from our specified inputs.
print(x[:5]) # This is how we specify that we only want back the first 5 rows of data. Output
# tf.Tensor(
#[[4.00000e+00 1.36468e+05 1.00000e+00 6.00000e+00 1.30000e+01 1.95000e+02
#  7.00000e+01 1.40000e+02]
# [4.00000e+00 1.29251e+05 2.00000e+00 3.00000e+00 1.30000e+01 1.42000e+02
#  7.70000e+01 1.10000e+02]
# [6.00000e+00 5.75430e+04 1.00000e+00 5.00000e+00 9.00000e+00 1.72000e+02
#  9.90000e+01 1.05000e+02]
# [4.00000e+00 5.05120e+04 2.00000e+00 6.00000e+00 8.00000e+00 1.44000e+02
#  6.80000e+01 7.90000e+01]
# [3.00000e+00 1.32129e+05 2.00000e+00 8.00000e+00 1.20000e+01 2.00000e+02
#  9.50000e+01 7.90000e+01]], shape=(5, 8), dtype=float64)
# Notice that we got back 5 randomized rows from our csv
# Note: The returned is only randomized because it is still being randaomized in the code above. This doesn't
#happen automatically.

# Now we will repeat the same process to get the output, (y)

y = tensor_data[:,-1] # Here we are specifying that we want all the rows, but now we only want the last column
print(y[:5]) # Output tf.Tensor([273728.  181428.5  77277.  300499.5 116254.5], shape=(5,), dtype=float64)
# Note: We can also change the shape of our return to make it easier to read by expanding it.

y = tensor_data[:,-1]
y = tf.expand_dims(y, axis = -1)
print(y[:5]) # Output
# tf.Tensor(
#[[448905.5]
# [253387. ]
# [146568. ]
# [308376.5]
# [504643.5]], shape=(5, 1)
# Notice that we now have a 2-d instead of a 1-d.
# Note: Also notice that our returns are still randomized. This isn't automatic and won't occur unless specified.


# From this point, another very common transformation that we can do on our data to enable our model to train faster
#is by normalizing our data. 
# Infact we will be normalizing our input.
# For every input, we will subtract the mean and divide by the standard deviation. 

# In the example below, the mean is firgured out with the first eight values in the x column.
# The mean of the eight values is about 138. 

# If we want to normalize this data, we are going to have 109 - 138 divided by the standard deviation.
# Let's give the standard deviation a value of 150 for example purposes. 
# This means that this point 109 - 138 / by 150 gets converted to -0.193. 

# Next let's take 206 and subtract 138 and divide by the standard deviation
# 206 - 138 / 150 will gives 0.45

# So if we notice, our input features have been rescaled before passing them into our model.
# And to carry out this feature scaling tensorflow has this method.tf.kerpas.layers.Normalization. And
#this method is just a feature-wise normalization of the data. 




#    X(hp)  |   Y(prices K$)
#   109     |   8
#   100     |   9.3
#   113     |   7.5
#   97      |   8.87
#   206     |   15.16
#   124     |   8.31
#   162     |   12.8
#   150     |   11.50
#   80      |
#   156     |


# Here we will go over the process of normalizing our data.

normalizer = Normalization()
x_normalized = tf.constant([[3,4,5,6,7]])
print(normalizer(x_normalized)) # Output tf.Tensor([[3. 4. 5. 6. 7.]], shape=(1, 5), dtype=float32)
# Notice that this is the return output we get before specifying our mean. Nothing has changed with our inputs

# Here will run the same code, but this time specify the mean

normalizer = Normalization(mean=5, variance=4) # Here we will specify our mean
x_normalized = tf.constant([[3,4,5,6,7]])
print(normalizer(x_normalized)) # Output tf.Tensor([[-1.  -0.5  0.   0.5  1. ]], shape=(1, 5), dtype=float32)
# Notice that our originally inputs have been rescaled this time. That is because we specified mean terms
#in our Normalization()

# Now we will add an additional row of inputs and run the same code

normalizer = Normalization(mean=5, variance=4) # Here we will specify our mean
x_normalized = tf.constant([[3,4,5,6,7],
                            [4,5,6,7,8]])
print(normalizer(x_normalized)) # Output tf.Tensor(
#                                                  [[-1.  -0.5  0.   0.5  1. ]
#                                                   [-0.5  0.   0.5  1.   1.5]]
# Note: The default axis is set to -1, so the normalization occurs in our columns
# So basically what we are doing is going over every coloumn and normalizing.
# Example 3 - 5 (5 because there are 5 columns) and then dividing by our standard deviation (4)
#(4) because of the variance = 4. standard deviation = (std)
# std squared 2 = sqrt(variance)

# Then we will do the same process for the next number in that column. Example 4 - 5 / 4.
# And so on and so forth.

(4 - 5)/ 2 # This is the formula that is being performed our inputs
print((4-5)/2) # Output -0.5
# Notice that we have -0.5, just like the return in our inputs.

# Note: It is not every situation that we can get the mean up front. There are cases when we just have the data,
#so we wouldn't perform mean and variance up front.

# What tensorflow allows us to do, is to obtain the mean and variance automatically. So the tensorflow
#allows us to adapt to the data given.

# So if we are given this data ([[3,4,5,6,7], 
#                                [4,5,6,7,8]]) for example, what we are going to do is 
#get the mean and variance for the the first column, then the second column, and so on and so forth.
# And we'll also be able to normalize our data. 

# This is the how we would run the adapted version

normalizer = Normalization()
x_normalized = tf.constant([[3,4,5,6,7],
                            [4,5,6,7,8]])
normalizer.adapt(x_normalized)
print(normalizer(x_normalized)) # Output tf.Tensor(
#                                                  [[-1. -1. -1. -1. -1.]
#                                                   [ 1.  1.  1.  1.  1.]]
# Notice that this the new output we get. This is what we obtain when we adapt automatically.
# Now let's understand what is going on.
# Here, the mean is 3.5, so we have x - 3.5 divided by the std
# We got 3.5 because our first column inputs are 3 and 4, which would give us a mean of 3.5
# If we have a mean of 3.5, that means our standard deviation (std) is 0.5. The std of 0.5 is the 
#difference between each number in either direction. Example 3.5 is deviation of 0.5 from 3, and 
#it is also a 0.5 deviation from 4.

# Now we will run an example
print((3-3.5)/0.5) # Output -1.0
# Notice that our return matches the first returned number in our first column.

# Now let's run the example for the second number in our first column (4).
print((4-3.5)/0.5) # Output 1.0
# Notice that our return matches the second returned number in our first column

# Now if we try this approach on the next column, it will not work correctly. That is because every
#column will have its own mean. The mean for 4 and 5 is 4.5.

# Now let's run an example
print((4-4.5)/0.5) # Output -1.0
# Notice that we now have the correct return for the first number in our second column.

# Now let's check the second number in our second column just to be sure.
print((5-4.5)/0.5) # Output 1.0
# Notice that we have the correct return for the second number in our second column.

# Now let's add an addition row to our inputs

normalizer = Normalization()
x_normalized = tf.constant([[3,4,5,6,7],
                            [4,5,6,7,8],
                            [32,1,56,3,5]])
normalizer.adapt(x_normalized)
print(normalizer(x_normalized)) # Output tf.Tensor(
#                                                  [[-0.7439795   0.39223233 -0.72800297  0.3922322   0.2672614 ]
#                                                   [-0.6695816   0.98058075 -0.6860028   0.9805806   1.0690452 ]
#                                                   [ 1.4135611  -1.3728129   1.4140056  -1.3728131  -1.3363061 ]]
# Notice that we are returned our normlized data automatically, unlike when we had to specify the mean
#for each and every column.
# Here our normalizer adapts to our input data 


# Here we will work on csv data.
print(x.shape) # Output (1000, 8)

# Now we are going to normalize for each of our 8 columns
# And instead of struggling to get the mean and variance for each column, tensorflow permits us to adapt
#to our data set

normalizer = Normalization()
normalizer.adapt(x)
print(normalizer(x)) # Output tf.Tensor(
#                                       [[-0.9084985  -0.7320671  -1.4178834  ...  0.524257   -0.56303626
#                                          0.93010384]
#                                        [ 0.8374949   0.5816051   1.4350008  ... -0.9799913  -0.5142717
#                                         -0.40017724]
#                                        [-1.4904963   1.1073486  -0.7046623  ...  0.7317395  -1.5383282
#                                         -0.30515718]
#                                         ...
#                                        [ 1.4194927  -0.4651454  -1.4178834  ...  1.509799    1.3875475
#                                         -0.68523747]
#                                        [-1.4904963   1.3798648   0.72177976 ...  0.8873514   1.3387829
#                                          1.167654  ]
#                                        [ 0.25549713 -1.1319177   0.72177976 ...  1.6654109   0.70484316
#                                         -0.3526672 ]]
# Notice that we are returned a tensor of our csv data

print(x) # Output tf.Tensor(
#                           [[3.00000e+00 7.89450e+04 1.00000e+00 ... 1.77000e+02 7.30000e+01
#                             1.23000e+02]
#                            [6.00000e+00 1.17220e+05 5.00000e+00 ... 1.48000e+02 7.40000e+01
#                             9.50000e+01]
#                            [2.00000e+00 1.32538e+05 2.00000e+00 ... 1.81000e+02 5.30000e+01
#                             9.70000e+01]
#                             ...
#                            [7.00000e+00 8.67220e+04 1.00000e+00 ... 1.96000e+02 1.13000e+02
#                             8.90000e+01]
#                            [2.00000e+00 1.40478e+05 4.00000e+00 ... 1.84000e+02 1.12000e+02
#                             1.28000e+02]
#                            [5.00000e+00 6.72950e+04 4.00000e+00 ... 1.99000e+02 9.90000e+01
#                             9.60000e+01]]
# Notice that we are returned the normalized version of our csv data.
# Note: It was done automatically by our adapt method

model = tf.keras.Sequential()
model.add(normalizer)
model.add(Dense(1))
model.summary()
print(model.summary())

#tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
#print(tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True))

#model.compile(loss= MeanSquaredError())
#print(model.compile(loss= MeanSquaredError()))

model.compile(optimizer=Adam(learning_rate = 1),
               loss= MeanAbsoluteError(),
               metrics= RootMeanSquaredError())
print(model.compile(optimizer=Adam(learning_rate = 1), 
                    loss= MeanAbsoluteError(),
                    metrics= RootMeanSquaredError()))

#model.compile(loss= Huber(delta=0.2))
#print(model.compile(loss= Huber(delta=0.2)))

history = model.fit(x,y, epochs = 100, verbose = 1)
#print(history = model.fit(x,y, epochs = 100, verbose = 1))

history.history
print(history.history)


plt.plot(history.history["loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train"])
plt.show()

plt.plot(history.history["root_mean_squared_error"])
plt.title("model performance")
plt.ylabel("rmse")
plt.xlabel("epoch")
plt.legend(["train"])
plt.show()

history = model.fit(x,y, epochs = 100, verbose = 1)

history.history
print(history.history)

model.evaluate(x,y)

TRAIN_RATIO = 0.8 # This is where we specify 80 percent of our training data
VAL_RATIO = 0.1 # This is where we specify 10 percent for validation our data
TEST_RATIO = 0.1 # This is where we specify 10 percent for testing our data
DATASET_SIZE = len(x) # The length of the data is (1000, 8) as specified by our csv

x_train = x[:int(DATASET_SIZE*TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE*TRAIN_RATIO)]
print(x_train.shape)
print(y_train.shape)

x_val = x[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_val = y[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
print(x_val.shape)
print(y_val.shape)

x_test = x[int(DATASET_SIZE*(TRAIN_RATIO+TEST_RATIO)):]
y_test = y[int(DATASET_SIZE*(TRAIN_RATIO+TEST_RATIO)):]
print(x_test.shape) 
print(y_test.shape)

history = model.fit(x_train,y_train,validation_data=(x_val,y_val), epochs = 100, verbose = 1)

history.history
print(history.history)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val_loss"])
plt.show()

plt.plot(history.history["root_mean_squared_error"])
plt.plot(history.history["val_root_mean_squared_error"])
plt.title("model performance")
plt.ylabel("rmse")
plt.xlabel("epoch")
plt.legend(["train", "val_rmse"])
plt.show()

model.evaluate(x_val,y_val)

model.evaluate(x_test,y_test)

print(x_test)

print(x_test.shape)

print(model.predict(x_test))

print(model.predict(x_test).shape)

# print(model.predict(x_test[0]))

print(model.predict(tf.expand_dims(x_test[0], axis = 0)).shape)

print(model.predict(tf.expand_dims(x_test[0], axis = 0)))

print(y_test[0])

#model = tf.keras.Sequential()
#model.add(InputLayer(input_shape = (8,)))
#model.add(normalizer)
#model.add(Dense(32, activation = "relu"))
#model.add(Dense(32, activation = "relu"))
#model.add(Dense(32, activation = "relu"))
#model.add(Dense(1)) 

#model.summary()

#print(model.summary())

model = tf.keras.Sequential([
                             InputLayer(input_shape = (8,)),
                             normalizer,
                             Dense(32, activation = "relu"),
                             Dense(32, activation = "relu"),
                             Dense(32, activation = "relu"),
                             Dense(1), 
])

model.summary()

print(model.summary())

model = tf.keras.Sequential([
                             InputLayer(input_shape = (8,)),
                             normalizer,
                             Dense(128, activation = "relu"),
                             Dense(128, activation = "relu"),
                             Dense(128, activation = "relu"),
                             Dense(1), 
])

model.summary()

print(model.summary())

#tf.keras.utils.plot_model(model, to_file = "model.png", show_shapes = True)
#print(tf.keras.utils.plot_model(model, to_file = "model.png", show_shapes = True))

model.compile(optimizer=Adam(learning_rate = 0.1),
               loss= MeanAbsoluteError(),
               metrics= RootMeanSquaredError())
print(model.compile(optimizer=Adam(learning_rate = 0.1), 
                    loss= MeanAbsoluteError(),
                    metrics= RootMeanSquaredError()))

history = model.fit(x_train,y_train,validation_data=(x_val,y_val), epochs = 100, verbose = 1)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val_loss"])
plt.show()

plt.plot(history.history["root_mean_squared_error"])
plt.plot(history.history["val_root_mean_squared_error"])
plt.title("model performance")
plt.ylabel("rmse")
plt.xlabel("epoch")
plt.legend(["train", "val_rmse"])
plt.show()

model.evaluate(x_test,y_test)


y_a = list(y_test[:,0].numpy())

y_p = list(model.predict(x_test)[:,0])
print(y_p)


ind = np.arange(100)
plt.figure(figsize=(40,20))

width = 0.1

plt.bar(ind, y_p, width, label = "Predicted Car Price")
plt.bar(ind + width, y_a, width, label = "Actual Car Price")

plt.xlabel("Actual vs Predicted Price")
plt.ylabel("Car Price Prices")

plt.show()


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

for x,y in train_dataset:
    print(x,y)
    break

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)

history = model.fit(train_dataset, validation_data=val_dataset, epochs = 100, verbose = 1)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val_loss"])
plt.show()

plt.plot(history.history["root_mean_squared_error"])
plt.plot(history.history["val_root_mean_squared_error"])
plt.title("model performance")
plt.ylabel("rmse")
plt.xlabel("epoch")
plt.legend(["train", "val_rmse"])
plt.show()

model.evaluate(x_test,y_test)

y_a = list(y_test[:,0].numpy())

y_p = list(model.predict(x_test)[:,0])
print(y_p)


ind = np.arange(100)
plt.figure(figsize=(40,20))

width = 0.1

plt.bar(ind, y_p, width, label = "Predicted Car Price")
plt.bar(ind + width, y_a, width, label = "Actual Car Price")

plt.xlabel("Actual vs Predicted Price")
plt.ylabel("Car Price Prices")

plt.show()