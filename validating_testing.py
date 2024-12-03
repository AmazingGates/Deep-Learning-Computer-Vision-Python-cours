import tensorflow as tf # For our Models
import pandas as pd # For reading and processing Data
import seaborn as sns # For Visualization
import keras.layers 
from keras.layers import Normalization, Dense
import matplotlib.pyplot as plt
import keras.losses
from keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import numpy as np


# This is where we will be going over the Validating and Testing of our Model.

# To better understand validation and testing, let's take a simple example. 
# Imagine that we get a class on the first day and we are being taught some course materials.
# So we've been taught the course materials throughout the entire term, and then at the end of the term
#the teacher tells every student to produce their own exams.
# And they will also have to take the exams they produced once they are done.

# So in essence they are coming up with the exam questions that they will have to answer.

# It's clear that most students will pass with an accuracy greater than 18 or 20, simply because they have come up 
#with the question that they have to answer in order to pass.

# It is possible that a student who has sat through the course work and mastered everything does come up
#with some tough questions and gets a max of 18 on their exam.

# It is also possible that a student struggled getting through the course work and came up with some with some
# rellay easy questions and they score greater than an 18 on their exam, without much effort or mastering
#the course work.

# That's why this strategy can be dangerous. 

# We need that external validation from the teacher.

# So far what we've been doing is kind of like the first method where we are producing the exams that we are
#testing ourselves with.

# That's simply because we're using our full data set training on our data set, and then evaluating the
#performance without taking into consideration data that our model has never seen before.

# So the idea is to be able to create a model that when it sees new data is able to come up with a reasonable
#current price which is as close as possible to the actual current price.

# And so when dealing with Machine Learning Models, it's important to break data up into sections.

# For example, let's say we have 1000 examples. 

# We could break this up into 800 examples which we train our data on, and then test that on the other 200
#remaining examples.

# So with that example, the model has never seen the 200 examples that will be tested with the 800 examples
#which we trained.

# This is a very important use of shuffling, because with shuffling we are sure that there is no bias in 
#the way the data set is constituted.

# If we just break up the data, we have one for testing, and one part for training, which has been ramdonly
#built.

# So we're saying if a model, for example, has a performance of, let's say rmse = 5 on the training data.

# Then on the test data, it has a rmse value of 50,000.

# Then it is clear that this is a very poor performing model, as it does well on data that it has seen,
#but on data it hasn't seen yet it doesn't perform well.

# And so recall that machine learning is all about empowering the machines to do stuff humans do.
# and obviously humans use some intelligence in doing all those task.

# So if we want to build a model we have to ensure that it performs well on data that it has never seen
# And so we always have to split the data set.

# Firstly, we shuffle our data set.
# It's very important we split this model set before proceeding with modeliing and training.
# 
# Now sometimes we don't want to wait until the model has trained before testing it out, and 
#discovery that it performs very poorly on data it has never seen.

# So what we want to do is, while doing the training, we want too be able to see how it performs on data
#that it has not yet seen.

# Now we will do validation testing on our csv data.

# This is how we will specify our training.

# TRAIN_RATIO = 0.8 # This is where we specify 80 percent of our training data
# VAL_RATIO = 0.1 # This is where we specify 10 percent for validating our data
# TEST_RATIO = 0.1 # This is where we specify 10 percent for testing our data
# DATASET_SIZE = len(x) # The length of the data is (1000, 8) as specified by our csv

# x_train = x##[:int(DATASET_SIZE*TRAIN_RATIO)]
# y_train = y##[:int(DATASET_SIZE*TRAIN_RATIO)]
# print(x_train.shape) # Output (800, 8)
# print(y_train.shape) # Output (800, 1)

# Notice that our returns are now in the shape of our training data

# Note that our dataset is now 800, which means we have 200 left over to validate and test our dataset against.
# 100 will go to our Validation set, and the last 100 will go to our test set.

# Next we will do the same for the VAL_RATIO

# x_val = x##[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
# y_val = y##[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
# print(x_val.shape) # Output (100, 8)
# print(y_val.shape) # Output (100, 1)

# Notice that our returns our now in the shape of our Validating size specifications

# Last we will run our TEST_Ratio

# x_test = x##[int(DATASET_SIZE*(TRAIN_RATIO+TEST_RATIO)):]
# y_test = y##[int(DATASET_SIZE*(TRAIN_RATIO+TEST_RATIO)):]
# print(x_test.shape) # Output (100, 8)
# print(y_test.shape) # Output (100, 1)

# Notice that our returns our now in the shape of our Testing size specifications

# One imporrtant note we shouldd remember is we have to avoid data leaking from the training into the 
#validation and the testing.

# So this means even when doing normalization and we're trying to adapt our normalizer to our data, we have
#to not use the validation and testing, we just have to use the training.

# So next we are going to use only the training set to adapt our normalizer to our data.

# From here we have to modify the way we do our training.

# From here we just have to specify that we have X_train and Y_train then our validation data is equal.

# We have a X_Val and Y_VAL

# We specify X_VAL Y_VAL as our validation data, and then we have X_train and Y_train as our training data.

# Now note that when doing this with validation data we actually specify X_VAL and Y_VAL, but there is another
#argument which is the validation split where we just specify the fraction of the training data to be used 
#as validation data.

# But for now, we will use the method we have below.

# This is how we would write the algorithm

# history = model.fit(X_train,Y_train,validation_data=(x_val,y_val), #epochs = 100, verbose = 1)
# This is the output

#Epoch 1/100
#25/25 ##[==============================] - 0s 10ms/step - loss: 306870.7500 - root_mean_squared_error: 331857.3125 - val_loss: 299053.9688 - val_root_mean_squared_error: 322364.2500
#Epoch 2/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 306845.4688 - root_mean_squared_error: 331833.0312 - val_loss: 299029.3438 - val_root_mean_squared_error: 322340.0312
#Epoch 3/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306820.2812 - root_mean_squared_error: 331807.5625 - val_loss: 299004.9688 - val_root_mean_squared_error: 322315.5000
#Epoch 4/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306794.7500 - root_mean_squared_error: 331782.2188 - val_loss: 298980.2188 - val_root_mean_squared_error: 322291.4062
#Epoch 5/100
#25/25 ##[==============================] - 0s 14ms/step - loss: 306769.3750 - root_mean_squared_error: 331757.4062 - val_loss: 298955.6250 - val_root_mean_squared_error: 322267.0000
#Epoch 6/100
#25/25 ##[==============================] - 0s 12ms/step - loss: 306744.0625 - root_mean_squared_error: 331732.3750 - val_loss: 298931.0312 - val_root_mean_squared_error: 322242.7500
#Epoch 7/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306718.5938 - root_mean_squared_error: 331706.6562 - val_loss: 298906.4062 - val_root_mean_squared_error: 322217.8438
#Epoch 8/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306693.2500 - root_mean_squared_error: 331681.5000 - val_loss: 298881.6875 - val_root_mean_squared_error: 322193.4062
#Epoch 9/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306667.9375 - root_mean_squared_error: 331655.8125 - val_loss: 298857.1250 - val_root_mean_squared_error: 322168.5000
#Epoch 10/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306642.7188 - root_mean_squared_error: 331630.5625 - val_loss: 298832.5938 - val_root_mean_squared_error: 322144.5312
#Epoch 11/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306617.1562 - root_mean_squared_error: 331605.5938 - val_loss: 298808.0000 - val_root_mean_squared_error: 322120.5000
#Epoch 12/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306591.7500 - root_mean_squared_error: 331579.9375 - val_loss: 298783.4688 - val_root_mean_squared_error: 322095.2500
#Epoch 13/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306566.4688 - root_mean_squared_error: 331554.9062 - val_loss: 298758.4062 - val_root_mean_squared_error: 322071.0938
#Epoch 14/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306541.0625 - root_mean_squared_error: 331529.1562 - val_loss: 298734.0938 - val_root_mean_squared_error: 322046.1562
#Epoch 15/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306515.6250 - root_mean_squared_error: 331504.1250 - val_loss: 298709.8438 - val_root_mean_squared_error: 322023.2188
#Epoch 16/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306490.2188 - root_mean_squared_error: 331478.7500 - val_loss: 298685.0000 - val_root_mean_squared_error: 321997.7812
#Epoch 17/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306464.8750 - root_mean_squared_error: 331453.3750 - val_loss: 298660.4062 - val_root_mean_squared_error: 321973.2188
#Epoch 18/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306439.3438 - root_mean_squared_error: 331427.6875 - val_loss: 298635.9688 - val_root_mean_squared_error: 321949.0625
#Epoch 19/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306414.0625 - root_mean_squared_error: 331402.6875 - val_loss: 298610.9062 - val_root_mean_squared_error: 321924.4062
#Epoch 20/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306388.7500 - root_mean_squared_error: 331376.7812 - val_loss: 298586.8750 - val_root_mean_squared_error: 321899.8438
#Epoch 21/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306363.2812 - root_mean_squared_error: 331351.7188 - val_loss: 298561.7812 - val_root_mean_squared_error: 321875.5000
#Epoch 22/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 306337.9062 - root_mean_squared_error: 331326.5625 - val_loss: 298537.5312 - val_root_mean_squared_error: 321851.1875
#Epoch 23/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306312.8125 - root_mean_squared_error: 331300.8125 - val_loss: 298512.8438 - val_root_mean_squared_error: 321825.4375
#Epoch 24/100
#25/25 ##[==============================] - 0s 12ms/step - loss: 306287.1250 - root_mean_squared_error: 331274.7188 - val_loss: 298488.4062 - val_root_mean_squared_error: 321801.7188
#Epoch 25/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306261.9062 - root_mean_squared_error: 331249.9375 - val_loss: 298463.8750 - val_root_mean_squared_error: 321777.5000
#Epoch 26/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306236.1562 - root_mean_squared_error: 331224.7188 - val_loss: 298439.0938 - val_root_mean_squared_error: 321753.1875
#Epoch 27/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306210.9688 - root_mean_squared_error: 331199.6562 - val_loss: 298414.3750 - val_root_mean_squared_error: 321728.3750
#Epoch 28/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306185.5312 - root_mean_squared_error: 331174.1250 - val_loss: 298389.8438 - val_root_mean_squared_error: 321703.8438
#Epoch 29/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306160.1875 - root_mean_squared_error: 331149.0000 - val_loss: 298365.0938 - val_root_mean_squared_error: 321679.8438
#Epoch 30/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 306135.0938 - root_mean_squared_error: 331123.2500 - val_loss: 298340.6250 - val_root_mean_squared_error: 321654.4062
#Epoch 31/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306109.4062 - root_mean_squared_error: 331097.8438 - val_loss: 298316.2500 - val_root_mean_squared_error: 321631.4375
#Epoch 32/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 306084.2188 - root_mean_squared_error: 331073.1250 - val_loss: 298291.3125 - val_root_mean_squared_error: 321606.0625
#Epoch 33/100
#25/25 ##[==============================] - 0s 9ms/step - loss: 306058.9062 - root_mean_squared_error: 331047.4688 - val_loss: 298267.4375 - val_root_mean_squared_error: 321581.9062
#Epoch 34/100
#25/25 ##[==============================] - 0s 12ms/step - loss: 306033.1250 - root_mean_squared_error: 331021.5938 - val_loss: 298242.4062 - val_root_mean_squared_error: 321557.3438
#Epoch 35/100
#25/25 ##[==============================] - 0s 14ms/step - loss: 306008.0625 - root_mean_squared_error: 330996.4375 - val_loss: 298217.7500 - val_root_mean_squared_error: 321532.5000
#Epoch 36/100
#25/25 ##[==============================] - 0s 19ms/step - loss: 305982.7500 - root_mean_squared_error: 330970.5312 - val_loss: 298193.4375 - val_root_mean_squared_error: 321507.8750
#Epoch 37/100
#25/25 ##[==============================] - 1s 16ms/step - loss: 305957.2500 - root_mean_squared_error: 330946.1875 - val_loss: 298168.8438 - val_root_mean_squared_error: 321484.9375
#Epoch 38/100
#25/25 ##[==============================] - 0s 12ms/step - loss: 305931.7500 - root_mean_squared_error: 330920.8750 - val_loss: 298144.1875 - val_root_mean_squared_error: 321459.3438
#Epoch 39/100
#25/25 ##[==============================] - 0s 9ms/step - loss: 305906.3125 - root_mean_squared_error: 330895.3125 - val_loss: 298119.3438 - val_root_mean_squared_error: 321435.2188
#Epoch 40/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305881.0312 - root_mean_squared_error: 330870.2500 - val_loss: 298094.7188 - val_root_mean_squared_error: 321410.7500
#Epoch 41/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305855.5938 - root_mean_squared_error: 330845.2500 - val_loss: 298070.2812 - val_root_mean_squared_error: 321386.7188
#Epoch 42/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305830.1875 - root_mean_squared_error: 330819.8438 - val_loss: 298045.7500 - val_root_mean_squared_error: 321362.0000
#Epoch 43/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305804.8125 - root_mean_squared_error: 330794.4688 - val_loss: 298020.9062 - val_root_mean_squared_error: 321337.3125
#Epoch 44/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305779.4688 - root_mean_squared_error: 330768.8125 - val_loss: 297996.5312 - val_root_mean_squared_error: 321312.8125
#Epoch 45/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305754.1875 - root_mean_squared_error: 330743.2188 - val_loss: 297972.0312 - val_root_mean_squared_error: 321288.5000
#Epoch 46/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305728.7812 - root_mean_squared_error: 330718.1562 - val_loss: 297947.5625 - val_root_mean_squared_error: 321264.9062
#Epoch 47/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305703.5000 - root_mean_squared_error: 330692.9062 - val_loss: 297922.9688 - val_root_mean_squared_error: 321239.3438
#Epoch 48/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305677.9375 - root_mean_squared_error: 330667.3750 - val_loss: 297898.4062 - val_root_mean_squared_error: 321215.8438
#Epoch 49/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305652.6562 - root_mean_squared_error: 330642.4375 - val_loss: 297873.9688 - val_root_mean_squared_error: 321191.5625
#Epoch 50/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305627.4062 - root_mean_squared_error: 330616.5000 - val_loss: 297848.7812 - val_root_mean_squared_error: 321165.4062
#Epoch 51/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 305601.8750 - root_mean_squared_error: 330591.0938 - val_loss: 297824.6250 - val_root_mean_squared_error: 321142.0000
#Epoch 52/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305576.3750 - root_mean_squared_error: 330566.3750 - val_loss: 297799.8125 - val_root_mean_squared_error: 321117.9375
#Epoch 53/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305551.0000 - root_mean_squared_error: 330541.1250 - val_loss: 297775.3125 - val_root_mean_squared_error: 321093.4688
#Epoch 54/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305525.6250 - root_mean_squared_error: 330515.6562 - val_loss: 297750.9375 - val_root_mean_squared_error: 321069.5000
#Epoch 55/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305500.3125 - root_mean_squared_error: 330490.7188 - val_loss: 297726.0312 - val_root_mean_squared_error: 321044.1562
#Epoch 56/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305475.2812 - root_mean_squared_error: 330465.1875 - val_loss: 297701.5312 - val_root_mean_squared_error: 321019.7188
#Epoch 57/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 305449.5938 - root_mean_squared_error: 330439.5000 - val_loss: 297676.7500 - val_root_mean_squared_error: 320994.7812
#Epoch 58/100
#25/25 ##[==============================] - 0s 9ms/step - loss: 305424.1250 - root_mean_squared_error: 330413.6875 - val_loss: 297652.5625 - val_root_mean_squared_error: 320970.7500
#Epoch 59/100
#25/25 ##[==============================] - 0s 10ms/step - loss: 305398.7500 - root_mean_squared_error: 330389.4062 - val_loss: 297627.8750 - val_root_mean_squared_error: 320947.1562
#Epoch 60/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 305373.3750 - root_mean_squared_error: 330364.2500 - val_loss: 297603.0312 - val_root_mean_squared_error: 320922.8125
#Epoch 61/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305347.9688 - root_mean_squared_error: 330338.3750 - val_loss: 297578.3750 - val_root_mean_squared_error: 320897.1875
#Epoch 62/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 305322.9688 - root_mean_squared_error: 330312.9375 - val_loss: 297554.3125 - val_root_mean_squared_error: 320873.2500
#Epoch 63/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 305297.1875 - root_mean_squared_error: 330287.3125 - val_loss: 297529.3125 - val_root_mean_squared_error: 320848.6875
#Epoch 64/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 305271.9688 - root_mean_squared_error: 330262.7812 - val_loss: 297504.7812 - val_root_mean_squared_error: 320824.7188
#Epoch 65/100
#25/25 ##[==============================] - 0s 12ms/step - loss: 305246.7188 - root_mean_squared_error: 330238.1250 - val_loss: 297479.7500 - val_root_mean_squared_error: 320800.8438
#Epoch 66/100
#25/25 ##[==============================] - 0s 10ms/step - loss: 305221.1562 - root_mean_squared_error: 330212.3750 - val_loss: 297455.5938 - val_root_mean_squared_error: 320775.6875
#Epoch 67/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305195.7188 - root_mean_squared_error: 330186.8750 - val_loss: 297430.9375 - val_root_mean_squared_error: 320751.7812
#Epoch 68/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305170.4375 - root_mean_squared_error: 330161.8438 - val_loss: 297406.4062 - val_root_mean_squared_error: 320727.2812
#Epoch 69/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305145.0000 - root_mean_squared_error: 330136.4062 - val_loss: 297382.1875 - val_root_mean_squared_error: 320702.5625
#Epoch 70/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305119.6250 - root_mean_squared_error: 330110.3438 - val_loss: 297357.3125 - val_root_mean_squared_error: 320677.7500
#Epoch 71/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305094.2500 - root_mean_squared_error: 330085.5000 - val_loss: 297332.9688 - val_root_mean_squared_error: 320654.0000
#Epoch 72/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305068.8750 - root_mean_squared_error: 330060.1562 - val_loss: 297308.1250 - val_root_mean_squared_error: 320629.4375
#Epoch 73/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305043.4062 - root_mean_squared_error: 330035.0312 - val_loss: 297283.8125 - val_root_mean_squared_error: 320604.8438
#Epoch 74/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 305018.0312 - root_mean_squared_error: 330009.5625 - val_loss: 297259.0000 - val_root_mean_squared_error: 320581.0625
#Epoch 75/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 304992.8125 - root_mean_squared_error: 329984.0625 - val_loss: 297234.4062 - val_root_mean_squared_error: 320555.5000
#Epoch 76/100
#25/25 ##[==============================] - 0s 8ms/step - loss: 304967.3438 - root_mean_squared_error: 329958.2188 - val_loss: 297209.7500 - val_root_mean_squared_error: 320530.7500
#Epoch 77/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304941.8438 - root_mean_squared_error: 329933.0312 - val_loss: 297185.0312 - val_root_mean_squared_error: 320507.0938
#Epoch 78/100
#25/25 ##[==============================] - 0s 11ms/step - loss: 304916.5312 - root_mean_squared_error: 329908.0000 - val_loss: 297160.6250 - val_root_mean_squared_error: 320482.5000
#Epoch 79/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304891.0938 - root_mean_squared_error: 329882.9688 - val_loss: 297135.9062 - val_root_mean_squared_error: 320458.5938
#Epoch 80/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304865.8750 - root_mean_squared_error: 329857.8125 - val_loss: 297111.6250 - val_root_mean_squared_error: 320433.6875
#Epoch 81/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304840.2500 - root_mean_squared_error: 329831.6562 - val_loss: 297086.9688 - val_root_mean_squared_error: 320408.8125
#Epoch 82/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304814.9062 - root_mean_squared_error: 329806.5000 - val_loss: 297062.5312 - val_root_mean_squared_error: 320385.0625
#Epoch 83/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304789.5625 - root_mean_squared_error: 329781.3125 - val_loss: 297037.5938 - val_root_mean_squared_error: 320360.7188
#Epoch 84/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304764.0938 - root_mean_squared_error: 329756.0625 - val_loss: 297013.0625 - val_root_mean_squared_error: 320336.0938
#Epoch 85/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304738.6875 - root_mean_squared_error: 329730.4062 - val_loss: 296988.4375 - val_root_mean_squared_error: 320310.9688
#Epoch 86/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304713.3750 - root_mean_squared_error: 329705.3750 - val_loss: 296963.9688 - val_root_mean_squared_error: 320287.2812
#Epoch 87/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304687.9688 - root_mean_squared_error: 329680.1875 - val_loss: 296939.7188 - val_root_mean_squared_error: 320263.4688
#Epoch 88/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304662.7812 - root_mean_squared_error: 329654.6250 - val_loss: 296914.5938 - val_root_mean_squared_error: 320237.4688
#Epoch 89/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304637.2812 - root_mean_squared_error: 329629.4688 - val_loss: 296890.3750 - val_root_mean_squared_error: 320214.1562
#Epoch 90/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304611.9375 - root_mean_squared_error: 329604.0625 - val_loss: 296865.7812 - val_root_mean_squared_error: 320189.7188
#Epoch 91/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304586.4375 - root_mean_squared_error: 329578.8438 - val_loss: 296841.1875 - val_root_mean_squared_error: 320165.4375
#Epoch 92/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304561.0000 - root_mean_squared_error: 329553.3438 - val_loss: 296816.4375 - val_root_mean_squared_error: 320140.3750
#Epoch 93/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304535.5625 - root_mean_squared_error: 329527.6562 - val_loss: 296791.8750 - val_root_mean_squared_error: 320115.8438
#Epoch 94/100
#25/25 ##[==============================] - 0s 6ms/step - loss: 304510.3125 - root_mean_squared_error: 329502.6562 - val_loss: 296767.3125 - val_root_mean_squared_error: 320091.5938
#Epoch 95/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304484.8125 - root_mean_squared_error: 329477.0000 - val_loss: 296743.0312 - val_root_mean_squared_error: 320067.4062
#Epoch 96/100
#25/25 ##[==============================] - 0s 11ms/step - loss: 304459.4688 - root_mean_squared_error: 329452.1250 - val_loss: 296718.5938 - val_root_mean_squared_error: 320043.7812
#Epoch 97/100
#25/25 ##[==============================] - 0s 10ms/step - loss: 304434.1250 - root_mean_squared_error: 329426.6250 - val_loss: 296693.8438 - val_root_mean_squared_error: 320018.4688
#Epoch 98/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304408.8125 - root_mean_squared_error: 329400.7500 - val_loss: 296669.3438 - val_root_mean_squared_error: 319993.5000
#Epoch 99/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304383.3750 - root_mean_squared_error: 329375.9688 - val_loss: 296644.6875 - val_root_mean_squared_error: 319969.6250
#Epoch 100/100
#25/25 ##[==============================] - 0s 7ms/step - loss: 304357.8750 - root_mean_squared_error: 329350.6562 - val_loss: 296620.1875 - val_root_mean_squared_error: 319945.5938

# Notice that we have some extra outputs. We have the val_loss and the val_root_mean_squared_error

# Next will run hisory.history again to get back only the values of the old inputs and the new inputs.
# This is the output

# {'loss': ##[306872.1875, 306846.875, 306821.46875, 306796.0625, 306770.875, 306745.40625, 306720.03125, 306694.625, 
#306669.375, 306644.09375, 306618.59375, 306593.28125, 306567.90625, 306542.375, 306517.21875, 306491.625, 306466.34375,
#306440.9375, 306415.65625, 306390.125, 306364.75, 306339.8125, 306314.25, 306288.84375, 306263.34375, 306238.09375, 
#306212.59375, 306187.0625, 306161.78125, 306136.4375, 306111.25, 306085.8125, 306060.25, 306035.1875, 306009.53125, 
#305984.28125, 305958.75, 305933.28125, 305908.0, 305882.8125, 305857.25, 305831.9375, 305806.34375, 305781.3125, 
#305755.8125, 305730.40625, 305705.03125, 305679.59375, 305654.40625, 305628.78125, 305603.5, 305578.0, 305552.78125, 
#305527.25, 305502.09375, 305476.53125, 305451.125, 305425.8125, 305400.375, 305374.9375, 305349.625, 305324.1875, 
#305298.8125, 305273.625, 305248.09375, 305222.90625, 305197.3125, 305172.0, 305146.625, 305121.1875, 305096.0, 
#305070.53125, 305045.1875, 305019.75, 304994.25, 304968.84375, 304943.59375, 304918.125, 304892.96875, 304867.53125, 
#304842.03125, 304816.5625, 304791.125, 304766.0, 304740.625, 304715.1875, 304689.625, 304664.40625, 304639.0, 
#304613.59375, 304588.375, 304562.75, 304537.59375, 304512.0625, 304486.59375, 304461.1875, 304435.90625, 304410.53125, 
#304385.125, 304359.875], 'root_mean_squared_error': ##[331857.46875, 331832.25, 331806.75, 331781.65625, 331756.59375, 
#331731.28125, 331705.6875, 331680.375, 331655.375, 331629.53125, 331604.78125, 331579.65625, 331554.0625, 331528.84375, 
#331503.84375, 331478.21875, 331452.59375, 331426.96875, 331402.0625, 331376.5, 331351.78125, 331326.15625, 
#331300.84375, 331275.84375, 331250.9375, 331225.375, 331200.09375, 331174.59375, 331149.21875, 331124.21875, 
#331098.84375, 331073.75, 331048.125, 331023.59375, 330997.46875, 330972.90625, 330947.0625, 330921.40625, 330896.21875, 
#330870.875, 330845.53125, 330820.5625, 330795.1875, 330769.71875, 330744.53125, 330719.28125, 330693.5, 330668.4375, 
#330643.0625, 330617.40625, 330592.5, 330567.21875, 330541.75, 330516.34375, 330491.09375, 330465.75, 330441.125, 
#330415.78125, 330390.34375, 330364.78125, 330339.21875, 330314.03125, 330288.90625, 330264.0, 330238.15625, 
#330213.15625, 330187.1875, 330162.25, 330137.0625, 330111.8125, 330086.1875, 330060.75, 330035.78125, 330010.75, 
#329985.03125, 329959.59375, 329934.90625, 329909.21875, 329884.25, 329858.96875, 329833.0, 329807.84375, 329782.34375, 
#329757.875, 329732.03125, 329706.3125, 329681.25, 329655.9375, 329630.8125, 329605.375, 329579.875, 329554.3125, 
#329529.1875, 329503.25, 329478.5, 329453.5, 329427.75, 329402.75, 329377.875, 329351.90625], 'val_loss': ##[299048.3125, 
#299023.96875, 298999.1875, 298974.59375, 298950.09375, 298925.46875, 298900.8125, 298876.25, 298851.46875, 298826.8125, 
#298802.25, 298777.90625, 298753.09375, 298728.71875, 298704.34375, 298679.6875, 298655.1875, 298630.3125, 298606.09375, 
#298581.375, 298556.4375, 298531.75, 298507.25, 298482.5625, 298458.09375, 298433.3125, 298409.0, 298384.5, 298359.625, 
#298335.09375, 298310.96875, 298285.875, 298261.375, 298237.40625, 298212.25, 298187.71875, 298163.3125, 298138.6875, 
#298114.125, 298089.1875, 298064.625, 298040.34375, 298015.59375, 297991.46875, 297966.65625, 297941.84375, 
#297917.59375, 297892.6875, 297868.125, 297843.34375, 297818.71875, 297794.15625, 297769.8125, 297745.15625, 
#297720.59375, 297696.1875, 297671.59375, 297646.71875, 297622.15625, 297597.65625, 297573.09375, 297548.53125, 
#297524.09375, 297499.46875, 297474.875, 297450.375, 297425.65625, 297401.40625, 297376.28125, 297351.8125, 297327.375, 
#297302.53125, 297278.125, 297253.46875, 297229.09375, 297204.34375, 297179.8125, 297155.34375, 297130.125, 
#297106.21875, 297081.40625, 297057.0625, 297032.46875, 297007.6875, 296983.03125, 296958.28125, 296933.8125, 
#296909.5, 296884.96875, 296860.5, 296836.03125, 296811.0, 296786.375, 296761.8125, 296737.375, 296712.84375, 
#296688.09375, 296663.65625, 296639.09375, 296614.6875], 'val_root_mean_squared_error': ##[322353.5, 322329.75, 
#322304.78125, 322280.15625, 322256.28125, 322231.34375, 322206.84375, 322183.0625, 322158.1875, 322132.96875, 
#322109.75, 322085.25, 322060.3125, 322036.5625, 322012.375, 321987.75, 321962.84375, 321938.0625, 321914.375, 
#321889.625, 321865.28125, 321839.84375, 321816.03125, 321791.65625, 321768.03125, 321743.0625, 321718.9375, 321694.625,
#321669.84375, 321645.75, 321621.5625, 321597.28125, 321572.46875, 321549.28125, 321523.25, 321499.71875, 321474.71875, 
#321450.34375, 321426.09375, 321400.59375, 321376.40625, 321352.84375, 321327.875, 321304.4375, 321279.1875, 
#321255.3125, 321230.53125, 321205.90625, 321181.5625, 321156.5625, 321133.0, 321108.125, 321084.0, 321059.40625, 
#321034.96875, 321010.6875, 320987.125, 320961.46875, 320938.125, 320912.96875, 320888.21875, 320864.28125, 320840.0625, 
#320816.0625, 320791.0, 320766.96875, 320741.5, 320718.3125, 320693.125, 320668.96875, 320644.125, 320619.78125, 
#320595.9375, 320571.59375, 320546.65625, 320523.0, 320498.5625, 320473.96875, 320449.1875, 320425.0625, 320399.6875, 
#320376.5, 320351.875, 320327.5, 320302.21875, 320277.53125, 320253.90625, 320229.25, 320205.375, 320181.15625, 
#320156.09375, 320131.4375, 320107.15625, 320082.09375, 320059.09375, 320033.96875, 320009.46875, 319985.5, 
#319961.40625, 319935.9375]}

# Notice that now we get back the values for our val_loss and our val_root_squared_mean_error parameters.

# So now we have the training and validation which has been outputted during the training process.

# Recall the use of the validation is for us to see how well our model performs on data it has never seen before.

# So let's go ahead and do some plotting.

# This is how we will formulate our plots to be deployed

# plt.plot(history.history##["loss"])
# plt.plot(history.history##["val_loss"])
# plt.title("model loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(##["train", "val_loss"])
# plt.show()

# Notice that our model loss plot now shows our loss and our val_loss.

# Also notice that we see that our model does better on our validation data, it actually has lower loss values
#for the validation data.

# We could repeat the same process for the root_mean_squared_error

# So next we will plot our rmse data.

# This is how we will formulate our rmse plot to be deployed.

# plt.plot(history.history##["root_mean_squared_error"])
# plt.plot(history.history##["val_root_mean_squared_error"])
# plt.title("model performance")
# plt.ylabel("rmse")
# plt.xlabel("epoch")
# plt.legend(##["train", "val_rmse"])
# plt.show()

# Notice that model performance plot now returns our root_mean_squared_error and our 
#val_root_mean_squared_error

# Also notice that this too has similar effects and our model seems to do better on our validation data.

# Now we will go back to our model evaluate to evaluate our model just by specifying this new formula
# We could actually put x_val and y_val, so we're not evaluating anymore on our training data, but on 
#our validation data directly.

# This is the formula we will run in our data preparation file
# model.evaluate(x_val,y_val) # Output
# 4/4 ##[==============================] - 0s 0s/step - loss: 296622.2188 - root_mean_squared_error: 319936.1562

# Notice we only got back the the data for validation data.

# Next we test our model data that it has never seen by running this formula
# model.evaluate(x_test,y_test) # Output
# 4/4 ##[==============================] - 0s 0s/step - loss: 296619.4688 - root_mean_squared_error: 319936.9062

# Notice we only got back the the data for test data.


# Now we will go our the process of testing our model.

# So here we are not just evaluating our model, or doing some validation, but we are passing in data and then
#we are allowing our model to predict the car price for us.

# This is how we would formulate that process.

# We have x_test, which we built already, 

# print(x_test) 

# Then we will get the shape of x_test

# print(x_test.shape)

# Now that we've trained our model, we will use model.predict(x_test). Since our model is trained,
#we don't need to use y_test when we run our prediction. With this, our model should predict the car price.

# #model.predict(x_test)

# Output

#4/4 #[==============================] - 0s 5ms/step
#[[ 8874.347 ]
 #[ 9422.781 ]
 #[ 8188.4814]
 #[ 8913.584 ]
 #[ 8325.188 ]
 #[ 9140.353 ]
 #[ 8271.814 ]
 #[ 8663.5625]
 #[ 9198.392 ]
 #[ 9163.207 ]
 #[10318.327 ]
 #[ 8336.909 ]
 #[ 8343.364 ]
 #[ 8093.965 ]
 #[ 9267.688 ]
 #[ 9359.162 ]
 #[ 8559.368 ]
 #[ 9848.764 ]
 #[ 7966.786 ]
 #[ 8975.543 ]
 #[ 9093.974 ]
 #[ 8802.347 ]
 #[ 7692.7495]
 #[ 8457.856 ]
 #[ 9042.19  ]
 #[ 9476.393 ]
 #[ 8001.221 ]
 #[ 8253.047 ]
 #[ 8637.303 ]
 #[ 9021.049 ]
 #[ 9653.109 ]
 #[ 8203.192 ]
 #[ 9350.01  ]
 #[ 8686.431 ]
 #[ 7651.786 ]
 #[ 9591.263 ]
 #[ 8311.876 ]
 #[ 9317.233 ]
 #[10721.63  ]
 #[ 9110.635 ]
 #[ 9025.545 ]
 #[ 8534.828 ]
 #[ 8164.782 ]
 #[ 8676.119 ]
 #[ 9820.723 ]
 #[ 8526.827 ]
 #[ 9843.148 ]
 #[ 9735.471 ]
 #[ 8395.107 ]
 #[ 7960.9644]
 #[ 9282.564 ]
 #[ 8350.006 ]
 #[ 8869.553 ]
 #[ 9899.224 ]
 #[ 7821.357 ]
 #[ 9518.35  ]
 #[ 8640.435 ]
 #[ 8004.9263]
 #[ 9224.015 ]
 #[ 8898.176 ]
 #[ 9554.609 ]
 #[ 9828.524 ]
 #[ 8565.964 ]
 #[10095.977 ]
 #[ 7737.019 ]
 #[ 8384.748 ]
 #[ 8469.9   ]
 #[ 8846.946 ]
 #[ 9623.199 ]
 #[ 9157.873 ]
 #[ 9199.62  ]
 #[ 8636.916 ]
 #[ 8647.863 ]
 #[ 9409.283 ]
 #[ 7832.562 ]
 #[ 8406.515 ]
 #[ 7315.679 ]
 #[ 9326.095 ]
 #[ 9769.293 ]
 #[ 8726.018 ]
 #[ 7958.3247]
 #[ 8283.684 ]
 #[ 9861.079 ]
 #[ 9421.347 ]
 #[ 7714.709 ]
 #[ 8864.041 ]
 #[ 7946.9307]
 #[ 8359.888 ]
 #[ 9908.314 ]
 #[ 8789.873 ]
 #[ 9189.591 ]
 #[ 7874.1914]
 #[ 9084.414 ]
 #[ 8587.065 ]
 #[ 8160.6914]
 #[ 8554.874 ]
 #[ 8749.595 ]
 #[10193.953 ]
 #[ 8854.63  ]
 #[10502.632 ]]

# Note that passing in x_test will return a lot of predictions.
# That is because we have a batch size of (100) for our x_test. That's a hundred rows of predictions.

# We can show that is the case by running model.predict(x_test).shape
# Output (100, 1)
# This shape specifies a hundred rows and one column.

# So to get a single prediction, we can index into our x_test.
# Which will look something like this model.predict(x_test[0])
# Output
# Note: Prediction changes slightly with every computation

# Also note that we can not use model.predict(x_test[0]).shape to get the shape of our prediction output,
#so we have to use this formula instead model.predict(tf.expand_dims(x_test[0], axis = 0)).shape.

# Now we will compare this x_test prediction to the price that the data has given us.
# Since we know the actual car price, we can compare it to the predicted car price.

# So let's look at y_test 0 and check out the value.
# We will do this running y_test[0] in our data preparation file.

# Output tf.Tensor([191566.], shape=(1,), dtype=float64)

# Notice that we have a value of 191,566

# Also notice that our x_test prediction was a little under 9,000.

# So clearly our model is performing very poorly.

