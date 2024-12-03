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

# Here we will be over the topic of Performance Measurement

# Here we will be measuring how well this model performs.

# And one common function used in regression models like this is the root mean square error.


# What We should note is that not in every case where we would have the performance measurement function, similar to 
#that of the loss, as the two are quite different concepts.

# For making use of the performance measurement, we are able to see if two models have the same performance, or
#have a different performance, and which of them outperforms the other. 

# So if we run performance measurement, we'll be able to see which model performs better.

# Next we will include this metric in our data prepration file where our csv is located.
# First we need to import RootMeanSquaredError from keras.metrics

# Then we add the metrics = RMSE to our algorithm.

# This is the updated algorithm we will be running our data preparation file

# model.compile(optimizer=Adam(learning_rate = 1),
#               loss= MeanAbsoluteError(),
#               metrics= RootMeanSquaredError())

# This is the output once we complile our code with the updated algorithm.

#Epoch 1/100
#32/32 [==============================] - 1s 5ms/step - loss: 308505.8438 - root_mean_squared_error: 333249.5312
#Epoch 2/100
#32/32 [==============================] - 0s 2ms/step - loss: 308473.4688 - root_mean_squared_error: 333219.4062
#Epoch 3/100
#32/32 [==============================] - 0s 4ms/step - loss: 308441.5625 - root_mean_squared_error: 333190.3125
#Epoch 4/100
#32/32 [==============================] - 0s 8ms/step - loss: 308409.7188 - root_mean_squared_error: 333161.4375
#Epoch 5/100
#32/32 [==============================] - 0s 6ms/step - loss: 308377.4375 - root_mean_squared_error: 333131.2812
#Epoch 6/100
#32/32 [==============================] - 0s 11ms/step - loss: 308345.5312 - root_mean_squared_error: 333101.3750
#Epoch 7/100
#32/32 [==============================] - 0s 5ms/step - loss: 308313.4062 - root_mean_squared_error: 333070.0938
#Epoch 8/100
#32/32 [==============================] - 0s 4ms/step - loss: 308281.5000 - root_mean_squared_error: 333039.8125
#Epoch 9/100
#32/32 [==============================] - 0s 4ms/step - loss: 308249.4062 - root_mean_squared_error: 333010.4062
#Epoch 10/100
#32/32 [==============================] - 0s 4ms/step - loss: 308217.5625 - root_mean_squared_error: 332980.4375
#Epoch 11/100
#32/32 [==============================] - 0s 4ms/step - loss: 308185.6562 - root_mean_squared_error: 332949.8125
#Epoch 12/100
#32/32 [==============================] - 0s 4ms/step - loss: 308153.4062 - root_mean_squared_error: 332920.1250
#Epoch 13/100
#32/32 [==============================] - 0s 4ms/step - loss: 308121.5000 - root_mean_squared_error: 332889.8125
#Epoch 14/100
#32/32 [==============================] - 0s 4ms/step - loss: 308089.5625 - root_mean_squared_error: 332860.5625
#Epoch 15/100
#32/32 [==============================] - 0s 4ms/step - loss: 308057.5312 - root_mean_squared_error: 332831.0312
#Epoch 16/100
#32/32 [==============================] - 0s 3ms/step - loss: 308025.5938 - root_mean_squared_error: 332802.1250
#Epoch 17/100
#32/32 [==============================] - 0s 3ms/step - loss: 307993.6562 - root_mean_squared_error: 332772.8750
#Epoch 18/100
#32/32 [==============================] - 0s 4ms/step - loss: 307961.5938 - root_mean_squared_error: 332742.7500
#Epoch 19/100
#32/32 [==============================] - 0s 4ms/step - loss: 307929.4688 - root_mean_squared_error: 332712.8438
#Epoch 20/100
#32/32 [==============================] - 0s 3ms/step - loss: 307897.5625 - root_mean_squared_error: 332682.4062
#Epoch 21/100
#32/32 [==============================] - 0s 3ms/step - loss: 307865.4062 - root_mean_squared_error: 332652.7500
#Epoch 22/100
#32/32 [==============================] - 0s 4ms/step - loss: 307833.3438 - root_mean_squared_error: 332623.7188
#Epoch 23/100
#32/32 [==============================] - 0s 3ms/step - loss: 307801.7188 - root_mean_squared_error: 332594.2188
#Epoch 24/100
#32/32 [==============================] - 0s 4ms/step - loss: 307769.4062 - root_mean_squared_error: 332564.5312
#Epoch 25/100
#32/32 [==============================] - 0s 4ms/step - loss: 307737.4688 - root_mean_squared_error: 332534.5000
#Epoch 26/100
#32/32 [==============================] - 0s 3ms/step - loss: 307705.3750 - root_mean_squared_error: 332504.9375
#Epoch 27/100
#32/32 [==============================] - 0s 4ms/step - loss: 307673.6250 - root_mean_squared_error: 332474.8438
#Epoch 28/100
#32/32 [==============================] - 0s 4ms/step - loss: 307641.5625 - root_mean_squared_error: 332444.4688
#Epoch 29/100
#32/32 [==============================] - 0s 7ms/step - loss: 307609.4062 - root_mean_squared_error: 332414.8438
#Epoch 30/100
#32/32 [==============================] - 0s 4ms/step - loss: 307577.4375 - root_mean_squared_error: 332385.1562
#Epoch 31/100
#32/32 [==============================] - 0s 3ms/step - loss: 307545.4062 - root_mean_squared_error: 332355.9688
#Epoch 32/100
#32/32 [==============================] - 0s 3ms/step - loss: 307513.4375 - root_mean_squared_error: 332326.0000
#Epoch 33/100
#32/32 [==============================] - 0s 3ms/step - loss: 307481.5625 - root_mean_squared_error: 332296.5625
#Epoch 34/100
#32/32 [==============================] - 0s 3ms/step - loss: 307449.7500 - root_mean_squared_error: 332266.8750
#Epoch 35/100
#32/32 [==============================] - 0s 3ms/step - loss: 307417.4688 - root_mean_squared_error: 332236.5625
#Epoch 36/100
#32/32 [==============================] - 0s 5ms/step - loss: 307385.5000 - root_mean_squared_error: 332207.3125
#Epoch 37/100
#32/32 [==============================] - 0s 4ms/step - loss: 307353.4062 - root_mean_squared_error: 332177.7188
#Epoch 38/100
#32/32 [==============================] - 0s 4ms/step - loss: 307321.4688 - root_mean_squared_error: 332147.9375
#Epoch 39/100
#32/32 [==============================] - 0s 4ms/step - loss: 307289.3438 - root_mean_squared_error: 332118.5625
#Epoch 40/100
#32/32 [==============================] - 0s 4ms/step - loss: 307257.4375 - root_mean_squared_error: 332088.3125
#Epoch 41/100
#32/32 [==============================] - 0s 3ms/step - loss: 307225.4375 - root_mean_squared_error: 332058.7500
#Epoch 42/100
#32/32 [==============================] - 0s 3ms/step - loss: 307193.2812 - root_mean_squared_error: 332028.7188
#Epoch 43/100
#32/32 [==============================] - 0s 3ms/step - loss: 307161.5938 - root_mean_squared_error: 331999.4375
#Epoch 44/100
#32/32 [==============================] - 0s 3ms/step - loss: 307129.4375 - root_mean_squared_error: 331969.9688
#Epoch 45/100
#32/32 [==============================] - 0s 3ms/step - loss: 307097.5625 - root_mean_squared_error: 331941.2500
#Epoch 46/100
#32/32 [==============================] - 0s 3ms/step - loss: 307065.5938 - root_mean_squared_error: 331911.4062
#Epoch 47/100
#32/32 [==============================] - 0s 4ms/step - loss: 307033.6562 - root_mean_squared_error: 331881.1562
#Epoch 48/100
#32/32 [==============================] - 0s 3ms/step - loss: 307001.4062 - root_mean_squared_error: 331851.9375
#Epoch 49/100
#32/32 [==============================] - 0s 3ms/step - loss: 306969.4688 - root_mean_squared_error: 331823.1250
#Epoch 50/100
#32/32 [==============================] - 0s 3ms/step - loss: 306937.5312 - root_mean_squared_error: 331793.2500
#Epoch 51/100
#32/32 [==============================] - 0s 3ms/step - loss: 306905.5000 - root_mean_squared_error: 331764.3438
#Epoch 52/100
#32/32 [==============================] - 0s 3ms/step - loss: 306873.5000 - root_mean_squared_error: 331735.0000
#Epoch 53/100
#32/32 [==============================] - 0s 3ms/step - loss: 306841.4062 - root_mean_squared_error: 331705.6250
#Epoch 54/100
#32/32 [==============================] - 0s 3ms/step - loss: 306809.3750 - root_mean_squared_error: 331675.5312
#Epoch 55/100
#32/32 [==============================] - 0s 3ms/step - loss: 306777.4688 - root_mean_squared_error: 331646.4062
#Epoch 56/100
#32/32 [==============================] - 0s 3ms/step - loss: 306745.4375 - root_mean_squared_error: 331616.0938
#Epoch 57/100
#32/32 [==============================] - 0s 3ms/step - loss: 306713.7188 - root_mean_squared_error: 331586.4375
#Epoch 58/100
#32/32 [==============================] - 0s 5ms/step - loss: 306681.5625 - root_mean_squared_error: 331557.4062
#Epoch 59/100
#32/32 [==============================] - 0s 6ms/step - loss: 306649.6875 - root_mean_squared_error: 331527.6875
#Epoch 60/100
#32/32 [==============================] - 0s 4ms/step - loss: 306617.5312 - root_mean_squared_error: 331498.3438
#Epoch 61/100
#32/32 [==============================] - 0s 5ms/step - loss: 306585.5938 - root_mean_squared_error: 331468.3750
#Epoch 62/100
#32/32 [==============================] - 0s 5ms/step - loss: 306553.4062 - root_mean_squared_error: 331438.9062
#Epoch 63/100
#32/32 [==============================] - 0s 4ms/step - loss: 306521.4062 - root_mean_squared_error: 331408.5938
#Epoch 64/100
#32/32 [==============================] - 0s 4ms/step - loss: 306489.5000 - root_mean_squared_error: 331379.0625
#Epoch 65/100
#32/32 [==============================] - 0s 4ms/step - loss: 306457.5000 - root_mean_squared_error: 331348.8438
#Epoch 66/100
#32/32 [==============================] - 0s 4ms/step - loss: 306425.7188 - root_mean_squared_error: 331318.5000
#Epoch 67/100
#32/32 [==============================] - 0s 4ms/step - loss: 306393.5312 - root_mean_squared_error: 331289.3438
#Epoch 68/100
#32/32 [==============================] - 0s 5ms/step - loss: 306361.4375 - root_mean_squared_error: 331260.3750
#Epoch 69/100
#32/32 [==============================] - 0s 5ms/step - loss: 306329.3438 - root_mean_squared_error: 331230.7188
#Epoch 70/100
#32/32 [==============================] - 0s 5ms/step - loss: 306297.5938 - root_mean_squared_error: 331202.7812
#Epoch 71/100
#32/32 [==============================] - 0s 6ms/step - loss: 306265.4375 - root_mean_squared_error: 331173.5938
#Epoch 72/100
#32/32 [==============================] - 0s 4ms/step - loss: 306233.4688 - root_mean_squared_error: 331143.8750
#Epoch 73/100
#32/32 [==============================] - 0s 7ms/step - loss: 306201.5312 - root_mean_squared_error: 331114.9062
#Epoch 74/100
#32/32 [==============================] - 0s 8ms/step - loss: 306169.5625 - root_mean_squared_error: 331084.8438
#Epoch 75/100
#32/32 [==============================] - 0s 5ms/step - loss: 306137.4375 - root_mean_squared_error: 331055.2812
#Epoch 76/100
#32/32 [==============================] - 0s 4ms/step - loss: 306105.4688 - root_mean_squared_error: 331025.5312
#Epoch 77/100
#32/32 [==============================] - 0s 4ms/step - loss: 306073.5625 - root_mean_squared_error: 330995.0000
#Epoch 78/100
#32/32 [==============================] - 0s 5ms/step - loss: 306041.5938 - root_mean_squared_error: 330966.0000
#Epoch 79/100
#32/32 [==============================] - 0s 5ms/step - loss: 306009.7188 - root_mean_squared_error: 330936.2188
#Epoch 80/100
#32/32 [==============================] - 0s 5ms/step - loss: 305977.4688 - root_mean_squared_error: 330906.5625
#Epoch 81/100
#32/32 [==============================] - 0s 5ms/step - loss: 305945.6250 - root_mean_squared_error: 330877.8750
#Epoch 82/100
#32/32 [==============================] - 0s 4ms/step - loss: 305913.5312 - root_mean_squared_error: 330848.0938
#Epoch 83/100
#32/32 [==============================] - 0s 5ms/step - loss: 305881.5625 - root_mean_squared_error: 330819.9062
#Epoch 84/100
#32/32 [==============================] - 0s 6ms/step - loss: 305849.4688 - root_mean_squared_error: 330789.5312
#Epoch 85/100
#32/32 [==============================] - 0s 6ms/step - loss: 305817.5312 - root_mean_squared_error: 330760.4062
#Epoch 86/100
#32/32 [==============================] - 0s 4ms/step - loss: 305785.5312 - root_mean_squared_error: 330730.2812
#Epoch 87/100
#32/32 [==============================] - 0s 3ms/step - loss: 305753.4062 - root_mean_squared_error: 330700.1875
#Epoch 88/100
#32/32 [==============================] - 0s 3ms/step - loss: 305721.3125 - root_mean_squared_error: 330670.1875
#Epoch 89/100
#32/32 [==============================] - 0s 3ms/step - loss: 305689.3438 - root_mean_squared_error: 330641.1875
#Epoch 90/100
#32/32 [==============================] - 0s 3ms/step - loss: 305657.9375 - root_mean_squared_error: 330612.7188
#Epoch 91/100
#32/32 [==============================] - 0s 3ms/step - loss: 305625.4062 - root_mean_squared_error: 330583.2188
#Epoch 92/100
#32/32 [==============================] - 0s 3ms/step - loss: 305593.5000 - root_mean_squared_error: 330554.9375
#Epoch 93/100
#32/32 [==============================] - 0s 3ms/step - loss: 305561.3750 - root_mean_squared_error: 330525.8438
#Epoch 94/100
#32/32 [==============================] - 0s 4ms/step - loss: 305529.5000 - root_mean_squared_error: 330496.5312
#Epoch 95/100
#32/32 [==============================] - 0s 3ms/step - loss: 305497.4062 - root_mean_squared_error: 330467.5000
#Epoch 96/100
#32/32 [==============================] - 0s 3ms/step - loss: 305465.4375 - root_mean_squared_error: 330438.0312
#Epoch 97/100
#32/32 [==============================] - 0s 4ms/step - loss: 305433.5938 - root_mean_squared_error: 330408.8125
#Epoch 98/100
#32/32 [==============================] - 0s 4ms/step - loss: 305401.4062 - root_mean_squared_error: 330379.8125
#Epoch 99/100
#32/32 [==============================] - 0s 3ms/step - loss: 305369.4688 - root_mean_squared_error: 330350.1562
#Epoch 100/100
#32/32 [==============================] - 0s 3ms/step - loss: 305337.5312 - root_mean_squared_error: 330320.3750

# Notice that we get back both the Loss and the Root Mean Squared Errors.

# Once the training is done, we'll be able to plot this out too by using this in our plt function
# plt.plot(history.history["root_mean_squared_error"])

# Once our original Model Loss loss plot is opened and closed, our Model Performance root mean squared error plot
#will open.

# Lastly, we'll rerun the history.history and this time we will back both the the loss values and the 
#rmse values. Ouput below
#{'loss': [305305.5625, 305273.53125, 305241.5, 305209.40625, 305177.71875, 305145.5, 305113.4375, 305081.40625, 
#305049.6875, 305017.5, 304985.4375, 304953.46875, 304921.59375, 304889.375, 304857.46875, 304825.4375, 304793.5, 
#304761.5, 304729.5, 304697.75, 304665.5, 304633.625, 304601.46875, 304569.5625, 304537.53125, 304505.5625, 
#304473.53125, 304441.53125, 304409.5, 304377.6875, 304345.46875, 304313.4375, 304281.53125, 304249.3125, 304217.53125,
#304185.4375, 304153.5, 304121.34375, 304089.5625, 304057.4375, 304025.59375, 303993.59375, 303961.59375, 303929.4375, 
#303897.53125, 303865.46875, 303833.34375, 303801.5625, 303769.5, 303737.40625, 303705.4375, 303673.59375, 303641.375, 
#303609.625, 303577.59375, 303545.53125, 303513.40625, 303481.5, 303449.4375, 303417.5, 303385.5625, 303353.375, 
#303321.5, 303289.5625, 303257.53125, 303225.46875, 303193.46875, 303161.40625, 303129.40625, 303097.40625, 
#303065.40625, 303033.71875, 303001.4375, 302969.5, 302937.59375, 302905.75, 302873.375, 302841.34375, 302809.625, 
#302777.375, 302745.46875, 302713.4375, 302681.5, 302649.46875, 302617.625, 302585.625, 302553.40625, 302521.375, 
#302489.59375, 302457.5, 302425.40625, 302393.53125, 302361.46875, 302329.5, 302297.46875, 302265.4375, 302233.3125, 
#302201.71875, 302169.46875, 302137.46875], 'root_mean_squared_error': [330283.3125, 330254.5, 330225.625, 330196.09375, 
#330167.71875, 330138.15625, 330107.625, 330078.0, 330048.71875, 330019.1875, 329988.71875, 329958.5625, 329928.34375, 
#329898.78125, 329869.59375, 329839.65625, 329810.375, 329780.53125, 329750.0, 329719.6875, 329690.375, 329661.78125, 
#329632.8125, 329604.625, 329575.84375, 329544.9375, 329515.125, 329486.625, 329456.625, 329426.375, 329396.34375, 
#329366.1875, 329337.21875, 329308.28125, 329277.90625, 329249.15625, 329218.96875, 329188.5, 329159.4375, 329129.65625, 
#329099.90625, 329070.4375, 329041.53125, 329010.46875, 328979.6875, 328949.59375, 328920.5625, 328892.65625, 
#328862.09375, 328833.15625, 328803.15625, 328772.5, 328742.5625, 328713.34375, 328684.25, 328654.90625, 328625.53125, 
#328595.78125, 328566.6875, 328537.40625, 328507.84375, 328477.96875, 328449.34375, 328419.09375, 328389.625, 
#328361.03125, 328331.59375, 328302.21875, 328273.25, 328243.15625, 328214.1875, 328184.71875, 328154.46875, 328125.25, 
#328094.0625, 328063.78125, 328034.40625, 328004.75, 327976.78125, 327947.3125, 327917.34375, 327887.0, 327857.03125, 
#327827.09375, 327797.71875, 327768.25, 327739.125, 327709.375, 327680.1875, 327650.03125, 327619.9375, 327589.5625, 
#327559.9375, 327530.96875, 327501.71875, 327472.34375, 327443.625, 327414.34375, 327384.9375, 327355.8125]}


# Another method that comes with Tensorflow that we will be looking at is the evalute method.

# This will allow us to see that we've evaluated our model and we have the models loss and the rmse.

# model.evaluate(x,y) # Output 
# 32/32 [==============================] - 0s 3ms/step - loss: 302120.2500 - root_mean_squared_error: 327342.3438

# Notice that we are returned the model's loss and rmse (root mean squared error) values.