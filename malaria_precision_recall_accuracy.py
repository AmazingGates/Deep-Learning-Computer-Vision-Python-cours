import tensorflow as tf # For our Models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Normalization, Input, Layer
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC
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

metrics = metrics = [TruePositives(name="tp"), FalsePositives(name="fp"), TrueNegatives(name="tn"), FalseNegatives(name="fn"),
                     BinaryAccuracy(name="ba"), Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]

lenet_custom_model.compile(optimizer = Adam(learning_rate=0.01),
              loss = BinaryCrossentropy(),
              metrics = metrics)

history = lenet_custom_model.fit(train_dataset, validation_data = val_dataset, epochs = 1, verbose = 1)

test_dataset = test_dataset.batch(1)

test_dataset

lenet_custom_model.evaluate(test_dataset)



# In this section we will be going over Model precisiion, recall and accuracy.

# Here we will be looking at other methods of evaluating our model other than the Binaryaccuracy (located inside the 
#metrics =) which we've seen so far.

# So in this section we'll look at how to compare true positives, false positives, true negatives, false negatives,
#the precision, the recall, the area under the curve, how to come up with confusion metrics, and finally, how to plot
#out a ROC curve, which permits us to select the threshold more efficiently.

# Let's start by looking at other ways we can evaluate our model other the accuracy.

# To better understand why working with other accuracy isn't always a good idea, we have to take into consideration
#the fact that our model on a test set has a 94% accuracy.

# This means that we have six out of one hundred predictions which are actually false.

# Now, what if we get to the hospital and we're told that we don't have malaria when in fact we do have the actual
#disease? 

# With that said, the model predicted uninfected, but we actual have the parasite in our bloodstream.

# This particular situation becomes very dangerous because the patient goes home thinking they don't need any treatment,
#whereas that patient actually has the parasite.

# We see that even with a 94% accuracy, we wouldn't be able to save ourselves from such chaotic model predictions.

# In another example, we have a situation where the actual is unparasitized.

# So actually, we do not have the parasite.

# But the model predicts that we have the parasite.

# In this case, altough we have the wrong prediction, we actually have a less chaotic situation as compared to the
#previous case since we're actually uninfected.

#  We'll consider negative to be represented by U, which stands for uninfected

# And we'll consider positive to be represnted by P, which stands for parasitic.

# Now if we set this, we will find that this first situation, where we actually have the parasite and the model
#predicts unparasitzed, is known as a false negative.

# And this because the model predicts negative, when it's actually not negative.

# So since we have this wrong prediction for negative, we call it a false negative.

# And in the case, where we have the model predicting parasitized, that's positive, when it's actually negative,
#we call this a false positive.

# So we would have an FN for the first scenario, and an FP for the second scenario.

# There are two other scenarios.

# That would be the TN and the TP.

# For the TN, that would be the True Negative.

# And for the TP, that would be the True Positive.

# For TN we have the model predicting negative, and we're actually negative.

# And for the TP, the model is predicting positive, and we are in fact positive.

# Now that we ubderstand the concepts of FN, FP, TN, and TP, we can summarize all of this information in what is 
#known as a confusion matrix.

#                Confusion Matrix
#                 __         __
#                |             |
#                |  TN     FP  |
#   Actual    ---               ---
#                |  FN     TP  | 
#                |__         __| 

# In this confusion matrix we have the number of true negatives, the number of true positives, the number of false
#negatives, and the number of false positives.

# This means that if we have a test set of say, 2,750 different data points, and then we evaluate this with our model,
#we'll be able to get this number of true negatives, get the number of false negatives, get the number of true
#positives, and get the number of false positives, and hence better evaluate this model.

# So if we take this example of where we've evaluated our model on the test set, and then this model A produces
#this confusion matrix, and then TP, which will be model B, produces this confusion matrix.

#                Confusion Matrix
#                 __         __
#                |             |
#                | 1000    700 |
#   Model A   ---               ---
#                |  50    1000 | 
#                |__         __| 


#                Confusion Matrix
#                 __         __
#                |             |
#                | 1000     50 |
#  Model B    ---               ---
#                | 700    1000 | 
#                |__         __| 


# Note: All the numbers are the accuracy of the field that they occupy.

# So, if we had to choose a model between A and B, we'd try to choose the model which minimizes the number of 
#false negatives.

# We are not saying that we shouldn't minimize the number of false positives, because we have to try to minimize
#all the false predictions.

# But with the false negatives, we would be telling a sick peerson that they are not sick, and this is worse than
#telling a healthy person that they are sick.

# And so we'll try to prioritize the number of false negatives, and based on the prioritization, we'd choose
#model A since we have the smaller FN number in that model.

# Now, we should note that depending on the kind of problem that we want to solve, in some cases, we will want 
#to prioritize minimizing the number of false positives over the number of false negatives.

# So us choosing model A over Model B in the first example depends on the problem that we were trying to solve.

# In our case, we were prioritizing the number of false negatives, so model was the correct choice for us.

# Based on what we've seen so far, we're going to introduce several new performance metrics.

# And all of these use these different formulas.

#                  TP
# - Precision = --------
#               TP + FP

#               TP
# - Recall = --------
#            TP + FN

#                 TN + TP
# Accuracy = -----------------
#            TN + TP + FN + FP

#                2 PR
# - F1-Score z --------
#                P + R

#                    TN
# - Specificity = --------
#                 TN + FP

# We have the precision, which is the number of true positives divided by the number of true positives, plus the
#number of false positives.

#The recall, true positives divided by the number of true positives plus false negatives. 

# The accuracy, true negatives plus true positives, divided by number of true negatives, true positives, false
#negatives, and false positives.

# We'll examine these first three for now.

# Now what do we notice?

# We'll notice that in the precision and recall, we have the true positive true positive, and true positive true
#positive.

# What differentiates them is the fact that in the precision we have false positive in the denominator, and in the
#recall, we have false negative in the denominator.

# This means that if the number of false negative is high, that's when we have a constant, for example, a constant
#k divided by a high value.

# So     k
#      -----
#       high

# This will gives us a low output.

# And so if we want to have a low recall, then we need to have a high number of false negatives.

# And if we want to have a low precision, then we need to have a high number of false positives.

# Now in our case, we're trying to minimize the number of false negatives.

# And since we're trying to minimize the number of false negatives, it means that we're trying to maximize the
#recall.

# Since minimzing the denominator, we're maximizing the the overall TP on TP plus FN.

# And so here we're trying to prioritize the recall over the precision.

# Now, if we look at the accuracy, you'll notice that we have TN + TP in the numerator and the denominator.

# And we also have FN + FP in the denominator.

# If we are keen enough, we should see that the accuracy doesn't give any priority to the false negatives
#or false positives, it treats the two the same.

# But as we've seen previously in the real world, in solving real world problems, many times we'll have to prioritize.

# Hence, the accuracy may not always be the best metrics for our problem.

# In our case, we find that using the recall is even better than using the accuracy.

# As with the recall, we get to see whether our model does well at minimizing the number of false negatives.

# Now, we also have the F-1 score, which is two times the precision times the recall, divided by the precision
#plus the recall.

# The specificity, which is the number of true negatives divided by the number true negatives plus the number of
#false positives.

# And then we have this ROC plot, which stands for Receiver Operating Characteristics.

#  TP rates |                    / ROC
#           |                  /
#           |                /
#           |              /
#           |            /
#           |          /
#           |        /
#           |      /
#           |    /
#           |  /
#           |/__________________________________
#                                       FP rates

# The plot is made up of the true positive rate, and the false positive rate.

# The true positive rate is the number of true positives divided by number of true positives plus number of false
#negatives, which happens to be the recall.

# And then the false positive rate is the number of fasle positives divided by number of fasle positives plus number
#of true negatives, which if we look carefully, we'll see that is equal to One minus the specificity.

# Before getting to understanding the ROC plot, let's recall the two models we had, model A and Model B.

#                Confusion Matrix
#                 __         __
#                |             |
#                | 1000    700 |
#   Model A   ---               ---
#                |  50    1000 | 
#                |__         __| 


#                Confusion Matrix
#                 __         __
#                |             |
#                | 1000     50 |
#  Model B    ---               ---
#                | 700    1000 | 
#                |__         __| 

# Remember that Model A has the small FN number.

# Now, if we pick out just the Model B, and then we are interested in reducing the number of FN in this model, which
#is 700 right now.
 
# Then one solution could be modifying the threshold. 

# So if we had a threshold of 0.5, meaning that below 0.5 we consider negative, above 0.5 we consider positive.

# Above would be considered parasitized and below would be considered uninfected.

#       1.0 = Parasitized
#      |
#------ 0.5
#      |
#       0.0 = Uninfected

# Then what we could do is reduce the threshold.

# So if we reduce the threshold to lets say a value of 0.2

#       1.0 = Parasitized
#      |
#------ 0.2
#      |
#       0.0 = Uninfected

# We would see that for most of the predictions, our model is going to say that this is a parastized output, since
#our threshold has been reduced.

# This means that if we have a model prediction of say 0.3, which would have initially been uninfected is now 
#parasitized since our threshold has been lowered

# And so this makes it more difficult for our model to have False Negatives, since our model now has this tendency
#of predicting that a given input image is parasitized.

# That said, we now need to look for a way that we could automate this process.

# That is we want to be able to choose this threshold correctly.

# Because if we say let's take a threshold of say 0.01, for example. 

# This means that anytime our model predicts less than 0.01, it's uninfected, and anything greater than 0.01 is 
#parasitized, then this will be very dangerous for the overall model performance, because now most of our predictions 
#for the input image would be parasitized.

# And so our aim here is to pick the threshold 0.2, such that the number of true positives and true negatives that 
#we have in our Model B don't get reduced.

# Now, the way we could look at this is now by using the ROC plot.

# With this ROC plot, what we actually have here is the different True Positive rates and False Positives rates 
#we'd have at a given threshold.

# So this means that a point picked on our ROC plot is just a given threshold.

# Now let's say that the threshold 0.5 is our point picked.

# We could also pick another threshold.

# Let's say that this one is 0.2.

# And we'll have one last point picked of 0.1.

#  TP rates |                    / ROC
#           |                  /
#           |                /-----0.5
#           |              /
#           |            /
#           |          /-----0.2
#           |        /
#           |      /
#           |    /-----0.1
#           |  /
#           |/__________________________________
#                                       FP rates

# Now we could have another model with a different ROC plot, one with this kind of ROC plot, but note
#that overall our aim is to ensure that the False Positive rates is minimized, and the True Positive rates is
#maximized.

#  TP rates |  __________________ / ROC
#           | |                 /
#           | |               /-----0.5
#           | |             /
#           | |           /
#           | |         /-----0.2
#           | |       /
#           | |     /
#           | |   /-----0.1
#           | | /
#           |/|__________________________________
#                                       FP rates

# So with our new ROC plot which we have drawn, then we'd be able to pick out the threshold of 0.X, for example.

#  TP rates |  __________________ / ROC
#           | |\                 /
#           | |  \             /
#           | |    \        /
#           | |      \    /
#           | |        \/
#           | |       /  \__ 0.X
#           | |     /
#           | |   /
#           | | /
#           |/|__________________________________
#                                       FP rates

# We'll be able to pick this threshold because for this threshold, at this point, the True Positive rate is at its
#highest value, and that is 1.

# And then the False Positive rate is at its lowest possible value, that is 0.

# So we have a 1 then a 0

#  TP rates |  __________________ / ROC
#         1 | |\                 /
#           | |  \             /
#           | |    \        /
#           | |      \    /
#           | |        \/
#           | |       /  \__ 0.X
#           | |     /
#           | |   /
#           | | /
#         0 |/|__________________________________
#                                       FP rates

# Now this value of X could five, or four, or whatever.

# So we have zero point, whatever value will lead us to 1.

# Nonetheless, many times we wouldn't have these kind of plots.

# So we will do it plus, which will look like this.

#  TP rates |  ___________________/ ROC
#           | |                 /
#           | |               /
#           | |             /
#           | |           /
#           | |         /
#           | |       /
#           | |     /
#           | |   /
#           | | /
#           |/|__________________________________
#                                       FP rates

# With a plot like this, the aim is to ask ourselves the right question.

# If we want to make sure that our recall is always maximized, which is the case for us, we will try to ensure that
#that we pick out the points at the top.

# So if we want to maximize our recall value, is it normal that we would pick out threshold values which would take 
#us around the region near the top, because it's around this region that our recall is maximized?

# That is the question.

# But the problem with picking a point around this region, is that when we pick a point around this region, it's
#in a region where the False Positive is very high.

# So we need to find a balance between this FP rate and TP rate.

# So it will be much more logical to pick a point near the curve of our plot line.

# So we would pick around that region instead.

# We can see that around that region,, if we picked a point it would correspond to a smaller FP rate, while our
#TP rate is maximized, though it isn't the best recall we coud have.

# But trying to focus on getting that recall of one will lead us into trouble since getting a recall of one in 
#this case will increase our FP rate.

# And if we're dealing with a problem where we're trying to maximize the precision, then in that case, we want to 
#ensure that this FP rate is minimized.

# And so in these kind of problems, we'll want to pick a point near the curve of the plot line.

# So you see, we would want to pick this kind of value, since at least our FP rate is minimized.

# But then if we want to pick a point near the bottom of the plot line, we would have a FP rate of zero, but doing
# #this will get us into trouble because that would give us a TP rate that is very small.

# And so that is why we need to have that balance.

# Now, if we have a problem where it doesn't really matter, which means we aren't trying to prioritize the precision
# #or the recall and working with acurracy is just fine, then we could pick out the points near the curve of our
#plot line.

# One great thing about this tool, the ROC plot, we're able to pick out the points and then automatically get the 
#threshold we need to work with.

# And so when doing predictions, we may not use the 0.5, but we're going to use a certain threshold, which will
#suit the objectivs we set initially.

# Now we'll move to the area under the curve.

# For the area under the curve, we generally use this when we are comparing two models.

# Let's call our first model Alpha.

#  TP rates |          (          / ROC
#           |         (         /
#           |        (        /
#           |       (       /
#           |      (      /
#           |     (_____/__Alpha
#           |    (    /
#           |   (   / 
#           |  (  /   
#           | ( /      
#           |(/__________________________________
#                                       FP rates


# Then we will have a model beta.

#  TP rates | __________(_________ / ROC
#           ||         (         /
#           ||__Beta  (        /
#           ||       (       /
#           ||      (      /
#           ||     (_____/__Alpha
#           ||    (    /
#           ||   (   / 
#           ||  (  /   
#           || ( /      
#           ||(/__________________________________
#                                       FP rates


# It's clear that model Beta is better because it gives us better options.

# Now we'll see that if we find ourselves there, we get a better True Positive rate, false positive rate balance,as
#compared to when we find ourselves at a position within model Alpha.

# So if we are comparing these two models we can make use of the area under the curve by calculating this area covered
#under the curve of model Alpha.

# So let's go ahead and bound this and then for alpha, we have this area under the curve, popularlly known as AuC,
#with a lowercase u.

#  TP rates |
#           | ______________________
#           ||          (\         / ROC
#           ||___Beta  (  \      /
#           ||        (    \   /
#           ||       (  \   \/
#           ||      (    \ / \
#           ||     (_____/\__Alpha
#           ||    (  \ /   \   \
#           ||   (   /\     \   \
#           ||  ( \/   \     \   \
#           || ( / \    \     \   \
#           ||(/____\____\_____\___\________________
#                                       FP rates

# And then for Beta we will have this area under the curve, which covers the area under the model Beta, plus the
#area under the model A.

#  TP rates |
#           | ______________________
#           ||      \   (\         / ROC
#           ||___Beta\ (  \      /
#           || \      (    \   /
#           ||  \    (  \   \/
#           ||   \  (    \ / \
#           ||\    (_____/\__Alpha
#           || \  (  \ /   \   \
#           ||   (   /\     \   \
#           ||  ( \/   \     \   \
#           || ( / \    \     \   \
#           ||(/____\____\_____\___\________________
#                                       FP rates

# And so in general, if we have two models, and then we want to compare them, then we could use the area under the 
#curve, since it shows us how much freedom we have in playing around with the thresholds.

# Now we will get back to the code and see how we are going to implement these new metrics we've just talked about.

# We will do this by importing them to our program from the keras.metrics.

# After the Accuracy, we will import the FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, 
#Recall, and finally the AUC.

# And since we're dealing with a Binary classification problem we'll add the BinaryAcurrcay.

# Next we will define the metrics list.

# Inside our metrics list, we will pass all of the inputs we just mentioned, including the AuC.

# We will do this by setting metrics equal to metrics on line 128

# And above our lenet_model.compile we will define metrics as such (see line 125)

# metrics = [TruePositives(name="tp"), FalsePositives(name="fp"), TrueNegatives(name="tn"), FalseNegatives(name="fn"),
#BinaryAccuracy(name="ba"), Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]

# Now let's compile and fit our model.

# We should see that as we are training we will have all of the items from our metrics list display and accounted
#for.

# Now we will go ahead and evaluate our model by using the lenet_custom_model.evaulate method. (See line 138)

# We will be using test data to evaluate our model and we can setup the process like this.

# Note: This will go above the evaluate method. (see lines 134 and 136)

# test_dataset = test_dataset.batch(1)

# test_dataset

# Now we should successfully have evaluated our model.