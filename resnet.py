# In this section we will be going over the RESNET.

# This model was first introduced in a paper intitled - Deep Residual Learning for Image Recognition

# Seven years later and this model is still greatly used.

# The high performance that comes from working with the resnet model comes from the fact that the ResNet model 
#relies on a residual block, that looks like this:

#       X | ------------
# [ weight layer ]      |
#  f(x)   | relu        |
# [ weight layer ]      |    X
#         |             | Identity
#  f(x)   + <-----------|
#         | relu       

# This permits us to get even better error rates as compared to the VGG and Google LeNet Models.

# In this section we are going to focus on understanding how this residual block works and how the ResNet Model
#is constructed based off this model.

# Recall that with the AlexNet, we had fewer number of layers.

# So we started out with AlexNet and its fewer number of layers, and then we moved to VGG where we looked at the 
#VGG 16 Version and the VGG 19 Version.

# We expect that if we keep increasing the number of layers then the error rates should be dropping, but what 
#happens is actually the opposite.

# So let's say for example that we have a 20 layer network and a 56 layer network. We would see that the 20 layer 
#model has a lower error rate than the 56 layer network.

# This same phenomenon is witnessed with a test set.

# In the test tool we would see the 20 layer network outperforming the 56 layer network.

# And so it's clear that just blindly stacking up Conv layers wouldn't help in dropping the training or test
#errors even though they are more expensive.

# This is why the ResNet model introduces this residual learning which is based off the residual block that we
# #just went over.

# Now notice that inside the residual block, the weight layers are simply convolutional layers.

# And so now, unlike before where we would just stack the WL(weight layer) on WL, as seen in the model we just 
#went over, now what we'll do is we'll create a connection between the input and the output.

# So we create the connection,and then there's some addition.

# So we get an output, and we add it with the other output to produce the new output.

#       X | ------------
# [ weight layer ]      |
#  f(x)   | relu        |
# [ weight layer ]      |    X
#         |             | Identity
#  f(x)   + <-----------|
#         | relu 

# So let's suppose that the input is X, and then what goes on between the WL is f of X ( f(x) ).

# And then between the WL's and the new output is also f of x ( f(x) ).

# Now our output can be given as h of x ( h(x) ), which simply equals f of x ( f(x) ) plus X.

# That's the input plus its output. H(x) = X + f(x), which now produces the new output H of x ( H(x) ).

# Now let's get a better undersatnding as to why we need the residual block.

# We need to first understand why models which are based on just stacking up conv layers, like the VGG, 
#actually underfit even when we increase the number of layers.

# The reason for that is exploding and vanishing gradients.

# Let's explain what it means for gradients to vanish.

# Recall that in the gradient descent process we have a weight.

# This weight is updated in such a way that we take its previous value minus the learning rate, times the
#partial derivative of the loss, with respect to that given weight.

# That equation would look like this.

# W = Wpv - LR(2loss / 2W)

# Now during the training process, in order to compute this partial derivative very efficiently is we would 
#use a method called back propagation.

# The way that back propagation works is that we have a model, let's say for example we call it model M.

# Then we will say that model M has an input and an output.

# That would like something like this.

# input -> [ M ] -> output

# Now let's say we name our outputs.

# The names will be y and y cap.

# We'll say that y is what the model expected, and y cap is what was predicted.

# It's the difference between the two outputs that produces the loss, and we're finding the partial derivative
#of this difference with respect to every weight which makes up the model.

# Then if we were to split the model up into different layers, we would note that each layer is composed of
#several weights.

# Now each split layer would have its own weights, but one point to note is that during the back propagation
#process, to obtain the partial derivative of the loss with respect to the weights, we make use of the partial
#derivative of the loss with respect to the weights which come after the layer we're looking at.

# So let's say that this is our new split model

#                                  input -> [ | | | M | | | ] -> output
#                                                |     |-|-|
# To obtain the partial derivatives here --------|     |
#                                                      |
# We make use of the weights that come after it -------|

# The problem that we have is because we're multipying the previous layers to get to the current layer, it 
#means that if while getting the partial derivative we obtain a value very close to 0, it is going to affect
#the partial derivative in the current layer in a sense that it will also be a very small value.

# And if the partial derivatives are too small then we will not get a change in the weight, because we have 
#the new weight we're trying to get being equal to or very close to the previous weight which will result
#in little or no changes.

# That is why even though we keep increasing the number layers, we cannot achieve a better performance due 
#to this vanishing gradient problem.

# This is because the model is now finding it difficult to update its weights such that the training error 
#can be decreased since the gradients are vanishing.

# So now we've seen that making our network deeper or increasing the number of layers makes it difficult
#to propagate information from one far end to the other end.

# And so what the authors suppose is that if the added layers can be constructed as identity mappings, a
#deeper model should have a training error no greater than its shallower counterpart.

# This means that if we have a swallow model and a deep model, we can construct the deeper model in a way
#that it is identical to the shallower model.

# Let's look at the example models below

# _____________________
#|                     |
#|                     | Shallow
#|_____________________|

# ______________________ _ _ _ _ __
#|                     |           |
#|                     | Identity  | Deep
#|_____________________|_ _ _ _ _ _|

# Basically, the two areas of the shallow and the deep models are identical, and the extended piece of the
#deeper model will be classified as the identity function, or a group of identity functions that get
#stacked together.

# This way the training error of the deeper model shouldn't be greater than the training error of the 
#shallower model.

# And so this means that if we want to pass information from point to point (see lines 169 and 177 )
#in our residual block, this path permits us to copy the input to the output.

# And obviously the space inbetween the path is the Identity Function. 

#   Residual Block
#         .
#       X | ------------
# [ weight layer ]      |
#  f(x)   | relu        |
# [ weight layer ]      |    X
#         |             | Identity
#  f(x)   + <-----------|
#         | relu  
#         .

# So let's say after passing through 20 layers we get to a point where the values are almost zero, such that 
#when the information passes it will also be practically zero.

# Then there would be a path which at least restores the exact same input we have. 

# Let's look at the diagram below for example of this.

# Input ||||||||||||||||||||() -> [] -> [] (+)
#                              |____________|

# And so this means that just as the author of the papers supposed in the example on line 148 -155,
#if we make our model or neural network deeper by adding the residual block, then there will be no
#increase in the error rate, and in practice this instead leads to a decrease in the error rate, 
#which is exactly what we want.

# And one other arguement which accounts for the fact that the residual blocks help in improving the 
#performance of the model is the fact that since we have several paths, the residual model now look 
#like a combination of several shallow models.

# So it looks like we're combining different shallow models, which when combined produce what we call
#an ensemble of shallow models, which help in making the overall model much more performant as
#compared to when we just have a single path.

# To better understand how the one by one convolutions work, let's look at this example.

# In this example we will have a 10 by 10 input size and output size.

# _______________________________________
#|___|___|___|___|___|___|___|___|___|___| Input
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|

# _______________________________________
#|___|___|___|___|___|___|___|___|___|___| Output
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|
#|___|___|___|___|___|___|___|___|___|___|

# We also have one weight.

# This one weight moves through each and every pixel value.

# So as our weight moves through our 10 by 10 input grid, we would notice that the input is the same shape 
#as the output.

# Another difference with the VGG and the other previous convnets is that instead of making use of the 
#max pool, what we do is we use a three by three convolution layer and we use strides.
 
# So we specify the stride number of two, for example, and this permits us to down sample the feature maps.

# Let's take a quick look at batch mormalization.

# Batch Normalization is the technique for accelerating deep neural network training by reducing internal
#covariate shift.

# To better understand batch normalization we'll start by explaining the notion of covariate shift, to
#better understand the notion of covariate shift let's suppose that we're trying to build a model which
#classifies whether an input image is of a car or not a car. 

# If we're building this kind of system and then we start by creating batches of toy cars for example,
#and we pass it through the model and model learns how to see this and know that it's a car and see 
#some other image and knows that that's not a car, then later on when we take a car from another
#distribution source, and we pass it into our system, it becomes difficult for the weights of our 
#model to adapt to this change in distribution, though the inputs are all cars.

# This is known as covariate shift,and this is why most times before passing the image into the model, 
#we normalize it.

# Let's suppose we have input x, we generally carry out some normalization in order to account for this
#covariate shift.

# So now after normalization what we're gonna have is that all those images from the first distribution or 
#the other distribution, have now been normalized to reduce the effect of the shift.

# And so now we could have a function that separates the cars from the non cars, and with much more ease.

# Now with that said, what if this kind of covariate shift instead happens in the hidden layers. That's 
#those layers which make up the core of our models. 

# So let's suppose that we have some convnets stacked with the activation function, and then we have the
#weights. That's those parameters that are part of the layer, now coming from different distributions.  

# Then in this case we have an internal covariate shift, and to remedy this situation we now make use 
#of the batch normalization.

# And the algorithm for the batch normalization is described in the paper we have been going over.

# So we have a mini batch and we obtain its mean. This means we try to obtain the average value of the 
#different weights.

# Then we also obtain the standard deviation, which is sigma, and the variance, which is sigma squared.

# So basically we obtain the mean and we obtain the variance, and it's this that we make use of.

# Now we will normalize our data.

# So now we take every weight, we subtract by the mean, and then we divide by the standard deviation,
#and then we add a small epsilon to avoid having a very small number or zero as the denominator.

# That's all, this is how the bath normalization process goes on.

# It should be noted that there are other normalization techniques, like the layer and group normalization,
#which are kind of similar to this process, but different in a sense that with a batch normalization the 
#mean is calculated over a given mini batch.

# Now after getting the new value x shuffle, what we now do is we multiply it by gamma and add beta.

# Now the gamma and beta are actually trainable parameters, so when working with batch normalization in
#say TensorFlow or PyTorch, we'll notice that the batch norm layer will always have its parameters.

# In some cases the role of the gamma and the beta are to scale and shift, and the parameters are 
#learned along with the original model parameters, and restore the representation power of the network.

# So when we set gamma to be the square root of the variance, and beta to be the expectance or the mean of
#x, then we could recover the original activations, if that were the optimal thing to do.

# Essentially what they're saying is if it's instead optimal for us not to use the batch normalization,
#then we could adapt the value of gamma and beta in such a way that we get the original value extra (Xi)

# The way this can be done is quite simple. All we need to do is multiply the X by let's say gamma square,
#plus epsilon.

# So we'll have gamma square plus epsilon, and then once we've multiplied that we see that we're left 
#with Xi minus the mean.

# Now when we're left with Xi minus the mean, if beta is equal to mean then we will have Xi minus the 
#mean plus beta, which in this case is the mean.

# We would see that it cancels out and leaves us with Xi, which is the original value on X.

# Then there's also an initialization that's the model, or the network, is trained from scratch.

# Stochastic gradient descent is used with a mini batch size of 256. 

# The learning rate starts from 0.1 and is divided by 10 when the error plateaus.

# So basically when we get to the point where the arrow starts to plateau, then at that point we could
#update the learning rate from 0.1 to 0.01.

# And then if it drops, we can see any plateaus, and then it drops and plateaus again repetatively, we carry 
#out the same computation.
