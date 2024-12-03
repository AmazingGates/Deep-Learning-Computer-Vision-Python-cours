# In this sectin we will be going over the EfficientNet.

# In the efficient net papers the authors proposed a more controlled manner of designing convolutional 
#neural networks in such a way that it suits our demands in accuracy and speed.

# If we were looking at example plots, we would be able to see that we could choose suitable parameters
#in such a way that we could modify or increase our accuracy while taking note of how the modification
#affects our speed.

# With that said, in this section, we'll see how Mingxing Tan and Quoc V.Le built the system for 
#automatically scaling our convolutional neural networks much more efficiently.

# Convnets are commonly developed at a fixed resource budget, and then scaled up for better accuracy if
#more resources are available.

# So with the case of the ResNet, we had resnet 34. Then we had resnet 50, and resnet 152.

# Depending on the kind of setting, we are going to pick the resnet model which permits us to run without
#any problems of latency while maintaining reasonable accuracy.

# So this means that if we are working in a high compute enviornment, then we could afford to work with that.

# Whereas if we are working in a low compute enviorment, then we would have to work with the model with 
#fewer conv layers.

# Now with that said, in the paper, the Authors propose a more systematic study of how the model scaling can
#be done.

# And unlike other methods, we systematically study model scaling and identify that carefully balancing network
#depth, width, and resolution can lead to better performance.

# Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/
#resolution using a simple yet highly effective compound coefficient. 

# Note: When we increase width, that increases the number of channels in our convolutional neural network.

# We'll Notice that as we increase the number of channels, at some point it starts to plateau. 

# And then when we increase the depth at some point, it starts to plateau.

# Then when we also increase the the input size as a resolution, at some point it starts to plateau.

# And so this is why the authors proposed a technique where we could combine all of this in such a way that 
#we get evn better results. 

# This would show us the effect of compund scaling.

# Now we will dive a little bit deeper and look at the compound coefficient, which was mentioned in the beggining.

# In the paper, we propose a new compound scaling method, which uses a compound coefficient to uniformly
#scale network width, depth, and resolution in a principled way.

# Let's take a look at the example equation below that has three formulas.

# The first one we have is depth: d = alpha times phi. 

# Now these phis are user specified coefficients that control how many more resources are available for model 
#scaling.

# So that means that the phi is some sort of scaling coeficient.

# Formulas 2 and three are the beta and the gamma times phi.
 

# 1.   depth: d = alpha * phi
# 2.  width: w = beta * phi
# 3. resolution : r = gamma * phi

# This next formula is designed in such a way that alpha time beta squared times gamma squared is equal to 2.

#       s.t. a * b(squared) * g(squared) = 2

# And in this next formula we are saying that alpha is greater than or equal to 1, beta is always greater than or 
#equal to 1, and gamma is always greater than or equal to 1.
 
#   a >= 1, b >= 1, g >= 1

# So now we are oing to carry out a grid search.

# So we're going to search for the best values for alpha, beta and gamma, and then fix them.

# Obviously they are constants, so we are going to fix the phis and vary them.

# We will do this in such a way that we carry out the scaling in a more systematic manner.

# In order to find out the values for alpha, beta, and gamma, we will fix the value of phi to equal 1.

# After doing this we obtain the new values for alpha,, beta, and gamma.

# a = 1.2
# b = 1.1
# g = 1.5

# 