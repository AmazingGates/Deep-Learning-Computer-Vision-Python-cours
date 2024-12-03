# In this section we will be going over the Mobilenet

# We will be treating the mobilenet architecture.

# This was first developed by google researchers in 2017, which gave us the mobile net version 1 and later 
#updated to mobile net version 2 which was developed in 2019.

# After this there was a mobile version net 3, but we are just going to focus on the mobile net version 2.

# We will be going over a paper titled "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

# The mobile nets have been built for enviornments with low compute resources like the mobile and edge devices.

# In this section we are going to focus on what permits this model (mobile net V2) to perform quite well in 
#terms of speed while producing high quality results.

# There are two major techniques which make the mobile net version 2 very powerful, or which permits us to 
#work at higher speeds while still maintaining reasonable quality results.

# Now these two are the depth wise separable convolutions, and the inverted residual bottleneck.

# We'll start by explaining what a depth separable convolution is. 

# A Depth Separable Convolution is simply a combination of a depth wise convolution and a point wise 
#convolution.

# Now this point wise convolution is not different from a normal convolution layer, but with a kernel size
#of one.

# So a one by one convolution would be a point wise convolution.

# Basically, the depth wise convolution is the two put together in sequence or sequentially.

# If we want an even deeper explaination and understanding of the depth wise convolution we can read the
#documentation.

# The next thing we will look at is the Inverted Residual Block.

# First of all, it's called inverted in comparison to the residual block.

# The Residual block has relatively large channel input, then it gets smaller in the middle,
#and finally it gets large again in the output.

# This means that we have some input, then we have residual blocks and then we have an output.

# We also obviously have the links that connect from the input to the output.

# With this we would notice that the data, or input, starts out big, then it goes to small, and then
#goes back to big.

# This is the residual block.

# But in the Inverted Residual Block what we do is we start by passing in a relatively small number of channels.

# A small number of channels in the input, then the middle block, or blocks get bigger, and finally the 
#output is a small number of channels, hence the term "inverted residual block."

# Now in addition to the fact that we're using depth wise convolution, instead of the normal convolutions,
#the fact that we have a relatively lower dimensions data going into the smaller blocks, and coming out
#of the smaller blocks (as inputs and output), means that we could transport very low dimensional data 
#throughout our mobile network. 

# So we have low dimensional data going in, low dimensional data coming out, and then inside we have an expansion.

# This expansion layer permits us to capture as much information as possible from our input features.

# One thing to also notice is the fact that we would be using the relu 6.

# The relu 6 is different from the usual relu in the sense that with a relu, we have for all x less than zero, 
#the value is zero.

# For all x greater than zero is x.

# So let's say we have y equals x, for example.

# With the relu 6, from the value 6, we actually clip the output.

# What we would have is all the values, it would remain x, but once we get to 6 it gets clipped.

# So for all values greater than 6, the value remains at 6.

# So that's the relu 6.

# And then, one of the most important points is that because we're carrying out a projection from high
#dimensional data to low dimensional data, the relu non-linearity will generally cause us to lose too much
#information.

# And because of that, there is would be no relu activation in our final layer of our network.

# Apart from classification the mobilenet v2 has been used in other tasks like object detection, semantic
#segmentation and other computer vision tasks where we have low compute resource.

