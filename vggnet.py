# In this section we will ve going over vggnet models.

# Very deep convolutional networks for large-scale image recognition.

# VGG actually stands for visual Geometry Group.

# In this session we are going to discuss different methods which the authors of the VGG paper used to drop the top
#one validation error rate from 38.1 to 23.7, where 38.1 was achieved by the breakthrough conv net model which is 
#known as the alexnet model.

# Now in the previous session in which we treated the alexnet model and we saw the power of working with conv nets
#and solving recognition tasks.

# One thing we could notice very clearly from this model is that it's quite shallow and so the authors go even
#deeper with the VGG model in the documentation paper.

# In the documentation the authors investigate the effect of the conv net of the convolutional network depth on
#its accuracy.

# So unlike the alexnet, where the depth is relatively small and is actually a shallow network, the VGG uses
#a deeper conv convolutional neural network and make use of smaller convolutional filters. 

# Now recall with alexnet, from the very first layer we already had 11 by 11 filters and we argue that this
#helped in capturing large spatial dependencies.

# Now we'll explain how it's possible for us to make use of the smaller more economical convolutional filters
#while still capturing large spatial dependencies like the bigger 5 by 5 and 11 by 11 filters will do.

# Now we'll see why it's better to work with smaller filters compared to working with larger filters 

# Let's consider the following examples.

# We'll start with the 5 by 5 filter.

# We'll have a kernel size of 5 and an input size of 10 and no padding, a dialation of 1 and a stride of 1.

# This will get us an output of 6 by 6.

# The next example we'll have all the same dimensions, the only difference will be our filter, which will be a 3 by 3.

# This will get us an output of 8 by 8.

# With the 5 by 5 we would have a total of 25 parameters ( 5 times 5 ).

# With the two 3 by 3 filters, we would have a total of 18 parameters because each 3 by 3 filter alone will have
#9 parameters ( 3 times 3 ), and when combuined we would have 18 total parameters. This would give us two output 
#layers of 8 by 8

# Because of this, it would be better to use the two smaller filters as opposed to using the single larger filter.

# We have a higher degree of learning capabilities with the outputs produced by the two 3 by 3 filters.

# The single 5 by 5 filter will have a single output of 6 by 6, where as the two 3 by 3 filters will have two 
#outputs of 8 by 8, thus giving us a higher learning capability percentage, allowing us to capture much more
#complex information from the inputs.

# Also, even though there are two filters of 3 by 3, it would still be cheaper, or cost efficient to run these
#two filters as opposed to the 5 by 5. This is because there are fewer parameters to work through. One 3 by 3
#filter would have 9 parameters, and combined they would have 18 parameters. This is still less that the 25 
#parameters of the 5 by 5 filter.

# So overall we see that it is better to use the conv layers with smaller kernel/filter sizes.
