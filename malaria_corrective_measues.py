# In this section we will be going over Corrective Measures and the process of Loading and Saving our Models, and
#how to build other types of models using different API's, how to use different kinds of metrics, visualizing what
#our model sees using callbacks, data augmentation, dropout regularization, early stopping, batch normalization, 
#instant normalization, layer normalization, weight initialization, learning rate scheduling, custom losses and
#metrics, sampling methods, custom training, tensor bot and hyperparameter tuning, weights abd biases logs, weights
#and biases artifacts, and finally weights and biases sweeps.


# In our previoud section we built a model based on convolutional networks to help detect the presence of malaria in
#blood cells.

# Nonetheless, in the real world, we are not always going to be using our models in a closed enviornment.

# Hence, we need to be able to save our model so it can be used externally.

# In this section we will learn how to save and load a model and also do the same process with google drive.

# That is, we will be able to save our model in our google drive and then later on when we want to use this model,
#we'll just load it from our google drive.

# So we've built this very performance model, though we could improve it.

# But then once we close it, we do not save the models current state.

# And so, if we have to come next time, the model will have randomly initialized weights which will be different
#from the weights we have now after training on this dataset.

# Another issue is in case we want to use this model in another scenario or in another enviornment, like a browser
#or on a mobile device, we'll need a find a way to export this model from here.

# And so Tensorflow allows us to save our model.

# Now, we'll have to differentiate between a model's configuration and a model's weight.

# Let's suppose we have a model that is defined as such.

# We have the input, which we pass into a count layer, then we have batch normalization, we have pooling for
#subsampling and then we flatten, and after flattening we pass through a dense layer and we have our output.

# Now, all the parameters for the creation of this model are known as the model's configuration.

# In the models configuration, we may have it that the model for example, like in this case, the model starts with
#a count layer with six filters, kernel size 3, batch norm and all this.

# So these are our model's configurations, but this model's configurations are different from the model's weights.

# The models weights are those filters we have, for example in the case of the Conv2D

# So we have the model weights and the model's configuration.

# And upon summarizing the model, we see clearly that we have a Conv2D, and then we have a number of parameters.

# And so whenever we want to save a model, we have to take into consideration the configuration and the weights
#because for this same configuration, we could have different weights.

# And so there are actually two main options.

# The first option is to save the full model.

# That is to save the model confihuration and the model weights.

# Another option will be to save only the model weights.

# Now this option is used when, for example, when we don't want to, or we don't even know the model 
#configuration upfront.

# So we've defined the model's configuration, we've trained it, we've got new weights, and this is the current model
#state.

# But if we take this to another enviornment where we don't have this configuration, then if we've saved this model's
#configuration and weights, all we need to do is just load this configuration and weights which have been packaged
#as the full model. 

# Now in another case where we are able to get the configuration and all we need is the weights, then we'll just save
#the weights and then reload the weights, since we already have the configuration.

# Either way, we'll always need the configuration and the weights.

# Nonetheless, it is important to note that the most important part of this is acutally the weights, since working 
#with a randomized or randomly initialized weights after we've trained our model isn't very useful.

# Sometimes it may take many days to train our models. So imagine we train our model for let's say, 10 days, for
#example, and then we want to reuse that model and the weights have been randomly initialized.

# Those 10 days of training will have been wasted.

# So we to ensure that we save our weights properly, such that we could reuse them.

# The great thing with tensorflow is that we could always continue training from that state.

# So this means that at this point where we've gotten the model's performance, let's say a 94%, for example,
#we could keep training from there so that we could get to an even high percentage, like 99%, for example.

# That is why we have to ensure that saving is done properly.

# Now let's get into that.

# But before we get into that, let's go over one last point.

# Also note that with the first method, apart from the model configurations, we also have information like the metrics.

# So the metrics we use like the accuracy, the loss we use, the opitimizer.

# So the optimizer information we use and all that.

# So this kind of hyperparameter information has been saved.

# So next time, all we have to do is just load our model and then make use of it.

# Whereas previously, all we were saving was the weights.

# With that said, let's save our model.