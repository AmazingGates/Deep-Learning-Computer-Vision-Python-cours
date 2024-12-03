import tensorflow as tf
# Here we will be going String Tensors

# tf.strings, used for operations for working with string tensors.

# Before we go over the strings method, we will go over the process of creating a simple string tensor.

tensor_string = tf.constant(["hello", "I am", "amazing"])
print(tensor_string) # Output tf.Tensor([b'hello' b'I am' b'amazing']
# Notice we are returned our simple string tensor

# Here we will be looking the tf.strings.join method. This method allows us to perform element-wise concatenation 
#of a list of string tensors.

print(tf.strings.join(tensor_string)) # Output tf.Tensor(b'helloI amamazing'
# Notice that our separate strings are now joined into one, but there is no spacing between our words. We can
#fix this by adding a separator in our formula. See example below.

print(tf.strings.join(tensor_string, separator=" ")) # Output tf.Tensor(b'hello I am amazing'
# Notice that we are returned our separate strings as one, but now because of the separator we have the proper
#spacing between our words.

# Here we will be looking at the tf.strings.length method. This method computes the length of each string given in 
#the input tensor.

print(tf.strings.length(tensor_string)) # Output tf.Tensor([5 4 7]
# Notice the we are returned the value of each input separately.
# Note: This method also counts the spaces included in any value in our input.

# Here we will be going over the tf.strings.lower method. This method converts all uppercase characters into their
#respective lowercase replacements.

print(tf.strings.lower("I LOVE YOU ALIA MARIE GATES")) #Output tf.Tensor(b'i love you alia marie gates'
# Notice that our returned string is converted to all lowercase.