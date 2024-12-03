import tensorflow as tf
# Here we will be going over Ragged Tensors. 

# Here we will be going over tf.ragged.
# This package defines ops for manipulating ragged tensors (tf.RaggedTensors), which are tensors with non uniform
#shapes. In particular, each RaggedTensor has one or more ragged dimensions, which are dimensions whose slices
#may have different lengths.
# For example, the inner (column) dimensions of rt = [[3, 1, 4, 1], [], [5, 9, 2], [6], []] is ragged, since the 
#column slices (rt[0, :],... rt[4, :]) have different lengths. 

# Here is an example of an ragged tensor

tensor_two_d = tf.constant([[1, 2, 0],
                            [3,],
                            [1, 5, 6],
                            [2, 3]])
print(tensor_two_d.shape)
# Note: If we ran the code like this, we would get back this output 
#ValueError: Can't convert non-rectangular Python sequence to Tensor.
# This is because our tensor is ragged.

# TensorFlow has a way of dealing with this issue, by using Ragged Tensors.

# Here is an example of a Ragged Tensor

tensor_two_d = [[1, 2, 0],
                 [3,],
                 [1, 5, 6, 5, 6],
                 [2, 3]]
# Note: For this, we will use tensor_two_d as a simple list, and not a tf.constant. Then we will pass
#tensor_two_d as a parameter into our tf.ragged.constant(). Also, we will be printing out the
#tensor_ragged variable

tensor_ragged = tf.ragged.constant(tensor_two_d)
print(tensor_ragged) # Output 


