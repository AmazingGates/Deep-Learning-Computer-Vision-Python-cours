import tensorflow as tf 
# Here we will be going over Sparse Tensors.

# tf.sparse.SparseTensor represents a sparse tensor.

tensor_sparse = tf.sparse.SparseTensor(
    indices=[[1,1], [3,4]], values=[11,56], dense_shape=[5,6]
)

print(tensor_sparse) # Ouput [[1 1]
#                             [3 4]], shape=(2, 2), dtype=int64), values=tf.Tensor([11 56], shape=(2,), dtype=int32), 
#                             dense_shape=tf.Tensor([5 6], shape=(2,), dtype=int64))
# Notice that our value is exactly how we specified.

# Now we will see how this relates to a regular tensor.

print(tf.sparse.to_dense(tensor_sparse)) # Output tf.Tensor(
#                                                           [[ 0  0  0  0  0  0]
#                                                            [ 0 11  0  0  0  0]
#                                                            [ 0  0  0  0  0  0]
#                                                            [ 0  0  0  0 56  0]
#                                                            [ 0  0  0  0  0  0]]
# Notice the shape we get back. It is the same shape we defined in our initial tensor_sparse.
# indices=[[1,1], [3,4]], values=[11,56], dense_shape=[5,6]. Notice that we specified inour indices that we 
#we want a value of 11 at our [1,1] position, and a value of 56 at our [3,4] position.
# This is how we are able to map this back to our usual tensor.