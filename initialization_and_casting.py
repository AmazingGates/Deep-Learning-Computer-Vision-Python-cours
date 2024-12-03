# Initialization and Casting - Here we will be creating some Tensors
import tensorflow as tf
import numpy as np

#tensor_zero_d = tf.constant(4) # This is how we create a zero dimensional tensor. Also "constsnt" takes in a data type
#a shape and a name.
#print(tensor_zero_d) # This will print tf.Tensor(4, shape=(), dtype=int32)

# With this we could go ahead and build out our tensor 1d, tensor 1d we have as usual our constant method,
#but this time around because it;s 1d we're going to have a list. We're going to take this list we had here
#[2 0 -3] and use it as our constant.

#tensor_one_d = tf.constant([2, 0, -3])
#print(tensor_one_d) # This will print tf.Tensor([ 2  0 -3], shape=(3,), dtype=int32). Reminder, the shape
#is counting the number of elements inside the tensor.

#tensor_one_d2 = tf.constant([2, 0, -3, 8, 90],dtype=tf.float32) # This is another way we can use a float instead
#of using the decimal point after the 90. Also, if we use the (float16), we will get the same output, but we 
#will use less memory in the storage to run this version. We could also use a (float64), but in certain circumstances
#where we have memory constrant we want to use the lower position tensor.
#print(tensor_one_d2) # This will print tf.Tensor([ 2.  0. -3.  8. 90.], shape=(5,), dtype=float32). Notice
#that because of the decimal after the 90., our datatype is now a float. And our shape is now 5, because of the added
#elements.

#tensor_two_d = tf.constant([
#    [1,2,0],
#    [3,5,-1],
#    [1,5,6],
#    [2,3,8]
#]) # Note, 1d tensors must be inside of a list to represent a 2d tensor.
#print(tensor_two_d) # This will print tf.Tensor([[ 1  2  0][ 3  5 -1][ 1  5  6][ 2  3  8]], shape=(4, 3), dtype=int32)

#tensor_three_d = tf.constant([
#    [[1,2,0],[3,5,-1]],
#    [[10,2,0],[1,0,2]],
#    [[5,8,0],[2,7,0]],
#    [[2,1,9],[4,-3,32]]
#]) # Note, each 2d tensor must be inside of a list, separated by a comma
#print(tensor_three_d) # This will print tf.Tensor([[[ 1  2  0][ 3  5 -1]] [[10  2  0][ 1  0  2]] [[ 5  8  0]
#[ 2  7  0]] [[ 2  1  9][ 4 -3 32]]], shape=(4, 2, 3), dtype=int32)

# We can also use print to print the shape of the tensor directly.
#print(tensor_three_d.shape) # This will print the shape of our 3d tensor. This will work for all of our tensors.

# We can also use print to print the number of our dimension numerically.
#print(tensor_three_d.ndim) # This will print the number 3, representing 3d.

# Next we will print a 4d tensor by adding 3 3d tensors together.
#tensor_four_d = tf.constant([
#    [[[1,2,0],[3,5,-1]],
#    [[10,2,0],[1,0,2]],
#    [[5,8,0],[2,7,0]],
#    [[2,1,9],[4,-3,32]]],
#    [[[1,2,0],[3,5,-1]],
#    [[10,2,0],[1,0,2]],
#    [[5,8,0],[2,7,0]],
#    [[2,1,9],[4,-3,32]]],
#    [[[1,2,0],[3,5,-1]],
#    [[10,2,0],[1,0,2]],
#    [[5,8,0],[2,7,0]],
#    [[2,1,9],[4,-3,32]]]
#])
#print(tensor_four_d) # This will print tf.Tensor([[[[ 1  2  0][ 3  5 -1]][[10  2  0][ 1  0  2]][[ 5  8  0]
#[ 2  7  0]][[ 2  1  9][ 4 -3 32]]] [[[ 1  2  0][ 3  5 -1]][[10  2  0][ 1  0  2]][[ 5  8  0][ 2  7  0]]
#[[ 2  1  9][ 4 -3 32]]] [[[ 1  2  0][ 3  5 -1]][[10  2  0][ 1  0  2]][[ 5  8  0][ 2  7  0]][[ 2  1  9]
#[ 4 -3 32]]]], shape=(3, 4, 2, 3), dtype=int32)

#casted_tensor_one_d = tf.cast(tensor_one_d,dtype=tf.int16) # This is how we would cast out tensor into an int.
#print(tensor_one_d)
#print(casted_tensor_one_d)

#casted_tensor_two_d = tf.cast(tensor_two_d,dtype=tf.bool) # This is how we would cast our tensor as a boolean
#print(tensor_two_d)
#print(casted_tensor_two_d)
# When casting a boolean, all values of zero will return false, but all other values will return true, even 
#negatives.

# We can even create our own booleans. 
#tensor_bool = tf.constant([True,True,False])
#print(tensor_bool)

# Another data type we can look at is string.
#tensor_string = tf.constant(["Alia Marie", "Gates "])
#print(tensor_string)

# Here we will convert a numpy array into a tensor.
#np_array = np.array([1,2,4])
#print(np_array)

# Now we can make use of this tensor flow convert to tensor method, so let's say we have...
#converted_tensor = tf.convert_to_tensor(np_array) # This will take in a non-py array
#print(converted_tensor)

# Here we will use the tf.eye method. This method let's us construct an identity matrix, or a batch of matrices.
#eye_tensor = tf.eye(
#    num_rows=3,
#    num_columns=None, # Because we have 3 rows, the number of columns will automatically be 3.
#    batch_shape=None,
#    dtype=tf.dtypes.float32,
#    name=None
#)
#print(eye_tensor)
#print(3*eye_tensor) # This will multiple and print all the values in our eye tensor.

#eye_tensor1 = tf.eye(
#    num_rows=3,
#    num_columns=None, # Because we have 3 rows, the number of columns will automatically be 3.
#    batch_shape=None,
#    dtype=tf.dtypes.bool, # Here we will use a bool instead of a float
#    name=None
#)
#print(eye_tensor1)

#eye_tensor2 = tf.eye(
#    num_rows=5,
#    num_columns=3, # Because we changed this to 3 instead of None, we erase 2 of columns by default.
#    batch_shape=None,
#    dtype=tf.dtypes.float32,
#    name=None
#)
#print(eye_tensor2)

#eye_tensor3 = tf.eye(
#    num_rows=5,
#    num_columns=None, # Because we have 5 rows, the number of columns will automatically be 5.
#    batch_shape=None,
#    dtype=tf.dtypes.float32,
#    name=None
#)
#print(eye_tensor3)

#eye_tensor4 = tf.eye(
#    num_rows=5,
#    num_columns=None,
#    batch_shape=[3,], # By changing the batch shape to 3, it gives us 3 batches or a 3d tensor of the same grid.
#    dtype=tf.dtypes.float32,
#    name=None
#)
#print(eye_tensor4)

#eye_tensor5 = tf.eye(
#    num_rows=5,
#    num_columns=None,
#    batch_shape=[2,4], # This will give us two by four, two by four by five by five. We should have 
    #8 grids (or 2 sets of 4) 5 by 5 grids.
#    dtype=tf.dtypes.float32,
#    name=None
#)

#print(eye_tensor5)


# Here we will be looking at the tf.fill method. This method creates a tensor filled with a scalar value.
# tf.fill evaluates at graph runtime and supports dynamic shapes based on other runtime tf.Tensors, unlike
#tf.constant(value, shape=dims), which embeds the value as a const node.
 
fill_tensor = tf.fill( # This is how we use the tf.fill
    [3,4],5, name=None # Thease are the elements
)
print(fill_tensor) # This is the output tf.Tensor([[5 5 5 5][5 5 5 5][5 5 5 5]], shape=(3, 4), dtype=int32)
# The tf.fill method fills the entire Tensor with the value we choose. In this case the number 5.


# Here we will be looking at the tf.ones method. This method creates a tensor with all elements set to one.

ones_tensor = tf.ones( # This is how we use the tf.ones method
    [5,3], # This is our shape that our ones will occupy
    dtype=tf.dtypes.float32, # This is the type cast of the tensor
    name=None
)
print(ones_tensor) # This is the output tf.Tensor([[1. 1. 1.][1. 1. 1.][1. 1. 1.][1. 1. 1.][1. 1. 1.]], 
#shape=(5, 3), dtype=float32)

# We can also change the above output into a 3d tensor by adding an additional element.

ones_tensor2 = tf.ones(
    [5,3,2],
    dtype=tf.dtypes.float32,
    name=None
)
print(ones_tensor2) # This is the output tf.Tensor([[[1. 1.][1. 1.][1. 1.]] [[1. 1.][1. 1.][1. 1.]] 
#[[1. 1.][1. 1.][1. 1.]] [[1. 1.][1. 1.][1. 1.]] [[1. 1.][1. 1.][1. 1.]]], shape=(5, 3, 2), dtype=float32)


# Here we will be looking the tf.ones_like. This method creates a tensor of all ones that has the same shape
#as the input.

ones_like_tensor = tf.ones_like(fill_tensor) # This is how we will write it
print(ones_like_tensor) # This is the output tf.Tensor([[1 1 1 1] [1 1 1 1] [1 1 1 1]], shape=(3, 4), dtype=int32)
# Notice that since fill_tensor has a shape of [3,4] that is the shape filled with our value of the number 1


# Here we will be looking at tf.zeros. This method creates a tensor with all-elements set to zero.

zeros_tensor = tf.zeros( # This is how we will write it
    [3,2], # This the shape our value (zero) will fill
    dtype=tf.dtypes.float32, # This is our tensors type and cast
    name=None
)
print(zeros_tensor) # This is the output tf.Tensor([[0. 0.][0. 0.][0. 0.]], shape=(3, 2), dtype=float32)
# Notice that our value (zero) has taking the shape of [3,2]


# Here we will be looking at the tf.shape. This method returns a tensor containing the shape of the input tensor.

print(tf.shape(ones_tensor2)) # This will be the output. tf.Tensor([5 3 2], shape=(3,), dtype=int32)
# Notice that this returns tensor, its shape, and its data type.


# Here we will be looking the tf.rank. This method returns the rank of a tensor. So it takes in a tensor
#and returns its rank.

# The shape of tensor "t" is [2,2,3]
t = tf.constant([[[1,1,1], [2,2,2]], [[3,3,3], [4,4,4]]])
tf.rank(t) #3
print(tf.rank(t))
# Note: The rank of a tensor is not the same as the rank of a matrix. The rank of a tensor is the number 
#of indices required to uniquely select each element of the tensor. Rank is also known as "order", "degree",
#or "ndims". We get a rank of 3 because we have 2 tensors, 2 times, each with 3 elements.


# Here we will be looking at tf.size. This method returns the size of a tensor.
# See also tf.shape
#Returns a 0d tensor representing the number of elements in input of type out_type. Defaults to tf.int32

t = tf.constant([[[1,1,1], [2,2,2]], [[3,3,3], [4,4,4]]])
tf.size(t)
print(tf.size(t)) # This will be the output tf.Tensor(12, shape=(), dtype=int32)
# Notice the 12. That will be the size of our tensor. We get 12 because that is how many elements we have 
#in total in our tensor.


# Here we will be looking at tf.random.normal. This method outputs random values from a normal distribution.

random_tensor = tf.random.normal(
    [3,2],
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
print(random_tensor) # This is the output tf.Tensor([[-0.10781351  1.6829909 ][-2.3236163  -0.24759848]
#[-1.6699739   1.0099307 ]], shape=(3, 2), dtype=float32)
# Notice that we get back a bunch of random float numbers in the shape of a 2d.
# Note: We will get a different set of random float numbers every time we print this variable.
# Also, we can alter the range of numbers we get by changing the mean.


# Here we will be looking at the tf.random.uniform. This method outputs random values from a uniform distribution.

random_tensor2 = tf.random.uniform(
    [1,],
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
print(random_tensor2) # This is the output. tf.Tensor([0.12439454], shape=(1,), dtype=float32)
# Notice that there is only one random number as specified by the number in our shape.

random_tensor2 = tf.random.uniform(
    [3,], # Here we will increase the number in our shape and watch its effects.
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
print(random_tensor2) # This is the output. tf.Tensor([0.58462346 0.5858331  0.8960749 ], shape=(3,), dtype=float32)
# Notice that we know have 3 random numbers as specified by the number in our shape.

random_tensor3 = tf.random.uniform(
    [6,],
    minval=0,
    maxval=8, # Here we will change the maximum value and watch its effects
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
print(random_tensor3) # This is the output. tf.Tensor([7.9702854 2.8587427 7.1154165 6.629303  5.2607365 2.166915 ],
#shape=(6,), dtype=float32)
# Notice that our random numbers are all higher because of the fact that re raised the max value to 8. Our
#numbers will be from anywhere between 0 and 8. 

random_tensor4 = tf.random.uniform(
    [6,],
    minval=0,
    maxval=15,
    dtype=tf.dtypes.int32, # Here we will change the float into an int and watch its effects
    seed=None,
    name=None
)
print(random_tensor4) # This is the output. tf.Tensor([10  5 13 13  9 10], shape=(6,), dtype=int32)
# Notice that all of our numbers are now ints instead of floats. And you can see that our dtype has been changed
#to int32.
# Note: For ints in the tf.random.uniform the maxval has to be higher than 0 in order to work.

# Another argument that we have in our code but have not spoken about is the seed argument.
# Seed = A Python Integer. Used in combination with Tf.random.set_seed to create a reproducible sequence
#of Tensors across multiple calls.

tf.random.set_seed(5)

print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10)) # tf.Tensor([4 3 1], shape=(3,), dtype=int32)
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10)) # tf.Tensor([4 3 2], shape=(3,), dtype=int32)
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10)) # tf.Tensor([1 1 1], shape=(3,), dtype=int32)
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10)) # tf.Tensor([1 3 3], shape=(3,), dtype=int32)
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10)) # tf.Tensor([4 2 2], shape=(3,), dtype=int32)
# Notice that we have 5 sets of 3 random integer numbers, and because of the seedbeing 10 for all of the prints, if
#we run the code again we will get the sam exact 5 sets of 3 random integer numbers.