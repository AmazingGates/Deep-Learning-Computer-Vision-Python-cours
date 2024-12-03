import tensorflow as tf

# Here we will be going over Math Operations

# Here we will be looking at the tf.math.abs function. This function computes the absolute value of a tensor.
# Absolute values are describe as such. If a number is a negative, it gets converted into a positive.
x_abs = tf.constant([-2.25, 3.25])
tf.abs(x_abs)
print(tf.abs(x_abs)) # This is our output. tf.Tensor([2.25 3.25], shape=(2,), dtype=float32)
# Notice that our [-2.25] gets returned as a positive number. But the [3.25] stays the same, because it is
#already a positive.

# Here we will be going over absolutes for complex tensors. 
# Given a tensor x of complex numbers, this operation returns a tensor of type float32 or float64 that is the
#absolute valueof each element in x. For a complex number a + bj. Its absolute value is computed as
#the square root of a squared + b squared.
x_abs_complex = tf.constant([-2.25 + 4.75j])
print(tf.abs(x_abs_complex)) # This will be our output. tf.Tensor([5.25594901], shape=(1,), dtype=float64)
# Note, we must include the j at the end of our complex equation for the correct result.

# Here we wil be looking at the square root method sqrt. This method computes element-wise square root of the
#input tensor.
# Note: This operation does not support integer types.
x_sqrt = tf.sqrt((-2.25)**2 + 4.75**2) # This is how we can also write the above equation without having to use a "j".
print(x_sqrt) # This is our output tf.Tensor(5.255949, shape=(), dtype=float32)
# Note: We must use a bracket on the first number to get back the correct result.

# Here we will look at an example addition.
x_1 = tf.constant([5,3,6,6,4,6], dtype = tf.int32) 
x_2 = tf.constant([7,6,2,6,7,11], dtype = tf.int32)
print(tf.add(x_1,x_2)) # This is our output tf.Tensor([12  9  8 12 11 17], shape=(6,), dtype=int32)

# Here we look at an example of multiplication
x_1 = tf.constant([5,3,6,6,4,6], dtype = tf.int32) 
x_2 = tf.constant([7,6,2,6,7,11], dtype = tf.int32)
print(tf.multiply(x_1,x_2)) # This is our output tf.Tensor([35 18 12 36 28 66], shape=(6,), dtype=int32)

# Here we will look at an example of subtraction
x_1 = tf.constant([5,3,6,6,4,6], dtype = tf.int32) 
x_2 = tf.constant([7,6,2,6,7,11], dtype = tf.int32)
print(tf.subtract(x_1,x_2)) # This is our output tf.Tensor([-2 -3  4  0 -3 -5], shape=(6,), dtype=int32)

# Here we will look at an example of division
x_1 = tf.constant([5,3,6,6,4,6], dtype = tf.int32) 
x_2 = tf.constant([7,6,2,6,7,11], dtype = tf.int32)
print(tf.divide(x_1,x_2)) # This is our output tf.Tensor([0.71428571 0.5        3.         1.         
#0.57142857 0.54545455], shape=(6,), dtype=float64)

# Here we will be looking at the tf.math.divide_no_nan method. This computes a safe divide which returns 0 if y
#(denominator) is zero.
x_1 = tf.constant([5,3,6,6,4,6], dtype = tf.int32) 
x_2 = tf.constant([0,6,2,6,7,11], dtype = tf.int32)
print(tf.divide(x_1,x_2)) # This is our output tf.Tensor([       inf 0.5        3.         1.         
#0.57142857 0.54545455], shape=(6,), dtype=float64)
# Notice that we changed the first number of our denominator to a zero, and in return we got back an inf(infinity) sign
# as a result of the block of equation.

# Here we will look at a solution for the inf(infinity) sign issue.
x_1 = tf.constant([5,3,6,6,4,6], dtype = tf.float32) 
x_2 = tf.constant([0,6,2,6,7,11], dtype = tf.float32)
print(tf.math.divide_no_nan(x_1,x_2)) # This is our output tf.Tensor([0.         0.5        3.         1.         
#0.5714286  0.54545456], shape=(6,), dtype=float32)
# Notice that this time we got back a 0 instead of an inf(infinity) sign. That's thanks to the tf.math.divide_no_nan
#formula.
# Note: The dtype must be a float in order for the equation to work. tf.float32

# Here we look at another example of addition.
x_1 = tf.constant([5,3,6,6,4,6], dtype = tf.int32) 
x_2 = tf.constant([7], dtype = tf.int32)
print(tf.add(x_1,x_2)) # This is our output tf.Tensor([12 10 13 13 11 13], shape=(6,), dtype=int32)
# Notice that even though we have one number in our bottom row, it is applied to every number in the top row
#to carry out our equation.
# This is what we call broadcasting. In broadcasting the smaller tensor is stretched out to match the shape
#of the larger tensor so that the equation can be carried out without error.

# Here we will look at an example of broadcasting in multiplication
x_1 = tf.constant([5,3,6,6,4,6], dtype = tf.int32) 
x_2 = tf.constant([7], dtype = tf.int32)
print(tf.multiply(x_1,x_2)) # This is our output tf.Tensor([35 21 42 42 28 42], shape=(6,), dtype=int32)

# Here we will use broadcasting in multiplication on a more complex equation.
x_1 = tf.constant([[5,3,6,6,4,6], [5,45,65,5,53,4]], dtype = tf.int32) 
x_2 = tf.constant([7], dtype = tf.int32)
print(tf.multiply(x_1,x_2)) # This is our output tf.Tensor([[ 35  21  42  42  28  42]
#[ 35 315 455  35 371  28]], shape=(2, 6), dtype=int32)

# Here we will use a 3 by 1 tensor to multiply our top row of numbers.
x_1 = tf.constant([[5,3,6,6,4,6],], dtype = tf.int32) 
x_2 = tf.constant([[7], [5], [3]], dtype = tf.int32)
print(tf.math.multiply(x_1,x_2)) # This is our output tf.Tensor([[35 21 42 42 28 42][25 15 30 30 20 30]
#[15  9 18 18 12 18]], shape=(3, 6), dtype=int32)
# Notice that our top row is multiplied by each number in our 3 by 1, and returns a separate list for each
#equation.
# Note: We must use the .math in our print to get the correct result. tf.math.multiply()

# Here we will be looking at the tf.math.maximum method. This returns the max of x and y (i.e x > y? x:y)
#element-wise.

x = tf.constant([3,6,9,12,15])
y = tf.constant([21,33,5,2,1])

tf.math.maximum(x, y)
print(tf.math.maximum(x, y)) # This is the output tf.Tensor([21 33  9 12 15], shape=(5,), dtype=int32)
# Notice that the max number from each column is returned.

# Here we will be looking at the tf.math.minimum method. This returns the min of x and y (i.e x > y? x:y)
#element-wise.

x = tf.constant([3,6,9,12,15])
y = tf.constant([21,33,5,2,1])

tf.math.minimum(x, y)
print(tf.math.minimum(x, y)) # This is the output tf.Tensor([3 6 5 2 1], shape=(5,), dtype=int32)
# Notice that the min number from each column is returned.

# Here we will be looking at the tf.math.argmax. This function returns the index with the largest value across axes 
#of a tensor. 

x_argmax = tf.constant([[2, 20, 30, 3, 6], 
                        [3, 11, 16, 1, 8], 
                        [14, 45, 23, 5, 27]])
print(tf.math.argmax(x_argmax)) # This is the output tf.Tensor([2 2 0 2 2], shape=(5,), dtype=int64)
# Notice the output gives us the location on the highest number from every column in every index.

# Here we will be looking at the tf.math.argmin. This function returns the index with the smallest value across axes 
#of a tensor.

x_argmin = tf.constant([[2, 20, 30, 3, 6], 
                        [3, 11, 16, 1, 8], 
                        [14, 45, 23, 5, 27]])
print(tf.math.argmin(x_argmin)) # This is the output tf.Tensor([0 1 1 1 0], shape=(5,), dtype=int64)
# Notice the output gives us the location on the lowest number from every column in every index.

# Here we will be looking at tf.math.equal. This returns the truth value of (x==y) element-wise.

x = tf.constant([2,4])
y = tf.constant(2)
tf.math.equal(x,y)
print(tf.math.equal(x,y)) # This is the output tf.Tensor([ True False], shape=(2,), dtype=bool)
# Notice that the y equation is using broadcasting, meaning that 2 will be applied to every element of x.
# Notice that we have a True False return. This is because 2 == 2 is True and 2 == 4 is False.
# Notice also that our data type is now a boolean.

# Here we will be looking at the power method tf.math.pow. This computes the power of value to another.
# Given a tensor x and a tensor y, this operation computes x by y for corresponding elements in x and y.

x = tf.constant([[2, 2],  [3, 3]])
y = tf.constant([[8, 16], [2, 3]])
tf.pow(x, y)
print(tf.pow(x, y)) # This is the output tf.Tensor([[  256 65536] [    9    27]], shape=(2, 2), dtype=int32)
# Notice that numbers in every column in every column in every index are paired. 
# Notice also that the numbers of x are multiplied to the power of the numbers of y.

# Here we will be going over the method tf.math.reduce_sum. This computes the sum of elements across dimensions
#of a tensor.

x = tf.constant([[1, 1, 1], [1, 1, 1]])
x.numpy()
tf.reduce_sum(x).numpy()
print(tf.reduce_sum(x).numpy()) # This is the output. 6
# Notice that we took all of the elements from both of our shapes and reduced them down to a single sum after
#we added them all together.

x = tf.constant([[1, 1, -2], [1, 1, 1]])
x.numpy()
tf.reduce_sum(x).numpy()
print(tf.reduce_sum(x).numpy()) # This is the output. 3
# Notice that we took all of the elements from both of our shapes and reduced them down to a single sum after
#we added them all together.
# Note: Because we have a -3 as one of our elements, that amount will be subtracted from the sum of the other
#numbers before a final sum is returned.

x = tf.constant([[2, 20, 30, 3, 6], 
                 [3, 11, 16, 1, 8], 
                 [14, 45, 23, 5, 27]])
print(tf.math.reduce_sum(x, axis=0, keepdims=False, name=None)) # This is the output tf.Tensor([19 76 69  9 41], 
#shape=(5,), dtype=int32)
# Note: By using this formula (axis=0, keepdims=False, name=None))on a more complex shape, we actually reduce the
#elements of each column and return the the sum of each column separately.
# Notice that we are returned a sum for each column from every index.

x = tf.constant([[2, 20, 30, 3, 6], 
                 [3, 11, 16, 1, 8], 
                 [14, 45, 23, 5, 27]])
print(tf.math.reduce_sum(x, axis=1, keepdims=False, name=None)) # This is the output tf.Tensor([ 61  39 114], 
#shape=(3,), dtype=int32)
# Note: By using this formula axis=1, keepdims=False, name=None)) we can actually sum up each row, instead of each
#column when we used the axis=0. Using the axis=1 will sum up every row in all indexes and return a sum for each
#row separately.

# Here we will be at the method tf.math.reduce_mean. This computes the elements across dimensions
#of a tensor to their average value.

x = tf.constant([[2, 20, 30, 3, 6], 
                 [3, 11, 16, 1, 8], 
                 [14, 45, 23, 5, 27]])
print(tf.math.reduce_mean(x, axis=0, keepdims=False, name=None)) # This is the output tf.Tensor([ 6 25 23  3 13], 
#shape=(5,), dtype=int32) 
# Note: By using this formula (axis=0, keepdims=False, name=None))on a more complex shape, we actually get the mean
#of each column and return the median number of each column separately. Also, even if the returned number is a 
#decimal, it will be returned as a whole number because our data type is set to integer.
# Notice that we get the median number for every column in every index. 

x = tf.constant([[2, 20, 30, 3, 6], 
                 [3, 11, 16, 1, 8], 
                 [14, 45, 23, 5, 27]])
print(tf.math.reduce_mean(x, axis=1, keepdims=False, name=None)) # This is the output tf.Tensor([12  7 22], shape=(3,), 
#dtype=int32)
# Note: By using this formula (axis=1, keepdims=False, name=None))on a more complex shape, we actually get the mean
#of each row and return the median number of each row separately. Also, even if the returned number is a 
#decimal, it will be returned as a whole number because our data type is set to integer.
# Notice that we get the median number for every row in every index.

# Here we will be looking at the method tf.math.sigmoid. This method computes sigmoid of x element-wise.
# This is the formula for calculating sigmoid (x) = y = 1/(1 + exp(-x))
# Example usage: If a positive number is large, then its sigmoid will approach to 1 since the formula
#will be y = <large_num> / (1 + <large_num>)

x = tf.constant([0.0, 1.0, 50.0, 100.0])
tf.math.sigmoid(x)
print(tf.math.sigmoid(x)) # Output tf.Tensor([0.5       0.7310586 1.        1.       ], shape=(4,), dtype=float32)
# Notice that all of our original elements are computed to their sigmoid versions. That is because when we 
#run this function, each element will be passed and processed individually through the function.

# Here we will be looking at the tf.math.top_k function. This finds values and indices of the k largest entries
#for the last dimension.

tensor_two_d = tf.constant([[1,2,0],
                           [3,5,1],
                           [1,5,6],
                           [2,3,8]])

tf.math.top_k(tensor_two_d)
print(tf.math.top_k(tensor_two_d)) # Output TopKV2(values=<tf.Tensor: shape=(4, 1), dtype=int32, numpy=
#array([[2],
#       [5],
#       [6],
#       [8]])>, indices=<tf.Tensor: shape=(4, 1), dtype=int32, numpy=
#array([[1],
#       [1],
#       [2],
#       [2]])>)
# Notice that we are returned two arrays. The first is the largest number from each index, and the second is the
#position of each of those numbers in the index.

tf.math.top_k(tensor_two_d, k = 2) # K=2 will return the top 2 numbers, and return their positions
print(tf.math.top_k(tensor_two_d, k = 2)) # Output TopKV2(values=<tf.Tensor: shape=(4, 2), dtype=int32, numpy=
#array([[2, 1],
#       [5, 3],
#       [6, 5],
#       [8, 3]])>, indices=<tf.Tensor: shape=(4, 2), dtype=int32, numpy=
#array([[1, 0],
#       [1, 0],
#       [2, 1],
#       [2, 1]])>)
# Notice that we are returned two arrays. The first is the 2 largest numbers from each index. The second is the 
#position of each of those numbers in tthe index.