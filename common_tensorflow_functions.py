import tensorflow as tf
# Here we will be going over Common Tensorflow Functions.

# We will be looking at the tf.expand_dims. This method returns a tensor with a length 1 axiss inserted at index axis

tensor_three_d = tf.constant([[[1,2,0],
                               [3,5,-1]],
                               
                               [[10,2,0],
                                [1,0,2]],

                                [[5,8,0],
                                 [2,7,0]],
                                 
                                 [[2,1,9],
                                  [4,-3,32]]])

print(tensor_three_d.shape) # Output (4, 2, 3) is the shape of our tensor
# Notice that we can add an extra axis to our tensor and change its shape/dimensions
print(tf.expand_dims(tensor_three_d, axis = 0).shape) # This is how we add an extra axis to our tensor.
# Output (1, 4, 2, 3) is our new ouptut with the extra added axis.
# Note: We must specify the axis = 0 in our print in order to output our answer successfully

#  Now let's take a simple example
x = tf.constant([2, 3, 4, 5])
print(x.shape) # This will print our original shape. Output (4,)
# Notice that this the shape of our original shape. This is a 1-d tensor with 4 elements
print(tf.expand_dims(x, axis = 0).shape) # This will print our expanded shape. Output (1, 4)
# Notice that this is the shape of expanded shape. It is now a 2-d shape with 4 elements

# Note: We can also expand our shape by adding extra brackets directly to our tensor.
# Example.
x = tf.constant([[2, 3, 4, 5]]) # Notice that this is the same shape from our previous example. We added brackets to
#indicate that we now want it to be a 2-d instead of a 1-d.
print(x.shape) # This will print our 2-d shape. Output (1, 4)
# Notice that we have changed our 1-d into a 2-d by manually adding brackets to our original shape.
print(tf.expand_dims(x, axis = 0). shape) # This will expand our modified shape. Output (1, 1, 4)
# Notice that we expanded our modified shape even futher by using the tf.expand_dims method. We went from
#a 2-d to a 3-d.
# Note: We specify where we want our extra axis to be placed by changing the axis = number. 
# Example 
print(tensor_three_d.shape) # This will print our original shape. Output (4, 2, 3)
print(tf.expand_dims(tensor_three_d, axis = 1). shape) # Notice that we changed the axis = 0 into an axis = 1 method.
# Output (4, 1, 2, 3)
# Notice that we added an extra axis, but instead of our output being (1, 4, 2, 3), we have (4, 1, 2, 3). That is
#because we specified that we want the expansion to happen at our 1 position, instead of our 0 location.

# Here we will be looking at tf.squeeze. This method removes dimensions of size 1 from the shape of a tensor.
#This works as an opposite of tf.expand_dims

x = tf.constant([2, 3, 4, 5])
print(x.shape) # Output  (4,)
x_expanded =tf.expand_dims(x, axis = 0) # Output (1, 4)
print(x_expanded) # # Output (1, 4)

x_squeezed = tf.squeeze(x_expanded, axis = 0) # This is how we select what we want to squeeze.
print(x_squeezed) # Output (4,)
# Notice that we went from a 2-d (1, 4) to a 1-d (4,). This happened because we used the x_squeezed method

# We can also perform this action one than once using a for loop.
# Example
x1 = tf.constant([[[2, 3, 4, 5]]])
print(x.shape)
x1_expanded =tf.expand_dims(x1, axis = 0)
print(x1_expanded) # Output (1, 1, 1, 4)
# Notice that our tensor is a 4-d, after we expanded it from a 3-d.

x1_squeezed = tf.squeeze(x1_expanded, axis = 0)
for i in range(2):
    x1_squeezed = tf.squeeze(x1_squeezed, axis = 0)

print(x1_squeezed) # Output (4,)
# Notice that our tensor is now a 1-d, after we squeezed it down from a 4-d using the for loop.

# Now we will perform the same squeeze function, but this time change its axis location.
print(tensor_three_d.shape)
x_exp = tf.expand_dims(tensor_three_d, axis = 3)
print(x_exp.shape) # This is our original shape (4, 2, 3, 1)
# Notice that it is a 4-d
print(tf.squeeze(x_exp, axis = 3)) # This is our new output once our tensor is squeezed (4, 2, 3)
# Notice that we went from a shape of (4, 2, 3, 1) to a shape of (4, 2, 3) by squeezing our tensor at the 
#axis = 3 location.

# Here we will be looking at the tf.reshape method. This method reshapes a tensor.
# Note: When reshaping We have to make sure that the number of values that we have in our initial tensor 
#actually fits in our new shape

x_reshape = tf.constant([[3,5,6,6],
                        [4,6,-1,2]])
tf.reshape(x_reshape, [8])
print(tf.reshape(x_reshape, [8])) # Output [ 3  5  6  6  4  6 -1  2]
# Notice that the reshape method reshaped our shape in a 1-d from a 2-d

# Here is another example
print(tf.reshape(x_reshape, [4,2])) # Output [[ 3  5]
#                                             [ 6  6]
#                                             [ 4  6]
#                                             [-1  2]]
# Notice that this time we changed our shape back into a 2-d, but this time it is in the form of 4 by 2,
#instead of a 2 by 4.

# One last example
print(tf.reshape(x_reshape, [4,2,1])) # Output [[[ 3]
#                                                [ 5]]

#                                               [[ 6]
#                                                [ 6]]

#                                               [[ 4]
#                                                [ 6]]

#                                               [[-1]
#                                                [ 2]]]

# Notice that we changed our shape one final time. This time we have a 3-d 4 by 2 by 1. (4,2,1)
# Even though our shape is technically a 3-d, it is valid because it still only has 8 elements.

# Here we will be looking at the tf.concat method. This method concatenates tensors along one dimension.

t1 = [[1, 2, 3],
      [4, 5, 6]]

t2 = [[7, 8, 9],
      [10, 11, 12]]

tf.concat([t1, t2], 0)
print(tf.concat([t1, t2], 0)) # Output [[ 1  2  3]
#                                       [ 4  5  6]
#                                       [ 7  8  9]
#                                       [10 11 12]]
# Notice that we now have a 4 by 3 (4,3), instead of two 2 by 3's (2,3). That is because we concatenated
# our t1 and t2 matrices. Also notice that we have an axis = 0 represented by the (0) in our tf.concat method.
# This tells the operation that we want to concatenate across our rows.

# Here we will perform the same exact concatenation, but this time, we will concatenate across our columns.

t1 = [[1, 2, 3],
      [4, 5, 6]]

t2 = [[7, 8, 9],
      [10, 11, 12]]

tf.concat([t1, t2], 0)
print(tf.concat([t1, t2], 1)) # Output [[ 1  2  3  7  8  9]
#                                       [ 4  5  6 10 11 12]]

# Notice that we now have a different shape than previously. That is because we have an axis = 1 represented
#by (1) inside our tf.concat method. This tells the operation that we want to concatenate across our columns
#instead of our rows. Also notice that we have a 2 by 6 (2,6) instead of a (4,3) like we get with the 
#axis = 0.

# Now we will use the tf.concat on a 3-d

t3 = [[[1, 2, 3],
      [4, 5, 6]]]

t4 = [[[7, 8, 9],
      [10, 11, 12]]]
# Note: We added extra brackets to our tensors to indicate that they are now 3-d instead of 2-d.

print(tf.constant(t3).shape) # Output (1, 2, 3)
print(tf.constant(t4).shape) # Output (1, 2, 3)
print(tf.concat([t3, t4], 0)) # Output [[[ 1  2  3]
#                                        [ 4  5  6]]

#                                        [[ 7  8  9]
#                                         [10 11 12]]] Shape (2, 2, 3)
# Notice that now instead of having two separate 3-d tensors with the shape of (1, 2, 3), we have now concatenated
#them together into one shape (2, 2, 3). Also notice that we were using the axis = 0.

# Now we will run the same exact code, but this time we will be using the axis = 1.

t5 = [[[1, 2, 3],
      [4, 5, 6]]]

t6 = [[[7, 8, 9],
      [10, 11, 12]]]
# Note: We added extra brackets to our tensors to indicate that they are now 3-d instead of 2-d.

print(tf.constant(t5).shape) # Output (1, 2, 3)
print(tf.constant(t6).shape) # Output (1, 2, 3)
print(tf.concat([t5, t6], 1)) # Output [[[ 1  2  3]
#                                        [ 4  5  6]
#                                        [ 7  8  9]
#                                        [10 11 12]]]
# Notice that we have a shape of (1, 4, 3) this time. That is because we used an axis = 1 this time. As we know,
#by using the axis = 1 allows us to concatenate across our columns instead of rows.

# Because we have a 3-d, we can use axis = 2 and see what shape we get.

t7 = [[[1, 2, 3],
      [4, 5, 6]]]

t8 = [[[7, 8, 9],
      [10, 11, 12]]]
# Note: We added extra brackets to our tensors to indicate that they are now 3-d instead of 2-d.

print(tf.constant(t7).shape) # Output (1, 2, 3)
print(tf.constant(t8).shape) # Output (1, 2, 3)
print(tf.concat([t7, t8], 2)) # Output [[[ 1  2  3  7  8  9]
#                                        [ 4  5  6 10 11 12]]], shape=(1, 2, 6)
# Notice that we now have a shape of (1, 2, 6). That is because we used the axis = 2 inside of our tf.concat 
#method.

# Here we will be looking at the tf.stack method. This method stacks a list of tensors into one tensor

tf.stack([t3, t4], axis = 0)
print(tf.stack([t1, t2], axis = 0)) # Output [[[ 1  2  3]
#                                              [ 4  5  6]]

#                                             [[ 7  8  9]
#                                              [10 11 12]]]
# Notice that we now have a similar shape to our concatenation shape. Our two (1, 2, 3)'s, we now have
#them stacked into one tensor with the shape of (2, 2, 3). The difference between the two is that when
#concatenated get returned with a shape of (4,3) as opposed to our stack method which returns a shape
#of (2, 2, 3).
# Basically, our tensors keep their shape when they get stacked, unlike the concatenation when the get
#combined into one shape.

# Here we will be looking at the tf.pad method. This method pads a tensor.

t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [2,2]])

tf.pad(t, paddings, "CONSTANT")
print(tf.pad(t, paddings, "CONSTANT")) # Output [[0 0 0 0 0 0 0]
#                                         [0 0 1 2 3 0 0]
#                                         [0 0 4 5 6 0 0]
#                                         [0 0 0 0 0 0 0]]
# Notice that our two tensors have padding around them. The padding is set to zero as default. We can change
#that and we will show an example of that next.

# Here we will change our padding number.
t = tf.constant([[7, 8, 9], [10, 11, 12]])
paddings = tf.constant([[1, 1,], [2,2]])

tf.pad(t, paddings, "CONSTANT", constant_values=3)
print(tf.pad(t, paddings, "CONSTANT", constant_values=3)) # Output [[ 3  3  3  3  3  3  3]
#                                                                   [ 3  3  7  8  9  3  3]
#                                                                   [ 3  3 10 11 12  3  3]
#                                                                   [ 3  3  3  3  3  3  3]]
# Notice that our padding now has a number of 3 instead of the default 0.
# Note: paddings = tf.constant([[1, 1,], [2,2]]) represents the set number of rows above, below, left and 
#right of the tensor. [1, 1] is going to be the top and bottom and [2, 2] is going to be the left  and right.
# We can also alter the this number by altering the numbers inside paddings = tf.constant([[1, 1,], [2,2]]).
# Let's look at an example of this below.

t = tf.constant([[13, 14, 15], [16, 17, 18]])
paddings = tf.constant([[3, 3,], [2,2]])

tf.pad(t, paddings, "CONSTANT", constant_values=3)
print(tf.pad(t, paddings, "CONSTANT", constant_values=3)) # Output [[ 3  3  3  3  3  3  3]
#                                                                   [ 3  3  3  3  3  3  3]
#                                                                   [ 3  3  3  3  3  3  3]
#                                                                   [ 3  3 13 14 15  3  3]
#                                                                   [ 3  3 16 17 18  3  3]
#                                                                   [ 3  3  3  3  3  3  3]
#                                                                   [ 3  3  3  3  3  3  3]
#                                                                   [ 3  3  3  3  3  3  3]]
# Notice that our padding between our tensors and the outside is now reflecting the numbers we added to the
#paddings = tf.constant([[3, 3,], [2,2]])
# Note: The numbers inside paddings = tf.constant([[3, 3,], [2,2]]) don't have to match. They can be any
#numbers we want them to be.

# Here we will be going the tf.gather method. This method gathers slices from parents axis according to
#indices.

params = tf.constant(["p0", "p1", "p2", "p3", "p4", "p5"])
params[0:3+1]
print(params[0:3+1]) # Output tf.Tensor([b'p0' b'p1' b'p2' b'p3']
# Notice that we are returned all the values specified in our params[0:3+1]. 0 is our starting location,
#3 is how many values we want to count up to, and + 1 adds one last value to the list. This gives us a total
#of 4 values.
# Note: we can also change our starting location. Let's look at the example below.

print(params[1:3+1]) # Output tf.Tensor([b'p1' b'p2' b'p3']
# Notice that we now start our return at the 1 location which is represented by the value "p1".
# Note: we can also specify locations in different ways. Let's look at an example below

print(tf.gather(params, [1,2,3])) # Output tf.Tensor([b'p1' b'p2' b'p3']
# Notice that we only get back the three values associated with the three locations we specified in our
#print(tf.gather(params, [1,2,3]))
# Note: When using this tf.gather method style, we must use a comma between params and our list of locations.
# There is also one other example we will go over. We will use a range() to specify the locations of the values
#we want to add to our return list. Let's look at an example of this below.

print(tf.gather(params, tf.range(1,4))) # Output tf.Tensor([b'p1' b'p2' b'p3']
# Notice that we are returned all the values associated with the locations specified in our
#print(tf.gather(params, tf.range(1,4)))
# Note: We are only returned the values between 1 and 4, not including 4. So that only leaves us with 3 values.

# Next we will go over the process of getting any values we want, not in order. 

print(tf.gather(params, [0,5,3])) # Output tf.Tensor([b'p0' b'p5' b'p3']
# Notice that we got our value for the 0 position [b"p0], and our value for our 5th location [b"p5], but
#then we reset back to 0 and get the value for our 3rd location [b"p3].

# Next we will perform our tensor slicing on more complex tensors.

params = tf.constant([[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]])
print(tf.gather(params, [3, 1])) # Output tf.Tensor(
#                                                   #[[30. 31. 32.]
#                                                     [10. 11. 12.]]
# Notice that we are only returned the 3 position row ([30, 31, 32]) and the 1 postion row ([10, 11, 12])
#as specified by our print(tf.gather(params, [3, 1]))

# In this example we want to get just the first row. Let's look at an example of this below.

params = tf.constant([[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]])
print(tf.gather(params, [0])) # Output tf.Tensor([[0. 1. 2.]]
# Notice that we are only returned one row as specified by our print(tf.gather(params, [0])). We are returned
#only the first row.

# Next, we will change our axis from its default 0 to a 1 to specify that we want target our column instead
#of our row. axis = 1 will target our column.

params = tf.constant([[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]])
print(tf.gather(params, [0], axis = 1)) # Output tf.Tensor(
#                                                          [[ 0.]
#                                                           [10.]
#                                                           [20.]
#                                                           [30.]]
# Notice that we have only returned the first column from our matrix. That is the first value from evrey 0
#position in our matrix.

# Next we will perform the same task but this time we only want to return our first row of values.
# We can do this by specifying [0], axis = 0. Let's see the example below

params = tf.constant([[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]])
print(tf.gather(params, [0], axis = 0)) # Output tf.Tensor([[0. 1. 2.]]
# Notice that we are only returned the 0 position row of values.

# In our next example we will target multiple rows at once. This will also work with our columns, but we will
#focus on the rows for now.

params = tf.constant([[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]])
print(tf.gather(params, [0,3], axis = 0)) # Output tf.Tensor(
#                                                            [[ 0.  1.  2.]
#                                                             [30. 31. 32.]]
# Notice that we are returned the values for the 0 position and the 3 position and 0 axis, as specified by our
#print(tf.gather(params, [0,3], axis = 0))

# Lastly, we will target multiple columns at once.

params = tf.constant([[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]])
print(tf.gather(params, [0,2], axis = 1)) # Output tf.Tensor(
#                                                            [[ 0.  2.]
#                                                             [10. 12.]
#                                                             [20. 22.]
#                                                             [30. 32.]]
# Notice that we are returned the values from our 0 position column and 2 position column on the 1 axis.

# Here we will use the tf.gather to target a 3-d tensor

params = tf.constant([
                     [[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]],

                     [[3, 1.0, 21],
                      [1, 3, 88],
                      [0, 5, 55],
                      [0, 2, 30]]
                     ])
print(params.shape) # (2, 4, 3)
print(tf.gather(params, [1, 0], axis = 0)) # Output tf.Tensor(
#                                                             [[[ 3.  1. 21.]
#                                                               [ 1.  3. 88.]
#                                                               [ 0.  5. 55.]
#                                                               [ 0.  2. 30.]]

#                                                              [[ 0.  1.  2.]
#                                                               [10. 11. 12.]
#                                                               [20. 21. 22.]
#                                                               [30. 31. 32.]]]
# Notice that we got back the bottom matrix first which is specified by our params [1] position being entered
#first, and our top matrix second as specified by our params[0] position being entered second. Also notice that 
#we got back all of the rows as specified by our axis = 0
# Note: The params specify which matricies we want to target and in what order.

params = tf.constant([
                     [[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]],

                     [[3, 1.0, 21],
                      [1, 3, 88],
                      [0, 5, 55],
                      [0, 2, 30]]
                     ])
print(params.shape) # (2, 4, 3)
print(tf.gather(params, [1, 0], axis = 1)) # Output tf.Tensor(
#                                                             [[[10. 11. 12.]
#                                                               [ 0.  1.  2.]]

#                                                              [[ 1.  3. 88.]
#                                                               [ 3.  1. 21.]]]
# Notice that we have a slightly differnt shape this time around. That is because we changed the axis = 0
#into an axis = 1. We still have our params set to [1, 0], but since we are now targeting columns, they
#have a different effect. Now we get back the row at the [1] position for each matrix first, and return
#that entire row, then we get the row at position [0] and return that whole row.
# Note: The params specify which matricies we want to target and in what order.

# Here we will go over the tf.gather_nd method. This method will gather slices from params into a Tensor
#with shape specified by indices.
# Note: The tf.gather_nd doesn't take any axis arguments.

indices = [[0],
           [1]]
params = [["a", "b"],
          ["c", "d"]]
print(tf.gather_nd(params, indices)) # Output tf.Tensor(
#                                                       [[b'a' b'b']
#                                                        [b'c' b'd']]
# Notice that we are returned a tensor in the shape specified by our indices. The [0] position [a, b] is 
#printed first, and the [1] position [c, d] is printed second.

# Next we will use three params in an example.

indices = [[2]]
params = [["a", "b"],
          ["c", "d"],
          ["e", "f"]]
print(tf.gather_nd(params, indices)) # Output tf.Tensor([[b'e' b'f']]
# Notice that we are only returned the tensor at the [2] position ["e", "f"] as specified by our indices.

# We will look at another example with three params, but with two indices this time

indices = [[2, 1]]
params = [["a", "b"],
          ["c", "d"],
          ["e", "f"]]
print(tf.gather_nd(params, indices)) # Output
# Notice that we are only returned the value of the tensor at position 2, column 1 ["f"] as specified by our indices.
# Since both numbers are inside the same index, it is specifying that we want to target row [2] and column [1].

# Next we will use more complex indices to shape a 3-d tensor.

indices = [[0, 1], [1, 0]]

params = [[["a0", "b0"],
           ["c0", "d0"]],
           
           [["a1", "b1"],
            ["c1", "d1"]]]
print(tf.gather_nd(params, indices)) # Output tf.Tensor(
#                                                       [[b'c0' b'd0']
#                                                        [b'a1' b'b1']]
# Notice that we are returned the first element and the second row as our first value ["c0", "d0"] as specified
#by our indices [0, 1], and our second value returned ["a1", "b1"] is from the second elements first row as
#specified by our indices [1, 0].