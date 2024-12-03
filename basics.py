# Tensor Basics - Tensors can be defined as multi dimensional arrays. An array is an ordered arrangement of 
#numbers.

# There are different types of arrays based on their dimensionality. Here are a few examples.

# 8 = 0 Dimensional array, which is simply because this array or this tensor contains a single element. so
#let's say we have 21. This is also a zero dimensionalarray. Let's say we will have a 1. This is a zero
#dimensional so on and so forth. 
# So essentially once we have a single element then it's a zero dimensional array.

#-----------------------------------------------------------------------------------------------------------------

# [2 0 -3] = 1 Dimensional array, which in fact is a combination of several 0 - D (Zero Dimensional) tensors.
# So if we look at this, we see that [2] is a 0 - D tensor. [0] is another 0 - D tensor. [-3] is the last 0 - D
#tensor in this array. So this vector is made of three kinds of elements. The lenght of the 1 - D array doesn't 
#change its status as a 1 - D array.

#-------------------------------------------------------------------------------------------------------------------

#[1 2 0] = 2 Dimensional array, which is essentially made of a combination of several 1 - D arrays. We can see
#[3 5 -1]  #that here. [1 2 0] is a 1 - D array. [3 5 -1] is a 1 - D array. [1 5 6] is a 1 - D array. [2 3 8] is
#[1 5 6]   #a 1 - D array. So when we combine these 1 - D tensors, we form this 2 - D tensor.
#[2 3 8]

#-------------------------------------------------------------------------------------------------------------------

#[1 2 0] = 3 Dimensional array, which essentially a combination of several 2 - D tensors. 
#[3 5 -1]

#[10 2 0]
#[1 0 2]

#[5 8 0]
#[2 7 0]

#[2 1 9]
#[4 -3 32]

#-------------------------------------------------------------------------------------------------------------------

# Now that we understand this we're going to take a look at the concept of tensor shapes.

# 8 - This 0 - D tensor has no shape.

# [2 0 -3] - This 1 - D tensor is of shape 3 because of its 3 elements.

#[1 2 0] - This 2 - D tensor has a shape of (4,3) 4 by 3. Because there are 4 1 - D tensors each with 3
#[3 5 -1]   #elements.
#[1 5 6]
#[2 3 8]

#[1 2 0] - This 3 - D tensor has a shape of (4,2,3) 4 by 2 by 3. Because there are 4 2 - D tensors with
#[3 5 -1]   #3 elements each

#[10 2 0]
#[1 0 2]

#[5 8 0]
#[2 7 0]

#[2 1 9]
#[4 -3 32]


# This is how we obtain our shapes.
# Notice that the number of elements we have here gives us information about the number of dimensionals our
#tensor has. So if we have a 1 - D tensor, our dimension is 1, because it's just [2 0 -3]. If we have a 
#2 - D tensor, our dimension is 2, because it has 4 tensors with 3 elements (4,3) = 2 - D. If we have a
#3 - D tensor, our dimension is 3, because it has 4 2 - D tensors with 3 elements (4,2,3) = 3 - D.

# Also in matrix notation we would count the number of rows and columns to determine the dimensions.
