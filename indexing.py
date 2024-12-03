# In this section we will be going over Indexing

import tensorflow as tf

tensor_indexed = tf.constant([3,6,2,4,6,66,7])
print(tensor_indexed) # This will return the full index of numbers

print(tensor_indexed[0:4]) # This will return the target numbers we select.
# Notice that we use 0 as our starting point, and the 4 tells the program how many numbers we want to return,
#starting from that point.

print(tensor_indexed[1:5+1]) # This will give us a starting point at 6 and stop at 6, the + 1 will add 66
#to our return list, giving us a list of 6,2,4,6,66 print(tensor_indexed[1:6]) will give us the same result.

print(tensor_indexed[1:6:2]) # This will give us a list of numbers from 6 throuh 66, but only counting every
#second number, returning 6, 4, 66

print(tensor_indexed[:4]) # This will return the same list as 0:4, since the blank space is treated as a the 
#zero location.

print(tensor_indexed[3:]) # This will return every number in our list starting from the 3 position. Since no end
#point is specified, al the numbers after the starting point will be returned.

print(tensor_indexed[3:-1]) # This will return every number in the list after the starting point of the third 
#position, except the last number. Using -1 will start to count backwards after the entire list is counted. Meaning 
#that our list will be returned as 4, 6, 66.

print(tensor_indexed[3:-2]) # This is the same principle as using the -1, but this time the last two numbers will
#be cut as specified by the -2.

# From this point, we will see how indexing is done on tensors with dimensions greater than 2.

tensor_two_d = tf.constant([[1,2,0],
                           [3,5,1],
                           [1,5,6],
                           [2,3,8]])
print(tensor_two_d[0:3,0:2]) # This will return [[1,2][3,5][1,5]]. This happens because the numbers to the left
#specify the rows we want to target([0:3,]) and the numbers to the right specify the columns we want to target
#([0:2])

print(tensor_two_d[0:3, :]) # This will return all the numbers in the 3 rows we have target since no specific
#column locations were chosen.

print(tensor_two_d[2,:]) # This will return the second row, with all of its numbers. This helps us target specific
#location that may not be at the beginning of a row

print(tensor_two_d[2,0]) # This will return selected numbers in a row we choose to target. Notice that we didn't
#use a colon for this step.

print(tensor_two_d[2,1:]) # This will return every number in the row after the starting point.

print(tensor_two_d[:,0]) # This will return every number in the zero column. This happens because we didin't
#specify any starting or ending location for our rows, so we'll get all the rows, but, since we specified that
#we only want the zero coloumn of each row, we will return an entire list of just zero column numbers.

print(tensor_two_d[1:3,0]) # This will return rows 1 and 2 as specified by the starting and ending location
#of 1:3, and we will only return the numbers in the zero column of those rows.

print(tensor_two_d[:,1]) # This will return every number in the one column. This happens because we didin't
#specify any starting or ending location for our rows, so we'll get all the rows, but, since we specified that
#we only want the one coloumn of each row, we will return an entire list of just one column numbers.

print(tensor_two_d[...,1]) # This is another way writing (tensor_two_d[:,1]) and will have the same exact
#output. [...] specifies that we want to access all of our rows.


# Next we will look at indexing 3d tensors.

tensor_three_d = tf.constant([[[1,2,0],
                               [3,5,-1]],
                               
                               [[10,2,0],
                                [1,0,2]],

                                [[5,8,0],
                                 [2,7,0]],
                                 
                                 [[2,1,9],
                                  [4,-3,32]]])
print(tensor_three_d[ 0, : , : ]) # Note that we have 2 commas this time. The three spaces that occur because of the commas
#represent our 3d. This will return only the first tensor because of the 0 starting location, which will return
#[[1,2,0],[3,5,1]] everything in those tensors because no ending location was specified. Also, no other rows,
#columns, or tensors are specified.

print(tensor_three_d[ 0, 0, :]) # This will return only the first row in the first tensor as specified by 
#locations we entered.

print(tensor_three_d[ 0, : , -1]) # This will return the last coloumn of all the rows in our first tensor.

print(tensor_three_d[ 0, : , 2]) # This is the same as writing [ 0, : , -1] and return the same exact output.

print(tensor_three_d[0:2, : , 2]) # This will return all the numbers in the last column of our first 2 tensors.
# This happens because we targeted the first 2 tensors with [0:2], and we targeted every row of those tensors
#with [ : ]. And lastly we only chose the numbers of the second column by using [, 2]

print(tensor_three_d[0:2, ... , 2]) # This is the as writing [0:2, : , 2] and has the same exact output.
# [ ...] means we want to access all of whatever location we put it.

print(tensor_three_d[..., : , 2]) # This will give us access to all the tensors and all the rows, but we only
#want to chose the numbers in the second column