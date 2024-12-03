import tensorflow as tf
import numpy as np

# Here we will be going over Linear Algebra Operations

# Here we will be looking at the tf.linalg.matmul method. This method multiplies matrix a by matrix b, producing
#a * b.

x_1 = tf.constant([[1,2,0],
                  [3,5,-1]])
x_2 = tf.constant([[1,2,0],
                   [3,5,-1],
                   [4,5,6]])



tf.linalg.matmul(
    x_1, x_2, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False,
    output_type=None, name=None
)
print(tf.linalg.matmul(
    x_1, x_2, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False,
    output_type=None, name=None
)) # Output tf.Tensor([[  7  12  -2] [ 14  26 -11]], shape=(2, 3), dtype=int32)
# Note: Number of rows from first tensor must equal the number of columns in the second tensor in order for
#operation to work.
print(x_1 @ x_2) # This is another we can run the program. But note, the equation must be x_1 @ x_2 to let
#the program knjow we are multiplying matrices and not single elements.

# Here we will be looking at the method tf.transpose. This method transposes a, where a is a tensor.
# Permutes the dimensions according to the value of perm.

tf.transpose(x_1)
print(tf.transpose(x_1)) # Output tf.Tensor([[ 1  3]
#                                            [ 2  5]
#                                            [ 0 -1]], shape=(3, 2), dtype=int32)
# Notice that we are returned a 3 by 2 of all the elements from our x_1 tensor. This is returned as columns
# Note: The same thing applies to x_2

print(x_1) # Output tf.Tensor([[ 1  2  0]
#                              [ 3  5 -1]], shape=(2, 3), dtype=int32)
# Notice that we are returned a 2 by 3 as opposed to when we ran print(tf.transpose(x_1)), which returned
#2 by 3. This is returned as rows.
# Note: The same thing applies to x_2

# Here we will be looking at the tf.linalg.band_part method. This method copies a tensor setting everything
#outside a central band in each innermost matrix to zero.

tensor_two_d = tf.constant([[1,2,0],
                           [3,5,1],
                           [1,5,6],
                           [2,3,8]])

tf.linalg.band_part(tensor_two_d, 0, 0) # ==> Diagonal. This means we will return a diagonal matrix
tf.linalg.band_part(tensor_two_d, -1, 0) # ==> Lower triangular part. This will only return the lower triangle of 
#our matrix. The upper portion will return zeros
tf.linalg.band_part(tensor_two_d, 0, -1) # ==> Upper triangular part. This will only return the upper triangle of 
#our matrix. The lower portion will return zeros

print(tf.linalg.band_part(tensor_two_d, 0, 0)) # Output. tf.Tensor(
                                                                    #[[1 0 0]
                                                                    # [0 5 0]
                                                                    # [0 0 6]
                                                                    # [0 0 0]], shape=(4, 3), dtype=int32)
# Notice that we get our output based on our tensor_two_d plus our conditions given to determine in the number in
#our index is true or false. If the number is true, it gets returned in our output, if it is false, a zero gets
#returned.

print(tf.linalg.band_part(tensor_two_d, -1, 0)) # Output tf.Tensor(
                                                                    #[[1 0 0]
                                                                    # [3 5 0]
                                                                    # [1 5 6]
                                                                    # [2 3 8]], shape=(4, 3), dtype=int32)
# Notice that the upper portion of our matrix has returned zeros. This is because of (tensor_two_d, -1, 0)



print(tf.linalg.band_part(tensor_two_d, 0, -1)) # Output tf.Tensor(
                                                                    #[[1 2 0]
                                                                    # [0 5 1]
                                                                    # [0 0 6]
                                                                    # [0 0 0]], shape=(4, 3), dtype=int32)
# Notice that the lower portion of our matrix has returned zeros. This is because of (tensor_two_d, 0, -1)


# These are conditions given in the documentation (m-n <= lower) and (n-m <= upper)
# m is for the rows and n is for the columns.


# Notice that these two tensors have the same exact shape as our input. What happens is we have m first of all.
# We have to understand that m is for the rows and n is for the columns.
# These are the conditions we are going to use to define our two matrices, one is m - n the other is n - m.

tensor_two_d_m_n = tf.constant([[0, -1, -2],
                                [1, 0, -1],
                                [2, 1, 0],
                                [3, 2, 1]], dtype = tf.float32)
# Note: To get the number for each index, we subtract the row from the column (m - n) Example. 0 - 0 = 0.
#0 - 1 = -1. 0 - 2 = -2. This is the output for the first row in our first index. 

tensor_two_d_n_m = tf.constant([[0, 1, 2],
                                [-1, 0, 1],
                                [-2, -1, 0],
                                [-3, -2, -1]], dtype = tf.float32)
# Note: To get the number for each index, we subtract the column from the row (n - m) Example. 0 - 0 = 0.
#1 - 0 = 1. 2 - 0 = 2. This is the output for the first row in our first index.


# Here we will look at the tf.linalg.inv method. This method computes the inverse of one or more square
#invertible matrices of their adjoints (conjugate transposes).
# Note: To obtain the inverse of a matrix, that matrix must be a square matrix. That is the number of rows
#must be equal to the number of columns.

tensor_two_d1 = tf.constant([[1,2,0],
                            [3,5,1],
                            [1,5,6]], dtype = tf.float32)

tf.linalg.inv(tensor_two_d1)
print(tf.linalg.inv(tensor_two_d1)) # Output tf.Tensor(
                                                #[[-2.7777781   1.3333335  -0.22222227]
                                                # [ 1.8888892  -0.6666668   0.11111114]
                                                # [-1.1111113   0.3333334   0.1111111 ]], shape=(3, 3), dtype=float32)
# Notice that we are returned the inverese of each number in our matrix.

# Here we will look at the tf.einsum method. Tensor contraction over specified indices and outer product.

A = np.array([[2, 6, 5, 2],
              [2, -2, 2, 3],
              [1, 5, 4, 0]])

B = np.array([[2, 9, 0, 3, 0],
              [3, 6, 8, -2, 2],
              [1, 3, 5, 0, 1],
              [3, 0, 2, 0, 5]])
print(A.shape) # Output (3, 4) 3 by 4
print(B.shape) # Output (4, 5) 4 by 5
print("Matmul C =: \n")
print(np.matmul(A,B), "\n") # Output [[33 69 77 -6 27]
#                                     [ 9 12  0 10 13]
#                                     [21 51 60 -7 14]]
# Notice that using this matmul / matrix multiplication on A,B returns our output.

print("Einsum C =: \n")
print(np.einsum("ij,jk -> ik", A,B)) # Output [[33 69 77 -6 27]
#                                              [ 9 12  0 10 13]
#                                              [21 51 60 -7 14]]
# Notice that "ij,jk -> ik" can be intepreted as A interacting with B to produce C. The way we chose the letters
#to represent A and B is by the way they matched up with our matrices shapes, and what kind of operation
#we are dealing with. 
# Also notice that the output is the same as the output of matrix A.

# Note: Even though we have the same exact output from each matrices, it is important that we use both 
#matrices. That is because in many applications we find that working with the einsum operator is going 
#to be easier than working with the usual numpy functions.


# In our next example will be using the einsum operator to perform element-wise multiplication.

A = np.array([[2, 6, 5, 2],
              [2, -2, 2, 3],
              [1, 5, 4, 0]])

B = np.array([[2, 9, 0, 3],
              [3, 6, 8, -2],
              [1, 3, 5, 0],])
# Note: We have to ensure that both of our matrices have the same shape in order to perform multiplication on
#them.

print("Hardamond C =: \n")
print(A*B, "\n") # Output [[  4  54   0   6]
#                          [  6 -12  16  -6]
#                          [  1  15  20   0]]
# Notice that our returned matriix is the sum of matrix A and B. Each element from matrix A is multiplied by
#it's counter part in matrix B.

print("Einsum C =: \n")
print(np.einsum("ij,ij -> ij", A,B)) # Output [[  4  54   0   6]
#                                              [  6 -12  16  -6]
#                                              [  1  15  20   0]]
# Notice that we have the same exact output as the previous ouput. That's because "ij,ij -> ij" is intepreted
#as A and B produce C. Using A and B as our template the einsum will return the same matrix as the Hardamond.

# Here we will transpose our A matrix
A = np.array([[2, 6, 5, 2],
              [2, -2, 2, 3],
              [1, 5, 4, 0]])

print("Transposed A =: \n")
print(A.T, "\n") # Output [[ 2  2  1]
#                          [ 6 -2  5]
#                          [ 5  2  4]
#                          [ 2  3  0]]
# Notice that our matrix has been transposed. That means that it was changed from a 3 by 4 (3,4) to a 4 by 3
#(4,3). Column 1 is now row 1 and so on and so forth.
print("einsum Transposed A =: \n")
print(np.einsum("ij -> ji", A)) # Output [[ 2  2  1]
#                                         [ 6 -2  5]
#                                         [ 5  2  4]
#                                         [ 2  3  0]]
# Notice that we the same exact output as our transposed output. That is because the einsum uses matrix A
#as a template and therefor produces the same output.


# Here we will begin working with 3 dimensional arrays.
   
# When data is placed in batches (example matrix 1 inside A np.array), all we need to do is ensure that each
#element in the batch multiplies the corresponding element in the batch in the other array.

A = np.array([
    [[2, 6, 5, 2],
    [2, -2, 2, 3],
    [1, 5, 4, 0]],

    [[1, 3, 1, 22],
     [0, 2, 2, 0],
     [1, 5, 4, 1]]
])

B = np.array([
    [[2, 9, 0, 3, 0],
     [3, 6, 8, 2, 2],
     [1, 3, 5, 0, 1],
     [3, 0, 2, 0, 5]],

    [[1, 0, 0, 3, 0],
     [3, 0, 4, -2, 2],
     [1, 0, 2, 0, 0],
     [3, 0, 1, 1, 0]]
])

print("Batch Multiplication C =: \n")
print(np.matmul(A,B), "\n") # Output [[[33 69 77 18 27]
#                                      [ 9 12  0  2 13]
#                                      [21 51 60 13 14]]

#                                     [[77  0 36 19  6]
#                                      [ 8  0 12 -4  4]
#                                      [23  0 29 -6 10]]]

# Notice that we get our output values by multiplying our row elements in matrix A by our column elements in
#matrix B, starting at position 0, then adding that to the results of the same process for position 1, and so
#forth and so on. Example matrix A = 2 multiplied by matrix B = 2 = 4 added to matrix A = 6 multiplied by 
#matrix B = 3. That will give us a sum of 22 with just those 4 elements.

print("einsum C =: \n")
print(np.einsum("bij,bjk -> bik", A,B)) # Output [[[33 69 77 18 27]
#                                                  [ 9 12  0  2 13]
#                                                  [21 51 60 13 14]]

#                                                 [[77  0 36 19  6]
#                                                  [ 8  0 12 -4  4]
#                                                  [23  0 29 -6 10]]]

# Here we will go over the process of summing up all of the elements in a given array

A = np.array([
    [[2, 6, 5, 2],
    [2, -2, 2, 3],
    [1, 5, 4, 0]],

    [[1, 3, 1, 22],
     [0, 2, 2, 0],
     [1, 5, 4, 1]]
])

print("Sum A =: \n")
print(np.sum(A), "\n") # Output. We are returned 72 after  we summed up all of the elements in both matrices
# Notice that we have an output of 72. That is because we added all of the elements in both matrices

print("Einsum A =: \n")
print(np.einsum("bij -> ", A )) # Output We are returned 72 after  we summed up all of the elements in both 
#matrices
# Notice that we have an output exactly the same as Sum A. That is beacuse the einsum is using the A matrix as
#a template and as a result is returning the same output.
# Also notice the we have a shape of bij with an arrow pointing to an empty output, all we're basically doing
#is summing up all the different possible values. That's what the arrow pointing to the empty output signifies.
# Note: It is also important to make sure we have the proper shape.

# We can also sum up all elements in a given dimension or given row or given column.

A = np.array([[2, 6, 5, 2],
             [2, -2, 2, 3],
             [1, 5, 4, 0]])

print("Axis 0 Sum A =: \n")
print(np.sum(A, axis = 0), "\n") # Output [5, 9, 11, 5]
# Notice that we summed up the elements in our columns separately

print("Einsum A =: \n")
print(np.einsum("ij -> j", A)) # Output [5, 9, 11, 5]
# Notice that we summed up the elements in our columns separately
# Note: j equals our columns in this instance
print("------------------------------------")

print("Axis 1 Sum A =: \n")
print(np.sum(A, axis = 1), "\n") # Output [15, 5, 10]
# Notice that we summed up the elements from our rows separately

print("Einsum A =: \n")
print(np.einsum("ij -> i", A)) # Output [15, 5, 10]
# Notice that we summed up the elements from our rows separately
# Note: i equals our rows in this instance

# Now moving to a more practical example.
# We will be working with this formula. 

# Q = batchsize, s_q, modelsize
# K = batchsize, s_k, modelsize
# This is a query of a transpose multiplied by a key

# If this key (K) has a shape of batchsize by sequence length of the key (s_k) by modelsize by sequence
# And the query (Q) shaped by batchsize by sequence length of the query (s_q) by modelsize, then 
#we can define this non-py array query of shape 32 by 64 by 512 and key 32 by 128 by 512.  
# And then what we will do now is define the query by transpose operation where we would have np einsum.
# This is considered to be the batchsize by query by M (bqm)
# This is considered to be the batchsize by key by M (bkm)
# That will give us bqm comma bkm which outputs b (bqm, bkm -> b)

# At this point we need to be careful we have query M times key M transpose. So this becomes 
#B query M times B MK and now when we multiply this query M and MK what we have is query K
# That gives us B query K
# bqm * bmk (transposed version of bkm). 
# Because the M cancels each other out we are left with bqk

Q = np.random.randn(32, 64, 512) # This is the shape bqm
K = np.random.randn(32, 128, 512) # This is the shape bkm

np.einsum("bqm, bkm -> bqk", Q,K)
print(np.einsum("bqm, bkm -> bqk", Q,K)) # Output [[[ 6.20045767e+00 -2.73767376e+01 -2.15562689e+01 ... -7.16929661e+00
#                                                                     2.18578692e+01  1.79423655e+01]
#                                                   [ 1.58776265e+01 -1.46064055e+01 -2.13104574e+01 ...  4.32902429e+01
#                                                                    -3.46358273e+01  7.78696965e+00]
#                                                   [ 5.42257481e-01  3.76806672e+00 -1.18611669e+01 ...  1.31028893e+01
#                                                                    -1.89969739e+01 -3.73652424e-01]
#                                                   ...
#                                                   [ 9.55777440e+00  3.67705933e+01  1.86692398e+01 ...  1.79046059e+01
#                                                                    -2.49285634e+01 -6.87017649e+00]
#                                                   [ 3.85504265e+01  3.35675311e+01  3.89958475e+01 ... -4.76688741e+00
#                                                                     1.80743584e+01 -3.64258146e+01]
#                                                   [-6.17248137e-01  1.47024657e+01 -1.42962112e+01 ... -3.29092926e+01
#                                                                     1.48775465e+01 -2.37408779e+01]]

#                                                  [[-3.62734817e+01  1.89614714e+01  1.04925836e+01 ...  1.16382264e+01
#                                                                     1.02622978e+01  2.50613717e+00]
#                                                   [ 4.22464250e+00  4.53490386e+00  1.29438916e+01 ...  7.66715385e+00
#                                                                     9.10112964e+00 -5.46446416e+00]
#                                                   [-2.66562871e+00 -5.46709653e+00  5.27514539e+00 ...  2.24432076e+01
#                                                                    -2.95006318e+00 -5.09270241e+00]
#                                                   ...
#                                                   [-1.96975185e+01  2.77820581e+01  1.35766832e+01 ...  1.52766384e+01
#                                                                     3.80332994e+01  1.25052606e+01]
#                                                   [ 8.94501236e+00  1.24079474e+01  1.35519607e+01 ... -4.72319627e+00
#                                                                    -1.41112418e+01  1.01465688e+01]
#                                                   [ 4.64266404e+00 -1.03388355e+01 -7.93055898e+00 ...  2.19335020e+00
#                                                                    -1.74483398e+01 -3.31383228e+01]]

#                                                  [[ 1.64597940e+01 -1.58839071e+01  4.30717179e+00 ...  9.64222688e+00
#                                                                    -1.25633105e+01  2.56655798e+01]
#                                                   [ 5.99564721e+01  7.52261026e-01 -4.92525990e+00 ... -7.54613732e+00
#                                                                    -2.60604641e+01 -4.80069603e+00]
#                                                   [-4.48426445e+01  9.96207981e+00  1.02167616e+01 ...  1.40467640e+01
#                                                                     1.54377886e+01 -1.30684602e+01]
#                                                   ...
#                                                   [ 6.22488486e+00 -1.15581853e+01  9.76300170e+00 ... -8.42219739e+00
#                                                                    -6.46267224e+00  3.72475256e+00]
#                                                   [ 3.14995967e+01  2.84252734e+01 -5.62148331e+01 ... -3.10833886e+01
#                                                                    -2.92039232e+01 -2.16863464e+01]
#                                                   [ 2.35917074e+00 -4.20971658e+00  7.38743325e+00 ... -3.08840367e+00
#                                                                     5.93984032e+01 -1.34681365e+01]]

#                                                   ...

#                                                  [[-1.41636036e+01  1.81320173e+01 -1.66871665e+01 ... -5.71317874e+00
#                                                                     2.78395031e+01 -1.20728979e+01]
#                                                   [-2.31336042e+01  4.10676196e+01 -6.80348876e+00 ...  1.99523636e+01
#                                                                     1.68675344e+01  6.73831223e+00]
#                                                   [ 1.80933313e+00  1.68183179e+00  3.49237664e+01 ...  9.47904491e+00
#                                                                     4.42671678e+00  1.68920729e+01]
#                                                   ...
#                                                   [ 1.00241075e+01  1.68156387e+01  4.46974958e+01 ... -4.41050893e+01
#                                                                     2.18921773e+00 -2.19196974e+01]
#                                                   [-4.53949683e+00  1.46152891e+01 -1.34681201e+01 ... -5.97144861e+00
#                                                                    -4.05595561e+01  5.06634337e-01]
#                                                   [-3.39727507e-01 -1.71263041e+01 -3.15607962e+00 ... -2.89280700e+01
#                                                                     1.04441495e+01 -2.96647654e+01]]

#                                                  [[-3.04396967e+01 -2.59925559e+01  6.29447987e+00 ... -4.80455610e+00
#                                                                    -2.03043171e+00 -2.54607315e+01]
#                                                   [-2.31108847e+01  4.70610583e+00 -9.47370129e+00 ...  3.72443890e+01
#                                                                    -1.01934932e+01  1.25566267e+01]
#                                                   [ 3.25588552e+00 -1.90825740e+01  3.17361342e+01 ...  3.59390905e+00
#                                                                    -1.41828871e+01  5.25371041e+01]
#                                                   ...
#                                                   [ 3.37108076e-02  1.56932509e+01 -2.90445047e+01 ...  2.48124065e+01
#                                                                    -1.54792647e+01 -1.42045737e+01]
#                                                   [-3.15411593e+01 -2.31036312e+01  9.18555059e+00 ...  5.91585309e+01
#                                                                     4.57234134e+00  3.42899135e+01]
#                                                   [-9.36254590e+00  2.92105032e+01 -3.57688174e+01 ...  5.70172433e+01
#                                                                     1.60355402e+01 -2.17380822e+01]]

#                                                  [[-2.11311535e+01 -2.86947515e+01  1.84385963e+01 ...  1.07406694e+01
#                                                                     1.47739914e+01 -2.06361303e+01]
#                                                   [ 2.41188665e+01 -2.24462460e+01 -1.00245682e+01 ...  2.76661660e+01
#                                                                    -6.55116853e+00  7.32462800e+00]
#                                                   [-4.64803445e+01 -9.34864284e+00 -1.63600948e+01 ... -6.46497149e+00
#                                                                    -3.44469935e+01  7.91981536e+00]
#                                                   ...
#                                                   [ 4.68795274e+01 -2.22451215e+01  1.62851083e+01 ... -2.89021603e+01
#                                                                     1.04521941e+01 -1.19887603e+01]
#                                                   [-9.66980048e+00 -2.22839548e+01  1.51719048e+01 ...  5.71163386e+01
#                                                                     3.76884608e+01 -3.63423367e+01]
#                                                   [ 3.90689286e+01 -1.37127895e+01  2.77319896e+01 ... -1.33944341e+01
#                                                                     7.72276497e+00 -7.59935043e+00]]]
# Notice that this is the shape of the output we returned (32, 64, 128)
# We get this shape because of our output shape b = 32, q = 64, k = 128


# Here we will be looking at how data can be broken up into chunks.

A = np.random.randn(2, 4, 4, 2) # This shape is bcij
B = np.random.randn(2, 4, 4, 1) # This shape is bcik

# What we want to find is B transposed by A.
# So what we are going to do here is np einsum of bcik.

np.einsum("bcik, bcij -> bckj", B, A) # bckj will be our output shape because we transposed bcik into bcki and 
#multiplied it by bcij. That eliminated the i from both shapes and left us with bckj

# Here we will ensure that we transposed Matrix B correctly
np.matmul(np.transpose(B, (0,1,3,2)), A).shape # We want to ensure have 0,1 fixed and then we have 3,2 because this is why
#we do the transposing. And now we pass in A. 

print(np.einsum("bcik, bcij -> bckj", B, A)) # Output        [[[[-1.5744667  -0.45779124]]

                                                            #  [[ 3.43430533 -2.66062505]]

                                                            #  [[-4.79551613 -0.31138478]]

                                                            #  [[ 1.24832531 -0.56024954]]]


                                                            # [[[-1.09291859  0.38789622]]

                                                            #  [[ 2.93358291  2.36435584]]

                                                            #  [[ 1.73425002 -0.22925511]]

                                                            #  [[-0.06080619  1.60122822]]]]

# Notice That our information is returned chunks that easier to read. There is also another way to perform 
#this task

print(np.matmul(np.transpose(B, (0,1,3,2)), A)) # Output (2, 4, 1, 2)
# Notice that we will back the same shape as the einsum. That is because it performing the same task and 
#producing the same output. The einsum is cleaner and more efficient in this case so we tend to use that more.
# Also notice that 2, 4, 1, 2 represents our output shape of bckj. b = 2, c = 4, k = 1, j = 2