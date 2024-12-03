import tensorflow as tf
# Here we will be going over Variables.

# We need to use variables which can be updated as we do model training.

x = tf.constant([1, 2])

x_var = tf.Variable(x) # Note: Variables must always be initialized
print(x_var) # Output <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([1, 2])>
# Notice that our return is all the information about our input.

# Here is another example.

x_var = tf.Variable(x, name = "var1")
print(x_var) # Output <tf.Variable 'var1:0' shape=(2,) dtype=int32, numpy=array([1, 2])>
# Notice that our return now has a name of "var1", as specified by the parameters we entered.

# Here is how we create a tensorflow variable

#tf.Variable(
#    initial_value=None, trainable=None, validate_shape=True, caching_device=None,
#    name=None, variable_def=None, dtype=None, import_scope=None, constraint=None,
#    synchronization=tf.VariableSynchronization.AUTO,
#    aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None
#)
# This is just a template. We will need to add specifications when we use this method in real-time.

# Here we will be going over the process in which we choose which device we want our variable to run on.
# These are our options cpu, gpu, tpu
# This is how we specify


with tf.device("GPU:0"):
    x_var = tf.Variable(0.2)

print(x_var.device) # Output /job:localhost/replica:0/task:0/device:CPU:0
# Notice that we are targeting our CPU as specified by the parameters we entered.

# Here we will go over the process of carrying out computations in our CPU.

with tf.device("CPU:0"):
    x_1 = tf.constant([1,3,4])
    x_2 = tf.constant([1])

with tf.device("CPU:0"):
    x_3 = x_1 + x_2

print(x_1) # Output tf.Tensor([1 3 4]
# Notice that we are returned our 3 element tensor

print(x_2) # Output tf.Tensor([1]
# Notice we are returned our 1 element tensor

print(x_3) # Output tf.Tensor([2 4 5]
# Notice that we are returned the sum of x_1 and x_2