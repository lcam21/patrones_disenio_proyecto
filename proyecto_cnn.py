import pickle
import helper
import tensorflow as tf
import numpy as np

def neural_net_image_input(image_shape):

	#Return a Tensor for a bach of image input
	#image_shape: Shape of the images
	#return: Tensor for image input.
	# TODO: Implement Function
	x = tf.placeholder(tf.float32, name = 'x',  shape=(None,image_shape[0],image_shape[1],image_shape[2]))
	#x = tf.placeholder(tf.float32, name = 'x',  shape=(None,image_shape[0],image_shape[1]))
	#x = tf.placeholder(tf.float32, shape=(None, image_shape), name="x")
	return x

def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    y = tf.placeholder(tf.float32, name = 'y',  shape=[None, n_classes])
    return y

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return keep_prob

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    
    # Create the weight and bias using conv_ksize, conv_num_outputs and the shape of x_tensor.
    # print(x_tensor.get_shape()[3])
    # print((conv_ksize[0], conv_ksize[1], x_tensor.get_shape()[3].value,conv_num_outputs))
    
    weights = tf.Variable(tf.truncated_normal((conv_ksize[0], conv_ksize[1], x_tensor.get_shape()[3].value,conv_num_outputs), stddev=0.05))
    bias = tf.Variable(tf.truncated_normal([conv_num_outputs], dtype=tf.float32))
	
    # Apply a convolution to x_tensor using weight and conv_strides.
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
        
    output = tf.nn.conv2d(x_tensor, weights, [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    
    # Add bias
    # tf.nn.bias_add(value, bias, data_format=None, name=None)
    
    output = tf.nn.bias_add(output, bias)
    
    # Add a nonlinear activation to the convolution.
    # tf.nn.relu(features, name=None)
    
    output = tf.nn.relu(output)
    
    # Apply Max Pooling using pool_ksize and pool_strides.
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    
    output = tf.nn.max_pool(output, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1], padding='SAME')
        
    return output 	

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    return tf.contrib.layers.flatten(x_tensor)

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    return tf.contrib.layers.fully_connected(inputs = x_tensor, num_outputs=num_outputs, activation_fn=tf.nn.relu)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    return tf.contrib.layers.fully_connected(inputs=x_tensor, num_outputs=num_outputs, activation_fn=None)

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    # Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    
    num_outputs_fc = 128
    num_outputs_out = 12
    conv_num_outputs1 = 64
    conv_num_outputs2 = 128
    conv_num_outputs3 = 256
    conv_ksize = (5,5)
    conv_strides = (1,1)
    pool_ksize = (3,3) 
    pool_strides = (2,2)
    keep_prob
    
    x = conv2d_maxpool(x, conv_num_outputs1, conv_ksize, conv_strides, pool_ksize, pool_strides)    
    x = conv2d_maxpool(x, conv_num_outputs2, conv_ksize, conv_strides, pool_ksize, pool_strides)    
    x = conv2d_maxpool(x, conv_num_outputs3, conv_ksize, conv_strides, pool_ksize, pool_strides)    

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    
    x = flatten(x)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    
    x = fully_conn(x, num_outputs_fc)
    x = fully_conn(x, num_outputs_fc)
    
    # as recommended I'm adding dropout funtion:
    
    x = tf.nn.dropout(x, keep_prob)
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    
    #y = output(x, num_outputs_out)  # corrected to innclude dropout
    
    y = tf.nn.dropout(output(x, num_outputs_out), keep_prob)
    
    #x = conv2d_maxpool(x, 64, (5,5), (1,1), (3,3), (2,2))
    #x = conv2d_maxpool(x, 128, (5,5), (1,1), (3,3), (2,2))
    #x = conv2d_maxpool(x, 256, (5,5), (1,1), (3,3), (2,2))

    #x = flatten(x)

    #x = fully_conn(x, 128)
    #x = fully_conn(x, 128)

    #y = output(x, 10)

    
    # TODO: return output
    return y
	
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
	#Optimize the session on a batch of images and labels
	#: session: Current TensorFlow session
	#: optimizer: TensorFlow optimizer function
	#: keep_probability: keep probability
	#: feature_batch: Batch of Numpy image data
	#: label_batch: Batch of Numpy label data
	
	input_layer = np.reshape(feature_batch, [-1, 128, 128, 1])
    
	# TODO: Implement Function
	session.run(optimizer, feed_dict={x: input_layer, y: label_batch, keep_prob: keep_probability})
	pass

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    # Calculate batch loss and accuracy
    loss = session.run(cost, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.})
    
    # adding the missing part...
    
    train_acc = session.run(accuracy, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.})
    
    valid_acc = session.run(accuracy, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1.})

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
        loss, train_acc,
        valid_acc))
    pass
	
##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
#x = tf.placeholder(tf.float32, shape=(None, 16384), name="X")
x = neural_net_image_input((128,128,1))
y = neural_net_label_input(12)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')



# TODO: Tune Parameters
epochs = 20
batch_size = 32
keep_probability = 0.7

save_model_path = './image_classification'

print('')
print('Training...')
with tf.Session() as sess:
	# Initializing the variables
	sess.run(tf.global_variables_initializer())
	# Training cycle
	for epoch in range(epochs):
		batch_i = 1
		for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_size):
			train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
		print('Epoch {:>2}: '.format(epoch + 1), end='')
		print_stats(sess, batch_features, batch_labels, cost, accuracy)
		
	saver = tf.train.Saver()
	save_path = saver.save(sess, save_model_path)
