import pickle
import helper
import tensorflow as tf
import numpy as np
import sklearn.preprocessing
import sklearn.metrics


def cnn_model_fn(features, keep_prob):

	# CCN variables
	conv_ksize = [5, 5]
	pool_ksize = [2,2]
	pool_strides = 2
	num_outputs = 128
	n_nodes = 7 * 7 * 36
	dropout_rate = 0.3
	optimizer_learning_rate = 0.001
	
	conv1_filters = 14 
	conv2_filters = 36
	conv3_filters = 36 
	
	keep_prob
	
	# Convolutional Layer 1
	conv1 = tf.layers.conv2d(inputs=features, filters=conv1_filters, kernel_size=conv_ksize, padding="same", activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_ksize, strides=pool_strides)

	# Convolutional Layer 2
	conv2 = tf.layers.conv2d(inputs=pool1, filters=conv2_filters, kernel_size=conv_ksize, padding="same", activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool_ksize, strides = pool_strides)
	
	# Convolutional Layer 3
	#conv3 = tf.layers.conv2d(inputs=pool2, filters=conv3_filters, kernel_size=conv_ksize, padding="same", activation=tf.nn.relu)
	#pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=pool_ksize, stridespool_strides)
	
	# Fully connected layer, using 1764 neurins
	#pool2_flat = tf.reshape(pool2, [-1, n_nodes])
	pool2_flat = tf.contrib.layers.fully_connected(inputs = pool2, num_outputs=num_outputs, activation_fn=tf.nn.relu)
	
	pool2_fully = tf.contrib.layers.fully_connected(inputs = pool2_flat, num_outputs=num_outputs, activation_fn=tf.nn.relu)
	
	#dense = tf.layers.dense(inputs=pool2_flat, units=n_nodes, activation=tf.nn.relu)
	
	dropout = tf.nn.dropout(pool2_fully, keep_prob)
	
	output = tf.contrib.layers.fully_connected(inputs=dropout, num_outputs=num_outputs, activation_fn=None)
	
	dropout2 = tf.nn.dropout(output, keep_prob)
	
	# Logits Layer
	logits = tf.identity(dropout, name='logits')
	
	return logits
	
##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Load the training and test data from the Pickle file
with open("fingers_train_data_unscaled.pickle", "rb") as f:
      train_data, train_labels= pickle.load(f)
	  
print("Shape of traing data: ", train_data.shape)
print("Shape of traing label: ", train_labels.shape)

# Scale the training and test data
#pixel_mean = np.mean(train_data)
#pixel_std = np.std(train_data)
#train_data = (train_data - pixel_mean) / pixel_std
#test_data = (test_data - pixel_mean) / pixel_std

# One-hot encode the labels
encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')
train_labels_onehot = encoder.fit_transform(train_labels.reshape(-1, 1))
num_classes = len(encoder.categories_[0])

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Image data
image_size_row = 128
image_size_col = 128
image_channel = 3
image_shape = [image_size_row, image_size_col, image_channel]

# Inputs
x = tf.placeholder(tf.float32, name = 'x',  shape=(None,image_shape[0],image_shape[1],image_shape[2]))
y = tf.placeholder(tf.float32, name = 'y',  shape=[None, num_classes])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Training constants
batch_size = 32
learning_rate = .03
n_epochs = 5
eval_step = 5

n_batches = int(np.ceil(nsamples / batch_size))

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {} Learning rate: {}".format(batch_size, n_batches, n_epochs, learning_rate))

# Input vector placeholders. Length is unspecified.
x = tf.placeholder(tf.float32, name = 'x', shape=(None,image_shape[0],image_shape[1],image_shape[2]))
y = tf.placeholder(tf.float32, name = 'y', shape=(None, num_classes))


# -- MODEL -- return logits layer
cnn_model = cnn_model_fn(x, keep_prob)
	
# Loss and Optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn_model, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Accuracy
correct_pred = tf.equal(tf.argmax(cnn_model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

##############################
## Traning the Neural Network ##
##############################
# Create TensorFlow session and initialize it

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print('\n\n ------------------------- \n Starting training... \n')

epoch = 0
while epoch < n_epochs:
	batch = 0

    # Save a vector of cost values per batch
	cost_vals = np.zeros(n_batches)
	while batch < n_batches:
		# Select the data for the next batch
		dataidx = batch * batch_size
		X_batch = train_data[dataidx:(dataidx+batch_size)]
		Y_batch = train_labels_onehot[dataidx:(dataidx+batch_size)]
		print(X_batch.shape)
		X_batch = np.reshape(X_batch, [-1, image_shape[0], image_shape[1], image_shape[2]])
		feed_dict = {x: X_batch, y: Y_batch, keep_prob: keep_prob}
		#print("Data between {:4d} and {:4d}".format(dataidx, (dataidx+batch_size)))
		# Run one iteration of the computation session to update coefficients
		sess.run(optimizer, feed_dict=feed_dict)
		#cost_vals[batch] = sess.run(cost, feed_dict={x: X_batch,y: Y_batch,keep_prob: 1.})
		batch += 1
		
	# Evaluate and print the results so far
	# Calculate epoch loss and accuracy
	loss = sess.run(cost, feed_dict={x: X_batch,y: Y_batch,keep_prob: 1.})
	train_acc = sess.run(accuracy, feed_dict={x: X_batch,y: Y_batch,keep_prob: 1.})
	print("Epoch: {:4d}	Trainig_cost: {:.5f}	Traing_Accuracy: {:.6f}".format(epoch, loss, train_acc))
	epoch += 1
	
# Save Model
save_model_path = './image_classification'
saver = tf.train.Saver()
save_path = saver.save(sess, save_model_path)
		

	




