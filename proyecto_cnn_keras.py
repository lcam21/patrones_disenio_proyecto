from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pickle
import numpy as np
import sklearn.preprocessing
import sklearn.metrics

from tensorflow.keras import datasets, layers, models

# Image data
image_size_row = 128
image_size_col = 128
image_channel = 3
image_shape = [image_size_row, image_size_col, image_channel]

# Load the training and test data from the Pickle file
with open("fingers_full_train_data_unscaled.pickle", "rb") as f:
      train_data, train_labels= pickle.load(f)

# Load the training and test data from the Pickle file
with open("fingers_full_test_data_unscaled.pickle", "rb") as f:
      test_data, test_labels= pickle.load(f)

encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')
#train_labels = encoder.fit_transform(train_labels.reshape(-1, 1))
#test_labels = encoder.fit_transform(test_labels.reshape(-1, 1))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))

model.summary()

print("Training...")
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

train_data = np.reshape(train_data, [-1, image_shape[0], image_shape[1], image_shape[2]])

model.fit(train_data, train_labels, epochs=4)

test_data = np.reshape(test_data, [-1, image_shape[0], image_shape[1], image_shape[2]])

print("\nTesting...")
test_loss, test_acc = model.evaluate(test_data, test_labels)

print("Accuracy: ", test_acc)

model.save("model_keras.h5")
