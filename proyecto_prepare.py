#GET DATA

import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_data_and_labels(images_path, num_of_rows, num_of_colums):
	try:
		num_of_items = len(os.listdir(images_path))
		num_of_image_values = num_of_rows * num_of_colums
		
		print("Number of images: ", num_of_items)
		print("Size of images: ", num_of_rows, "by", num_of_colums)
		
		labels_name_files = ["0R","1R","2R","3R","4R","5R","0L","1L","2L","3L","4L","5L"]
		
		# XX -> hand number
		# 1X -> rigth number
		# 2X -> left number
		labels_definition = [10,11,12,13,14,15,20,21,22,23,24,25,0]
		
		data = [[None for x in range(num_of_image_values)]
                for y in range(num_of_items)]
		
		labels = []
		
		num_of_items = 0
		for filename in os.listdir(images_path):
		
			if (num_of_items % 1000) == 0:      
				print("Current image number: %7d" % num_of_items)
			img = mpimg.imread(images_path + "/" + filename)
			data[num_of_items] = np.reshape(np.array(img),(16384))
			
			if (filename.find(labels_name_files[0])>-1):
				labels.append(labels_definition[0])
			elif (filename.find(labels_name_files[1])>-1):
				labels.append(labels_definition[1])
			elif (filename.find(labels_name_files[2])>-1):
				labels.append(labels_definition[2])
			elif (filename.find(labels_name_files[3])>-1):
				labels.append(labels_definition[3])
			elif (filename.find(labels_name_files[4])>-1):
				labels.append(labels_definition[4])
			elif (filename.find(labels_name_files[5])>-1):
				labels.append(labels_definition[5])
			elif (filename.find(labels_name_files[6])>-1):
				labels.append(labels_definition[6])
			elif (filename.find(labels_name_files[7])>-1):
				labels.append(labels_definition[7])
			elif (filename.find(labels_name_files[8])>-1):
				labels.append(labels_definition[8])
			elif (filename.find(labels_name_files[9])>-1):
				labels.append(labels_definition[9])
			elif (filename.find(labels_name_files[10])>-1):
				labels.append(labels_definition[10])
			elif (filename.find(labels_name_files[11])>-1):
				labels.append(labels_definition[11])
			else:
				print("Image " + filename + " doesn't have label, id: ", num_of_items)
				labels.append(labels_definition[12])

			num_of_items = num_of_items + 1
			
		labels = np.array(labels)
		data = np.array(data)
		return data, labels

	finally:
			print("Completed data reading")
			
def save_data_and_labels(data_file_name, labels_file_name, data_array, labels_array):
	try:
		np.savetxt(data_file_name + ".csv", data_array, delimiter=",")
		np.savetxt(labels_file_name + ".csv", labels_array, delimiter=",")
		
	finally:
			print("Saved files")



print("Reading training dataset")
train_data, train_labels = get_data_and_labels("train", 128, 128)
train_size = train_data.shape[0]

print("Reading test dataset")
test_data, test_labels = get_data_and_labels("test", 128, 128)
test_size = test_data.shape[0]

'''
# Plot a histogram of pixel values before normalization
print("Histogram of pixel values before normalization")
hist, bins = np.histogram(train_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Train dataset")
plt.show()
'''

# Compute the mean and stddev
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
print("Pixel mean:", pixel_mean)
print("Pixel stddev:", pixel_std)

# Scale the training and test data
train_data_scaled = (train_data - pixel_mean) / pixel_std
test_data_scaled = (test_data - pixel_mean) / pixel_std

'''
# Plot a histogram of pixel values
print("Histogram of pixel values after normalization")
hist, bins = np.histogram(train_data_scaled, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Train dataset scaled")
plt.show()

hist, bins = np.histogram(test_data_scaled, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Test dataset scaled")
plt.show()
'''

# Archive the scaled training and test datasets into a pickle file
#with open("fingers_dataset.pickle", 'wb') as f:
#      pickle.dump([train_data_scaled, train_labels, test_data_scaled, test_labels], \
#                  f, pickle.HIGHEST_PROTOCOL)

# Archive the original set as uint8, to save space
print("Saving data and labels in  pickle file")
with open("fingers_dataset_unscaled.pickle", 'wb') as f:
      pickle.dump([train_data, train_labels, test_data, test_labels], \
                  f, pickle.HIGHEST_PROTOCOL)

#print("Saving test dataset")
#save_data_and_labels("test_data_file", "test_labels_file", test_data, test_labels)

#print("Saving traing dataset")
#save_data_and_labels("train_data_file", "train_labels_file", train_data, train_labels)

# Some information about the test dataset
#print("Test dataset size: ", test_data.shape)
#print("Class histogram: ")
#print(np.histogram(test_labels, 12)[0])

# Some information about the training dataset
#print("Training dataset size: ", train_data.shape)
#print("Class histogram: ")
#print(np.histogram(train_labels, 12)[0])



'''
# Showing 12 images as sample
for idx in range(12):
  image = test_data[idx].reshape(128,128)
  plt.figure()
  plt.imshow(image, cmap="gray_r")
  plt.title("Label: "+str(test_labels[idx]))
plt.show()
'''





