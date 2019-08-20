
import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

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
			img = Image.open(images_path + "/" + filename)
			img = img.resize((num_of_rows, num_of_colums))
			#img = mpimg.imread(images_path + "/" + filename)
			array = np.array(img)
			data[num_of_items] = np.reshape(array,(num_of_image_values))
			
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
		print("Completed data resizing")
	
image_size = 32
print("Reading training dataset")
train_data, train_labels = get_data_and_labels("train", image_size, image_size)
train_size = train_data.shape[0]

print("Reading test dataset")
test_data, test_labels = get_data_and_labels("test", image_size, image_size)
test_size = test_data.shape[0]

# Showing 12 images as sample
for idx in range(5):
  image = test_data[idx].reshape(image_size,image_size)
  plt.figure()
  plt.imshow(image)
  plt.title("Label: "+str(test_labels[idx]))
plt.show()

# Compute the mean and stddev
pixel_mean = np.mean(train_data)
pixel_std = np.std(train_data)
print("Pixel mean:", pixel_mean)
print("Pixel stddev:", pixel_std)

# Scale the training and test data
train_data_scaled = (train_data - pixel_mean) / pixel_std
test_data_scaled = (test_data - pixel_mean) / pixel_std

# Archive the original set as uint8, to save space
print("Saving data and labels in  pickle file")
with open("fingers_dataset_scaled_resize_32_32.pickle", 'wb') as f:
      pickle.dump([train_data_scaled, train_labels, test_data_scaled, test_labels], \
                  f, pickle.HIGHEST_PROTOCOL)
