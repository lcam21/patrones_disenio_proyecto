# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

from utils import load_model

if __name__ == '__main__':
	img_width, img_height = 224, 224
	model = load_model()
	#model.load_weights('models/model.76-0.90.hdf5')

	cars_meta = scipy.io.loadmat('devkit/cars_meta')
	class_names = cars_meta['class_names']  # shape=(1, 196)
	class_names = np.transpose(class_names)
	results = []
	results.append(["Prediction","Probability"])
	test_path = 'data/prediction/'
    
	
	
	####
	test_path = 'data/test/'
	test_images = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]

	num_samples =100
	samples = random.sample(test_images, num_samples)
	
	for i, image_name in enumerate(samples):
		filename = os.path.join(test_path, image_name)
		bgr_img = cv.imread(filename)
	####
	#i = -1
	#for filename in os.listdir(test_path):
	#	print('Start processing image: {}'.format(filename))
	#	i = i + 1
	#	bgr_img = cv.imread(test_path + "/" + filename)
	###
		bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
		rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
		rgb_img = np.expand_dims(rgb_img, 0)
		preds = model.predict(rgb_img)
		prob = np.max(preds)
		class_id = np.argmax(preds)
		text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
		print(text)
		results.append([class_names[class_id][0][0], prob])
	np.savetxt("results.csv", results, delimiter=",", fmt="%s")
	K.clear_session()  