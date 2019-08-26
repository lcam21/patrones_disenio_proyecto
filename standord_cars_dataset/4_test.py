import os
import time
import scipy.io
import cv2 as cv
import keras.backend as K
import numpy as np
#from console_progressbar import ProgressBar

from utils import load_model

if __name__ == '__main__':
	model = load_model()
	
	cars_meta = scipy.io.loadmat('devkit/cars_meta')
	class_names = cars_meta['class_names']  # shape=(1, 196)
	class_names = np.transpose(class_names)

    #pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')
	print("Predicting test data...")
	num_samples = 8041
	start = time.time()
	out = open('result.txt', 'a')
	for i in range(num_samples):
		filename = os.path.join('data/test', '%05d.jpg' % (i + 1))
		bgr_img = cv.imread(filename)
		rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
		rgb_img = np.expand_dims(rgb_img, 0)
		preds = model.predict(rgb_img)
		prob = np.max(preds)
		class_id = np.argmax(preds)
		#out.write('{}\n'.format(str(class_id + 1)))
		out.write('label: %s - prob: %.2f' .format(class_names[class_id][0][0], prob))
		#pb.print_progress_bar((i + 1) * 100 / num_samples)

	end = time.time()
	seconds = end - start
	print('avg fps: {}'.format(str(num_samples / seconds)))

	out.close()
	K.clear_session()