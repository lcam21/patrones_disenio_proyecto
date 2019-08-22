import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model

model_saved = load_model('model_keras.h5')

img = mpimg.imread("testing_jpg/img_21.jpg")
array = np.array(img)
print(array.shape)

array = np.reshape(array, [-1,128,128,3])

#prediction = model_saved.predict(array)
prediction = model_saved.predict_classes(array)
print(prediction)
