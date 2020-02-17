# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:09:21 2020

@author: Ian_b
"""

test = True

import sys
import numpy as numpy
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import time
#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import regularizers
from keras import models
# SKLEARN
from sklearn.model_selection import train_test_split





from keras.models import load_model
from keras.preprocessing import image
import numpy as np

numpy.set_printoptions(threshold=sys.maxsize)

model = models.load_model('/Users/Ian_b/Downloads/Crappy.h5')
#model = models.load_model('/Users/Ian_b/Downloads/Cnn2_30_epochs_08.02_regulariser_varying_0.0001.h5')
#model = models.load_model('/Users/Ian_b/Downloads/with_held_back_data.h5')
img = cv2.imread('/Users/Ian_b/Downloads/CNNClassifier-master/kc_nn_test.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')  # graph it
plt.show()


#print(img.shape)
img = cv2.resize(img,(200,200))
img = np.reshape(img,(-1, 200,200, 1))

img = img.astype('float32')
img = img/255 #normalise
prediction = model.predict(img)

rounded =[numpy.round(prediction, 3) for prediction in prediction]


print(rounded)

#print("Predictions: Class 1: {0:.10f} Class 2: {0:.10f} Class 3:{0:.10f}".format(prediction[0][0],prediction[0][1],prediction[0][2]))





#print(predictions_single)

#TODO Understand why the model is sometimes so certain and sometimes not