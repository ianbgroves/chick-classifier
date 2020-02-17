# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 23:49:05 2020

@author: Ian_b
"""



#Import required libaries
import numpy as np
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




# load the trained convolutional neural network


from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images
img_width, img_height = 200, 200

# load the model we saved
print("[INFO] loading network...")
model = models.load_model('/Users/Ian_b/Downloads/Crappy.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

import numpy as np
from PIL import Image
from keras.preprocessing import image

#img = image.load_img('/Users/Ian_b/Downloads/CNNClassifier-master/test_hh10.jpg', target_size=(img_width, img_height))

#img = Image.open('/Users/Ian_b/Downloads/CNNClassifier-master/test_hh10.jpg')



img = cv2.imread('/Users/Ian_b/Downloads/CNNClassifier-master/10.3_test.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)


img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')  # graph it
plt.show()  # display!

img = image.img_to_array(img)
                     
#img = cv2.resize(img,(200,200))
img = np.reshape(img,(-1, 200,200, 1))

img = img.astype('float32')
img = img/255.0
prediction = model.predict(img)
#prediction = np.argmax(prediction) #argmax returns the position of the largest value
print("Predictions: Class 1: {0:.10f} Class 2: {0:.10f} Class 3:{0:.10f}".format(prediction[0][0],prediction[0][1],prediction[0][2]))



                                                                          
#img = image.img_to_array(img)
#
#img  = img.reshape((1,) + img.shape)
## img  = img/255
#img = np.reshape(img,[200,200,1])
#img_class=model.predict_classes(img) 
#
#prediction = img_class[0]
#classname = img_class[0]
#print("Class: ",classname)
#
## predicting images
#img = image.load_img('/Users/Ian_b/Downloads/CNNClassifier-master/test_hh10.jpg', target_size=(img_width, img_height,))
##img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to grayscale
#
#x = image.img_to_array(img)
##x = np.expand_dims(x, axis=0)
#x = x.reshape(1, 200, 200, 1)
## make a prediction
#y = model.predict_classes(x)
## show the inputs and predicted outputs
#print("X=%s, Predicted=%s" % (x[0], y[0]))





## classify the input image
#(10.1, 10.2, 10.3) = model.predict(image)[0]
#
#if (10.1 > 10.2): 
#     label = "10.1"
#elif (10.1 > 10.3):
#     label = "10.1"
#if (10.2 > 10.1): 
#     label = "10.2"
#elif (10.2 > 10.3):
#     label = "10.2"
#if (10.3 > 10.1): 
#     label = "10.3"
#elif (10.3> 10.2):
#     label = "10.3"    
#proba = 10.1 if 10.1 > 10.2 or 10.3 
#label = "{}: {:.2f}%".format(label, proba * 100)