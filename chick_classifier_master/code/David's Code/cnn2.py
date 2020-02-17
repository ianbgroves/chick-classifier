#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:28:00 2019

@author: David
"""

#%%

#Import required libaries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

# SKLEARN
from sklearn.model_selection import train_test_split

#%%
#Directory location for image data - USE OWN DIRECTORY HERE
DATADIR = '/Users/David/Desktop/CNN-image-classifier/training_data'
#Sub directories for different categories
CATEGORIES = ["10_1","10_2","10_3"]

#%%
#Display one image to ensure images are loading correctly
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
    
print(img_array)
print(img_array.shape)

#%%
training_data = []

#add all images to training data
def create_training_data():
    for category in CATEGORIES:  # 10_1, 10_2, 10_3

        path = os.path.join(DATADIR,category)  # create path to different stages
        class_num = CATEGORIES.index(category)  # get the classification  (0, 1 or 2)

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

#check all images have been added to training data
print(len(training_data))

#%%
#shuffle training_data so that 10_1s, 10_2s and 10_3s are not together
import random

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])
    

#%%
X = [] #Images
y = [] #Labels

# input image dimensions
img_rows, img_cols = 200, 200

#split trainging data in images (X) and labels (y)
for features, label in training_data:
    X.append(features)
    y.append(label)
   
X = np.array(X).reshape(-1, img_rows, img_cols, 1)

# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#reshape X so the CNN can read it 
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

#check that X has been correctly split into train and test sets
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#%%

#variables for CNN training
#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 20

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
#prepare model - I copied this from a model I found online, may be worth looking into this further,
#adding more layers and changing variables to see if I can create a better fitted model
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols,1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

#%%
#compile the model include 'metrics=['accuracy'] to give feedback in console
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                verbose=1, validation_data=(X_test, Y_test))

#get and display model test score and accuracy
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])












