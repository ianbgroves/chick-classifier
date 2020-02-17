# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 01:12:51 2020

@author: Ian_b
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:39 2020

@author: Ian_b
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:28:00 2019

@author: David furley.david@gmail.com
"""

#%%

#Import required libaries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

# SKLEARN
from sklearn.model_selection import train_test_split

#%%
#Directory location for image data - USE OWN DIRECTORY HERE
DATADIR = '/Users/Ian_b/Downloads/CNNClassifier-master/training_data'


#Sub directories for different categories
CATEGORIES = ["10_1","10_2","10_3"]

#%%
#Display one image to ensure images are loading correctly
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.figure('test')
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
#    
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

#print(y)
  
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
nb_classes = 3
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#%%

#variables for CNN training
#batch_size to train
batch_size = 32
# number of output classes
# moved to line 118

# number of epochs to train
#nb_epoch = 2

#iterate epoch values
#epoch_iterate = [10,20,30,40,50]
epoch = 30
# number of convolutional filters to use
nb_filters = [32]
# iterate through different values of convolutional filters
#iterate = 8
#nb_filters_iteration = [16,24,32,40,48]
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

count = 0

#history_list = []
#%%
#prepare model - I copied this from a model I found online, may be worth looking into this further,
#adding more layers and changing variables to see if I can create a better fitted model
#for iterate in nb_filters:

#for epoch in epoch_iterate:
        #initialize CNN | Convolution -> Pooling -> Flattening
model = Sequential()
#Convolution
model.add(Conv2D(32, (5, 5), activation ='relu',  input_shape=(img_rows, img_cols, 1)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation ='relu',  input_shape=(img_rows, img_cols, 1)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5), activation ='relu',  input_shape=(img_rows, img_cols, 1)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (5, 5), activation ='relu',  input_shape=(img_rows, img_cols, 1)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (5, 5), activation ='relu',  input_shape=(img_rows, img_cols, 1)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512, activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
#%%
#compile the model include 'metrics=['accuracy'] to give feedback in console
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#train the model

history =  model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch,
        verbose=1, validation_data=(X_test, Y_test))
#history_list.append(history)

#list data in history variable
print(history.history.keys())
name = " Accuracy, Epochs {}".format(epoch)
plt.figure(name)
#plot history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plot history for loss

name = " Loss, Epochs {}".format(epoch)
plt.figure(name)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#get and display model test score and accuracy
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
    

#    count += 1
#from keras.models import load_model
#
model.save('/Users/Ian_b/Downloads/Cnn2_30_epochs_06.02_b.h5')  # creates a HDF5 file 'my_model.h5'
##del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = load_model('/Users/Ian/Downloads/Cnn2.h5') #uncomment this to load model



#TODO: add functionality to save figures rather than show




